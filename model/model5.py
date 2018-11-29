#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Desc: 
    @Author: shane 
    @Contact: iamshanesue@gmail.com
    @Software: PyCharm
    @Since: Python3.6
    @Date: 2018/11/27
    @All right reserved
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from model.layers import SelfAttention

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class NGramSumRNN(BaseModel):

    def __init__(self, args, hidden_size, embedding_size, vocabulary_size, rnn_layers = 1, bidirection = False, kernel_size = 3, stride = 1, num_class = 2):
        super(NGramSumRNN, self).__init__(args = args)
        self.args = args
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocalbulary_size = vocabulary_size
        self.bidirection = bidirection
        self.kernel_size = kernel_size
        self.rnn_layers = rnn_layers
        self.stride = stride

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.pos_embedding = nn.Embedding(5, embedding_size)
        self.pos_embedding2 = nn.Embedding(5, embedding_size)
        self.rnn1 = nn.LSTM(embedding_size * 3, hidden_size = hidden_size, num_layers = rnn_layers, bias = False, bidirectional = bidirection,
                            batch_first = True)
        self.linear = nn.Linear(hidden_size * 2 if bidirection else hidden_size, num_class, bias = False)
        self.params = nn.Parameter(torch.eye(embedding_size * 3))
        self.dropout = nn.Dropout(p = self.args.keep_prob)
        self.param = nn.Parameter(torch.eye(embedding_size, embedding_size))
        self.param2 = nn.Parameter(torch.eye(embedding_size, embedding_size))
        self.atten = SelfAttention(args, embedding_size * 3, 3)
        self.atten2 = SelfAttention(args, embedding_size * 5, 5)

    def forward(self, x, y):
        max_len = x.shape[1]
        if max_len % self.stride:
            max_len += self.stride - (max_len % self.stride)
            x = sequence.pad_sequences(x, maxlen = max_len, padding = 'post')
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)
        # x_embed2 = self.embedding(torch.cat([x[:, :1], x[:, :-1]], -1))
        # x_embed3 = self.embedding(torch.cat([x[:, 1:], x[:, :1]], -1))
        # x_embed4 = self.embedding(torch.cat([x[:, -2:], x[:, :-2]], -1))
        # x_embed5 = self.embedding(torch.cat([x[:, 2:], x[:, :2]], -1))

        x_embed2 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda(), x[:, :-1]], -1))
        x_embed3 = self.embedding(torch.cat([x[:, 1:], torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda()], -1))
        x_embed4 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda(), x[:, :-2]], -1))
        x_embed5 = self.embedding(torch.cat([x[:, 2:], torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda()], -1))
        #
        batch_size = x.shape[0]
        pos1 = self.pos_embedding(LongTensor([0, 1, 2]))
        pos2 = self.pos_embedding2(LongTensor([3, 0, 1, 2, 4]))

        ngram3 = torch.stack([x_embed2, x_embed, x_embed3], -2)
        ngram5 = torch.stack([x_embed4, x_embed2, x_embed, x_embed3, x_embed5], -2)
        ngram3 = F.softmax(torch.sum(ngram3 * pos1, -1), -2).unsqueeze(-1) * ngram3
        ngram5 = F.softmax(torch.sum(ngram5 * pos2, -1), -2).unsqueeze(-1) * ngram5

        # x_embed = torch.cat([torch.sum(ngram3, 2).squeeze(2), torch.sum(ngram5, 2).squeeze(2), x_embed], -1) * mask.unsqueeze(2)
        x_embed = torch.cat([ngram3.max(2)[0], ngram5.max(2)[0], x_embed], -1)

        outputs, (h, c) = self.rnn1(x_embed)
        output_maxpooled, _ = torch.max(outputs, 1)
        # output_maxpooled = self.gather_rnnstate(outputs, mask).squeeze(1)
        class_prob = self.linear(output_maxpooled)
        return class_prob, F.dropout(output_maxpooled)

    def gather_rnnstate(self, data, mask):
        """
        return the last state valid
        :param data:
        :param mask:
        :return:
        """
        real_len = torch.sum(mask, -1)
        return torch.gather(data, 1, real_len.long().view(-1, 1).unsqueeze(2).repeat(1, 1, data.shape[-1]))

    def expand(self, data, length):
        result = []
        max_len = data.shape[1]
        for i in range(0, max_len - 1):
            result.append(data[:, i].unsqueeze(1).repeat(1, self.kernel_size, 1))
        result.append(data[:, -1].unsqueeze(1).repeat(1, length - max_len * self.kernel_size + self.kernel_size, 1))
        result = torch.cat(result, 1)
        return result

    def reduce_ngram(self, data, mask):
        data = data
        max_len = data.shape[1]

        reduced = []
        reduced_mask = []
        for i in range(0, max_len, self.stride):
            reduced.append(torch.mul(data[:, i:i + self.kernel_size, :], self.param).sum(1))
            reduced_mask.append(torch.max(mask[:, i:i + self.kernel_size], 1)[0])
        reduced = torch.stack(reduced, 1)
        reduced_mask = torch.stack(reduced_mask, 1)
        outputs, (h, c) = self.rnn1(reduced)
        assert outputs.shape[1] == reduced_mask.sum(-1).max()[0].int().item()
        return outputs * reduced_mask.unsqueeze(-1), h, reduced_mask

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn1.all_weights.data)
        nn.init.xavier_uniform_(self.rnn2.all_weights.data)

    def loss(self, out, label):
        # if self.param.norm(p = 2, dim = 0).mean() > 1:
        #     return F.cross_entropy(out, LongTensor(label)) + self.param.norm(p = 2, dim = 0).mean()
        return F.cross_entropy(out, LongTensor(label))
