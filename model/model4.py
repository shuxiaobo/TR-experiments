#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/2

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from tensorflow.contrib.keras.api.keras.preprocessing import sequence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class NGramConcatRNN(BaseModel):

    def __init__(self, args, hidden_size, embedding_size, vocabulary_size, rnn_layers = 1, bidirection = False, kernel_size = 3, stride = 1, num_class = 2):
        super(NGramConcatRNN, self).__init__(args = args)
        self.args = args
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocalbulary_size = vocabulary_size
        self.bidirection = bidirection
        self.kernel_size = kernel_size
        self.rnn_layers = rnn_layers
        self.stride = stride

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.rnn1 = nn.LSTM(embedding_size * 3, hidden_size = hidden_size, num_layers = rnn_layers, bias = False, bidirectional = bidirection,
                            batch_first = True)
        self.linear = nn.Linear(hidden_size * 2 if bidirection else hidden_size, num_class, bias = False)
        self.params = nn.Parameter(torch.eye(embedding_size * 3))
        self.dropout = nn.Dropout(p = self.args.keep_prob)
        self.param = nn.Parameter(torch.randn(self.stride, embedding_size))

    def forward(self, x, y):
        max_len = x.shape[1]
        if max_len % self.stride:
            max_len += self.stride - (max_len % self.stride)
            x = sequence.pad_sequences(x, maxlen = max_len, padding = 'post')
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)
        x_embed2 = self.embedding(torch.zeros_like(x).cuda() + torch.cat([torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda(), x[:, 1:]], -1))
        x_embed3 = self.embedding(torch.zeros_like(x).cuda() + torch.cat([x[:, :-1], torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda()], -1))
        x_embed = torch.cat([x_embed2, x_embed, x_embed3], -1)
        x_embed = x_embed @ self.params
        outputs, (h, c) = self.rnn1(x_embed)
        output_maxpooled, _ = torch.max(outputs, 1)
        # output_maxpooled = h.view(h.shape[1], -1)
        class_prob = self.linear(output_maxpooled)
        return class_prob, F.dropout(output_maxpooled)

    def gather_rnnstate(self, data, mask):
        """
        return the last state valid
        :param data:
        :param mask:
        :return:
        """
        real_len_index = torch.sum(mask, -1) - 1
        assert torch.max(real_len_index)[0].item() + 1 == data.shape[1]
        return torch.gather(data, 1, real_len_index.long().view(-1, 1).unsqueeze(2).repeat(1, 1, data.shape[-1])).squeeze()

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
        if self.param.norm(p = 2, dim = 0).mean() > 1:
            return F.cross_entropy(out, LongTensor(label)) + self.param.norm(p = 2, dim = 0).mean()
        return F.cross_entropy(out, LongTensor(label))
