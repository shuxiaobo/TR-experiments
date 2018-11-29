#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/2

import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class NGramRNN(BaseModel):

    def __init__(self, args, hidden_size, embedding_size, vocabulary_size, rnn_layers = 1, bidirection = False, kernel_size = 3, stride = 1, num_class = 2):
        super(NGramRNN, self).__init__(args = args)
        self.args = args
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocalbulary_size = vocabulary_size
        self.bidirection = bidirection
        self.kernel_size = kernel_size
        self.stride = stride
        self.rnn_layers = rnn_layers

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.rnn1 = nn.LSTM(embedding_size, hidden_size, num_layers = rnn_layers, bias = False, bidirectional = bidirection, batch_first = True)
        self.rnn2 = nn.LSTM(embedding_size + hidden_size * 2 if bidirection else embedding_size * 2, hidden_size, rnn_layers, bias = False,
                            bidirectional = bidirection,
                            batch_first = True)

        # self.linear = nn.Linear(embedding_size * 2 if bidirection else embedding_size, 2, bias = False)
        self.linear = nn.Linear(hidden_size * 2 if bidirection else hidden_size, num_class, bias = False)
        self.dropout = nn.Dropout(p = self.args.keep_prob)

    def forward(self, x, y):
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)
        length = x_embed.shape[1]
        # reduce
        outputs, h, reduced_mask = self.reduce_ngram(x_embed, mask)  # (seq_len, batch, hidden_size * num_directions)

        # expand
        expanded_out = self.expand(outputs, length)

        outputs, (h, c) = self.rnn2(torch.cat([expanded_out, x_embed], -1))
        outputs = outputs * mask.unsqueeze(-1)

        # output_maxpooled, _ = torch.max(outputs, 1)
        output_maxpooled = h.transpose(0, 1).contiguous().view(x.shape[0], -1)

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
        return torch.gather(data, 1, LongTensor(real_len).view(-1, 1).unsqueeze(2).repeat(1, 1, data.shape[-1]))

    def expand(self, data, length):
        result = []
        max_len = data.shape[1]
        for i in range(0, max_len - 1):
            result.append(data[:, i].unsqueeze(1).repeat(1, self.stride, 1))
        result.append(data[:, -1].unsqueeze(1).repeat(1, length - max_len * self.stride + self.stride, 1))
        result = torch.cat(result, 1)
        assert length == result.shape[1]
        return result

    def reduce_ngram(self, data, mask):
        data = data * mask.unsqueeze(-1)
        max_len = data.shape[1]

        reduced = []
        reduced_mask = []
        for i in range(0, max_len, self.stride):
            reduced.append(torch.sum(data[:, i:i + self.kernel_size, :], 1))
            reduced_mask.append(torch.max(mask[:, i:i + self.kernel_size], 1)[0])
        reduced = torch.stack(reduced, 1)
        reduced_mask = torch.stack(reduced_mask, 1)
        outputs, (h, c) = self.rnn1(reduced)
        return outputs, h, reduced_mask

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn1.all_weights.data)
        nn.init.xavier_uniform_(self.rnn2.all_weights.data)

    def loss(self, out, label):
        return F.cross_entropy(out, LongTensor(label))
