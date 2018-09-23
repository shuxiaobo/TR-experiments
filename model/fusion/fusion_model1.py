#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/19
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RNN.indrnn import IndRNN
from model.base_model import BaseModel
from utils.util import gather_rnnstate
from model.RNN.basic_rnn import BaseRNN
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class FusionModel(BaseModel):

    def __init__(self, args, vocabulary_size, embedding_size, hidden_size, num_layers, bidirection = True, num_class = 2):
        super(FusionModel, self).__init__(args)
        self.args = args
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.rnn = nn.ModuleList()

        self.rnn.append(nn.GRU(embedding_size, hidden_size, num_layers = 1, bias = False, bidirectional = bidirection, batch_first = True))
        for i in range(1, num_layers):
            self.rnn.append(nn.GRU(embedding_size + hidden_size + int(bidirection) * hidden_size, hidden_size, num_layers = 1, bias = False,
                                   bidirectional = bidirection, batch_first = True))
        self.linear = nn.Linear(hidden_size * self.num_layers * 2 if bidirection else hidden_size * self.num_layers, num_class, bias = False)
        self.dropout = nn.Dropout(self.args.keep_prob)

        self.ws = list()
        for i in range(num_layers):
            self.ws.append(nn.Parameter(torch.randn(embedding_size, 2 * hidden_size)).cuda())

    def forward(self, x, y):
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x) * mask.unsqueeze(-1)  # here can not use the mask while using the last hidden state.
        # x_embed = self.embedding(x)  # here can not use the mask while using the last hidden state.
        x_embed = self.dropout(x_embed)

        rnn_inputs = x_embed
        rnn_out_concat = list()
        for i in range(self.num_layers):
            rnn_inputs = torch.relu(rnn_inputs)
            rnn_out, rnn_last_states = self.rnn[i](rnn_inputs)
            rnn_out_concat.append(rnn_out)
            # att = torch.softmax(torch.squeeze(torch.matmul(torch.matmul(x_embed, self.ws[i]),
            #                                                rnn_last_states.view(rnn_last_states.shape[1], rnn_last_states.shape[0] * rnn_last_states.shape[2],
            #                                                                     1)), -1), -1)
            # rnn_inputs = torch.cat([x_embed * torch.unsqueeze(att, -1), rnn_out], -1)
            rnn_inputs = torch.cat([x_embed, rnn_out_concat[0]], -1)
        outputs = torch.cat(rnn_out_concat, -1)

        # last outputs
        outputs = gather_rnnstate(data = outputs, mask = mask)

        # max pooled
        # outputs, _ = torch.max(outputs, 1)
        class_prob = self.linear(outputs)
        return class_prob, outputs

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.all_weights.data)
        nn.init.xavier_uniform_(self.linear.weight.data)

    def loss(self, out, label):
        return F.cross_entropy(out, LongTensor(label))
