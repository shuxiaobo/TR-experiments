#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/13
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


class BaseLineRNN(BaseModel):

    def __init__(self, args, hidden_size, embedding_size, vocabulary_size, rnn_layers = 1, bidirection = False, kernel_size = 3, stride = 1, num_class = 2):
        super(BaseLineRNN, self).__init__(args = args)
        self.args = args
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocalbulary_size = vocabulary_size
        self.bidirection = bidirection
        self.kernel_size = kernel_size
        self.stride = stride

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # self.rnn = nn.GRU(embedding_size, hidden_size, num_layers = rnn_layers, bias = False, bidirectional = bidirection, batch_first = True)
        self.rnn = BaseRNN(args, input_size = embedding_size, hidden_size = hidden_size, n_layer = rnn_layers, bidirectional = bidirection, batch_first = True)
        self.linear = nn.Linear(hidden_size * 2 if bidirection else hidden_size, num_class, bias = False)
        self.dropout = nn.Dropout(p = self.args.keep_prob)

    def forward(self, x, y):
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        # x_embed = self.embedding(x) * mask.unsqueeze(-1)  # here can not use the mask while using the last hidden state.
        x_embed = self.embedding(x)  # here can not use the mask while using the last hidden state.
        x_embed = self.dropout(x_embed)
        outputs, h = self.rnn(x_embed)
        # outputs, (h, c) = self.rnn(x_embed)
        batch_size = x.shape[0]
        h = h.transpose(0, 1).contiguous().view(batch_size, -1)
        outputs = gather_rnnstate(data = outputs, mask = mask)
        class_prob = self.linear(outputs)
        return class_prob, outputs

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.all_weights.data)
        nn.init.xavier_uniform_(self.linear.weight.data)

    def loss(self, out, label):
        return F.cross_entropy(out, LongTensor(label))
