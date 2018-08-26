#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/24
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RNN.basic_rnn import BaseRNN
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def train():
    rnn = nn.LSTM(10, 10)
    emb = nn.Embedding(10, 10)
    inputs = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]).long()

    embd = emb