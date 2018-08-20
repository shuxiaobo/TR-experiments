#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/20

import torch
from torch import nn
import torch.nn.functional as F
from .base_model import BaseModel

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class ClassifyNet(BaseModel):

    def __init__(self, args, input_size, num_cls = 3):
        super(ClassifyNet, self).__init__(args)
        self.classify = nn.Linear(input_size * 4, num_cls)

    def forward(self, x1, x2):
        features = torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), 1)

        return self.classify(features)

    def loss(self, pred, label):
        return F.cross_entropy(pred, LongTensor(label))
