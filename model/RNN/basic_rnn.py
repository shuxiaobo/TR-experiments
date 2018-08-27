#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/21

import torch
from model.base_model import BaseModel

import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class BaseRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias = False, nonlinearity = "relu",
                 hidden_min_abs = 0, hidden_max_abs = None,
                 hidden_init = None, recurrent_init = None,
                 gradient_clip = 5):
        super(BaseRNNCell, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.hidden_init = hidden_init
        self.recurrent_init = recurrent_init
        if self.nonlinearity == "tanh":
            self.activation = F.tanh
        elif self.nonlinearity == "relu":
            self.activation = F.relu
        elif self.nonlinearity == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        self.weight_ih = Parameter(torch.eye(hidden_size, input_size))
        self.weight_hh = Parameter(torch.eye(hidden_size, hidden_size))
        self.weight_hh1 = Parameter(torch.eye(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.randn(hidden_size))
        else:
            self.register_parameter('bias_ih', None)

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "bias" in name:
                weight.data.zero_()
            elif "weight_hh" in name:
                if self.recurrent_init is None:
                    nn.init.constant_(weight, 1)
                else:
                    self.recurrent_init(weight)
            elif "weight_ih" in name:
                if self.hidden_init is None:
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.hidden_init(weight)
            else:
                weight.data.normal_(0, 0.01)
                # weight.data.uniform_(-stdv, stdv)
        self.check_bounds()

    def check_bounds(self):
        if self.hidden_min_abs:
            abs_kernel = torch.abs(self.weight_hh.data).clamp_(min = self.hidden_min_abs)
            self.weight_hh.data = self.weight_hh.mul(torch.sign(self.weight_hh.data), abs_kernel)
        if self.hidden_max_abs:
            self.weight_hh.data = self.weight_hh.clamp(max = self.hidden_max_abs, min = -self.hidden_max_abs)

    def forward(self, input, hx):
        return self.activation(F.linear(input, self.weight_ih, self.bias_ih) + torch.matmul(hx, self.weight_hh.matmul(self.weight_hh1)))


class BaseRNN(BaseModel):

    def __init__(self, args, input_size, hidden_size, n_layer = 1, batch_norm = False,
                 batch_first = False, bidirectional = False,
                 hidden_inits = None, recurrent_inits = None, truncate = False,
                 **kwargs):
        super(BaseRNN, self).__init__(args)
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.truncate = truncate
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        num_directions = 2 if self.bidirectional else 1

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        cells = []
        for i in range(n_layer):
            directions = []
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            in_size = input_size if i == 0 else hidden_size * num_directions
            for dir in range(num_directions):
                directions.append(BaseRNNCell(in_size, hidden_size, **kwargs))
            cells.append(nn.ModuleList(directions))
        self.cells = nn.ModuleList(cells)

        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns.append(nn.BatchNorm1d(hidden_size * num_directions))
            self.bns = nn.ModuleList(bns)

        h0 = torch.zeros(hidden_size * num_directions)
        self.register_buffer('h0', torch.autograd.Variable(h0))

    def forward(self, x, hidden = None):
        batch_norm = self.batch_norm
        time_index = self.time_index
        batch_index = self.batch_index
        num_directions = 2 if self.bidirectional else 1

        for i, directions in enumerate(self.cells):
            hx = self.h0.unsqueeze(0).expand(
                x.size(batch_index),
                self.hidden_size * num_directions).contiguous()

            x_n = []
            for dir, cell in enumerate(directions):
                hx_cell = hx[:, self.hidden_size * dir: self.hidden_size * (dir + 1)]
                cell.check_bounds()
                outputs = []
                hiddens = []
                x_T = torch.unbind(x, time_index)
                if dir == 1:
                    x_T = reversed(x_T)
                for k, x_t in enumerate(x_T):
                    hx_cell = cell(x_t, hx_cell)
                    if self.truncate and k % 3 == 0:
                        hx_cell = hx_cell.detach()
                    outputs.append(hx_cell)
                if dir == 1:
                    outputs = outputs[::-1]
                x_cell = torch.stack(outputs, time_index)
                x_n.append(x_cell)
                hiddens.append(hx_cell)
            x = torch.cat(x_n, -1)

            if batch_norm:
                if self.batch_first:
                    x = self.bns[i](x.permute(batch_index, 2, time_index).contiguous()).permute(0, 2, 1)
                else:
                    x = self.bns[i](x.permute(batch_index, 2, time_index).contiguous()).permute(2, 0, 1)
        return x.squeeze(2), torch.cat(hiddens, -1)
