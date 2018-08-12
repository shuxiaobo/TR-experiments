#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.base_model import BaseModel
from torch.optim import Adam, Adadelta, SGD

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class PCNN_ONE(BaseModel):
    '''
    Zeng 2015 DS PCNN
    '''

    def __init__(self, args, pos_size, pos_dim, vocab_size, word_dim, filters_num, filters, use_pcnn, rel_num, drop_out, norm_emb, w2v_path, p1_2v_path, p2_2v_path):
        """

        :param pos_size:
        :param pos_dim:
        :param word_dim:
        :param filters_num:
        :param filters:
        :param use_pcnn:
        :param rel_num:
        :param drop_out:
        :param norm_emb:
        :param w2v_path:
        :param p1_2v_path:
        :param p2_2v_path:
        """
        super(PCNN_ONE, self).__init__(args = args)
        self.pos_size = pos_size
        self.pos_dim = pos_dim
        self.word_dim = word_dim
        self.filters_num = filters_num
        self.filters = filters
        self.use_pcnn = use_pcnn
        self.rel_num = rel_num
        self.drop_out = drop_out
        self.norm_emb = norm_emb
        self.rel_num = rel_num
        self.w2v_path = w2v_path
        self.p1_2v_path = p1_2v_path
        self.p2_2v_path = p2_2v_path
        self.vocab_size = vocab_size

        self.model_name = 'PCNN_ONE'

        self.pos1_embs = nn.Embedding(self.pos_size, self.pos_dim)
        self.pos2_embs = nn.Embedding(self.pos_size, self.pos_dim)

        feature_dim = self.word_dim + self.pos_dim * 2

        # for more filter size
        self.convs = nn.ModuleList([nn.Conv2d(1, self.filters_num, (k, feature_dim), padding = (int(k / 2), 0)) for k in self.filters])

        all_filter_num = self.filters_num * len(self.filters)

        if self.use_pcnn:
            all_filter_num = all_filter_num * 3

        self.linear = nn.Linear(all_filter_num, self.rel_num)
        self.dropout = nn.Dropout(self.drop_out)

        self.init_model_weight()
        self.criterion = nn.CrossEntropyLoss()
        # self.init_word_emb()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        for conv in self.convs:
            nn.init.xavier_uniform(conv.weight)
            nn.init.constant(conv.bias, 0.0)

        nn.init.xavier_uniform(self.linear.weight)
        nn.init.constant(self.linear.bias, 0.0)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.w2v_path)
        p1_2v = p_2norm(self.p1_2v_path)
        p2_2v = p_2norm(self.p2_2v_path)

        self.pos1_embs.weight.data.copy_(p1_2v.cuda())
        self.pos2_embs.weight.data.copy_(p2_2v.cuda())

    def init_optimizer(self):
        if self.args.optimizer2.lower() == 'adam':
            self.optimizer = Adam(self.parameters(), lr = self.args.lr)
        elif self.args.optimizer2.lower() == 'sgd':
            self.optimizer = SGD(self.parameters(), lr = self.args.lr, weight_decay = 0.99999)
        elif self.args.optimizer2.lower() == 'adad':
            self.optimizer = Adadelta(self.parameters(), lr = self.args.lr)
        else:
            raise ValueError('No such optimizer implement.')

    def piece_max_pooling(self, x, insPool):
        '''
        simple version piecewise
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = insPool
        batch_res = []

        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
            pool = split_pool[i]  # 2
            pool.sort()
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)  # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.filters_num
        return out

    def loss(self, prediction, label):
        return self.criterion(prediction, Variable(LongTensor(label)))

    def forward(self, ent1, ent2, word_emb, insPF1, insPF2, insPool):
        # word_emb, insPF1, insPF2, insPool = x
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1)
        x = self.dropout(x)

        x = [F.tanh(conv(x)).squeeze(3) for conv in self.convs]

        if self.use_pcnn:
            x = [self.piece_max_pooling(i, insPool) for i in x]
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.linear(x)

        return x
