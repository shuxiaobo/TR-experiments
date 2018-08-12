#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/2
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta
import time


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.model_name = str(type(self))  # model name

    def init_weight(self):
        """
        implement the method to init the model parameters
        :return:
        """
        raise NotImplementedError('Please implement the init_weight method in your model class. See models.base_models')

    def init_optimizer(self):
        if self.args.optimizer.lower() == 'adam':
            self.optimizer = Adam(self.parameters(), lr = self.args.lr)
        elif self.args.optimizer.lower() == 'sgd':
            self.optimizer = SGD(self.parameters(), lr = self.args.lr, weight_decay = 0.99999)
        elif self.args.optimizer.lower() == 'adad':
            self.optimizer = Adadelta(self.parameters(), lr = self.args.lr)
        else:
            raise ValueError('No such optimizer implement.')

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name = None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '.pth'
        torch.save(self.state_dict(), name)
        return name
