#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/2
import os
import json
import torch
import logging
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.model_name = self.__class__.__name__  # model name

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

    def loss(self, prediction, label):
        raise NotImplementedError('Please implement the loss method in your model class. See models.base_models')

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        logging.info("Model {} preparing to load...".format(self.model_name))
        self.load_args(os.path.join(path, 'args.json'))
        self.load_state_dict(torch.load(path))
        logging.info("Model {} has been loaded...".format(self.model_name))

    def save(self, datetime, matric_str = ''):
        """
        保存模型，默认使用“checkpoints/{datetime}/模型名字”作为文件名
        """
        logging.info("Prepare to save model...")
        prefix = self.args.tmp_dir + 'checkpoints/'
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        logging.info("prefix path created : {}".format(prefix))
        path = os.path.join(prefix, datetime)
        if not os.path.exists(path):
            os.mkdir(path)
        logging.info("Entire path created : {}".format(path))
        self.save_args(path)
        name = os.path.join(path, self.model_name + '_' + matric_str + '.pth')
        logging.info("Save model to : {}".format(name))
        torch.save(self.state_dict(), name)
        logging.info("Save model over...")
        return name

    def save_args(self, path):
        file = os.path.join(path, 'args.json')
        with open(file, mode = 'w', encoding = 'utf-8') as f:
            json.dump(vars(self.args), f, sort_keys = True, indent = 4)
        logging.info('Save args over : %s' % file)

    def load_args(self, filename):
        if not filename:
            return
        json_dict = json.load(open(filename, encoding = "utf-8"))
        for k, v in json_dict.items():
            self.args.set_default(k, v)
        logging.info('Load args over : %s' % filename)

    def correct_prediction(self, pred, label):
        """
        return the indices of the batched data which are correct prediction
        :param pred:
        :param label:
        :return:
        """
        return torch.argmax(torch.eq(torch.argmax(pred, dim = -1), label), dim = -1)
