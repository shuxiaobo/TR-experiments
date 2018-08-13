#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
from __future__ import absolute_import
from torch.utils.data import DataLoader
import torch
import numpy as np
from datasets.imdb import ImdbDataSet
from model.model1 import NGramRNN
from model.model2 import NGramRNN2
from model.baseline import BaseLineRNN
from utils.util import accuracy
import logging

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def train(args):
    train_dataloader, test_dataloader, model = init_from_scrach(args)
    for i in range(args.num_epoches):
        loss_sum = 0
        acc_sum = 0.0
        samples_num = 0
        for j, a_data in enumerate(train_dataloader):
            model.optimizer.zero_grad()

            out = model(*a_data)
            loss = model.loss(out, a_data[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            model.optimizer.step()

            loss_sum += loss.item()

            samples_num += len(a_data[0])
            acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
            if (j + 1) % args.print_every_n == 0:
                logging.info('train: Epoch = %d | iter = %d/%d | ' %
                             (i, j, len(train_dataloader)) + 'loss sum = %.2f | accuracy : %.4f' % (
                                 loss_sum * 1.0 / j, acc_sum / samples_num))

        logging.info("Testing......")
        logging.error('Test result acc1: %.4f' % test(args, model, test_dataloader))


def evaluation(args, model, data_loader):
    model.eval()
    samples_num = 0
    acc_sum = 0.0

    for j, a_data in enumerate(data_loader):
        out = model(*a_data)

        samples_num += len(a_data[0])
        acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
    model.train()
    return acc_sum / samples_num


def test(args, model1, data_loader):
    return evaluation(args, model1, data_loader)


def dev(args, model1, data_loader):
    return evaluation(args, model1, data_loader)


def init_from_scrach(args):
    logging.info('No trained model provided. init model from scratch...')

    logging.info('Load the train dataset...')
    train_dataset = ImdbDataSet(args, num_words = args.num_words, skip_top = args.skip_top)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True,
                                  collate_fn = ImdbDataSet.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Load the test dataset...')
    test_dataset = ImdbDataSet(args, train = False, num_words = args.num_words, skip_top = args.skip_top)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True,
                                 collate_fn = ImdbDataSet.batchfy_fn, pin_memory = True, drop_last = False)

    model = BaseLineRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
                     rnn_layers = 1,
                     bidirection = True, kernel_size = args.kernel_size, stride = args.stride)

    model.cuda()
    model.init_optimizer()

    return train_dataloader, test_dataloader, model

#
# if __name__ == '__main__':
#     train()
