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
from model.model3 import NGramRNN3
from model.baseline import BaseLineRNN
from utils.util import accuracy
import logging
from datasets.binary import *
from datasets.snli import SNLI

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def train(args):
    train_dataloader, test_dataloader, model = init_from_scrach(args)
    best_acc = 0.0
    best_epoch = 0
    logger('Begin training...')
    for i in range(args.num_epoches):
        loss_sum = 0
        acc_sum = 0.0
        samples_num = 0
        for j, a_data in enumerate(train_dataloader):
            model.optimizer.zero_grad()

            out, feature = model(*a_data)
            loss = model.loss(out, a_data[-1])
            loss.backward()
            if args.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            model.optimizer.step()

            loss_sum += loss.item()

            samples_num += len(a_data[0])
            acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
            if (j + 1) % args.print_every_n == 0:
                logging.info('train: Epoch = %d | iter = %d/%d | ' %
                             (i, j, len(train_dataloader)) + 'loss sum = %.2f | accuracy : %.4f' % (
                                 loss_sum * 1.0 / j, acc_sum / samples_num))

        logging.info("Testing...... | Model : {0} | Task : {1}".format(model.__class__.__name__, train_dataloader.dataset.__class__.__name__))
        testacc = test(args, model, test_dataloader)
        best_epoch = best_epoch if best_acc > testacc else i
        best_acc = best_acc if best_acc > testacc else testacc
        logging.error('Test result acc1: %.4f | best acc: %.4f | best epoch : %d' % (testacc, best_acc, best_epoch))


def evaluation(args, model, data_loader):
    model.eval()
    samples_num = 0
    acc_sum = 0.0

    for j, a_data in enumerate(data_loader):
        out, _ = model(*a_data)

        samples_num += len(a_data[0])
        acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
    model.train()
    return acc_sum / samples_num


def test(args, model1, data_loader):
    return evaluation(args, model1, data_loader)


def dev(args, model1, data_loader):
    return evaluation(args, model1, data_loader)


def get_batch(args):
    """Generate the adding problem dataset"""
    # Build the first sequence
    TIME_STEPS = 100
    BATCH_SIZE = args.batch_size
    add_values = np.random.rand(args.batch_size, TIME_STEPS)

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = np.zeros_like(add_values)
    half = int(TIME_STEPS / 2)
    for i in range(BATCH_SIZE):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, TIME_STEPS)
        add_indices[i, [first_half, second_half]] = 1

    # Zip the values and indices in a third dimension:
    # inputs has the shape (batch_size, time_steps, 2)
    inputs = np.dstack((add_values, add_indices))
    targets = np.sum(np.multiply(add_values, add_indices), axis = 1)
    return inputs, targets


def init_from_scrach(args):
    logging.info('No trained model provided. init model from scratch...')

    logging.info('Load the train dataset...')
    # train_dataset = CR(args)
    # test_dataset = CR(args, is_train = False)
    # train_dataset = MR(args)
    # test_dataset = MR(args, is_train = False)
    # train_dataset = SST(args, nclasses = 2)
    # test_dataset = SST(args, is_train = False, nclasses = 2)
    # train_dataset = MPQA(args)
    # test_dataset = MPQA(args, is_train = False)
    # train_dataset = SUBJ(args)
    # test_dataset = SUBJ(args, is_train = False)
    # train_dataset = TREC(args)
    # test_dataset = TREC(args, is_train = False)
    train_dataset = Kaggle(args)
    test_dataset = Kaggle(args, is_train = False)
    # train_dataset = ImdbDataSet(args, num_words = args.num_words, skip_top = args.skip_top)
    # test_dataset = ImdbDataSet(args, train = False, num_words = args.num_words, skip_top = args.skip_top)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = ImdbDataSet.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Train data max length : %d' % train_dataset.max_len)

    logging.info('Load the test dataset...')
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,
                                 collate_fn = ImdbDataSet.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Test data max length : %d' % test_dataset.max_len)

    logging.info('Initiating the model...')
    model = BaseLineRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
                        rnn_layers = args.num_layers,
                        bidirection = args.bidirectional, kernel_size = args.kernel_size, stride = args.stride, num_class = train_dataset.num_class)
    model.cuda()
    model.init_optimizer()
    logging.info('Model {} initiate over...'.format(model.__class__.__name__))
    logger(model)
    return train_dataloader, test_dataloader, model

#
# if __name__ == '__main__':
#     train()
