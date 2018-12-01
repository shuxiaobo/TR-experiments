#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/20
from __future__ import absolute_import
from torch.utils.data import DataLoader
import os
import time
import torch
import numpy as np
from model.model1 import NGramRNN
from model.model2 import NGramRNN2
from model.model3 import NGramRNN3
from model.model4 import NGramConcatRNN
from model.baseline import BaseLineRNN
from model.model5 import NGramSumRNN
from utils.util import accuracy
from datasets.binary import *
from datasets.snli import SNLI
from model.layers import ClassifyNet
from tensorboardX import SummaryWriter
from model.fusion.fusion_model1 import FusionModel
from model.fusion.fusion_ngram import FusionNGramRNN

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def train(args):
    train_dataloader, test_dataloader, model, model2 = init_from_scrach(args)
    best_acc = 0.0
    best_epoch = 0
    iter = 0
    logger_path = os.path.join(args.log_dir, time.strftime('%Y%m%d_%H:%M:%S'))
    logger('Save log to %s' % logger_path)
    writer = SummaryWriter(log_dir = logger_path)
    for i in range(args.num_epoches):
        loss_sum = 0
        acc_sum = 0.0
        samples_num = 0
        for j, a_data in enumerate(train_dataloader):
            iter += 1
            model.optimizer.zero_grad()
            _, feature = model(a_data[0], a_data[-1])
            _, feature2 = model(a_data[1], a_data[-1])
            out = model2(feature, feature2)
            loss = model2.loss(out, a_data[-1])
            loss.backward()
            if args.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            model.optimizer.step()
            loss_sum += loss.item()
            samples_num += len(a_data[0])
            acc = accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
            acc_sum += acc
            writer.add_scalar('epoch%d/loss' % i, loss_sum / (j + 1), iter)
            writer.add_scalar('epoch%d/accuracy' % i, acc_sum / samples_num, iter)

            if (j + 1) % args.print_every_n == 0:
                logging.info('train: Epoch = %d | iter = %d/%d | ' %
                             (i, j, len(train_dataloader)) + 'loss sum = %.4f | accuracy : %.4f' % (
                                 loss_sum * 1.0 / j, acc_sum / samples_num))
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
                        writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), j)

        logging.info("Testing...... | Model : {0} | Task : {1}".format(model.__class__.__name__, train_dataloader.dataset.__class__.__name__))
        testacc = test(args, model, model2, test_dataloader)
        best_epoch = best_epoch if best_acc > testacc else i
        best_acc = best_acc if best_acc > testacc else testacc
        logging.error('Test result acc1: %.4f | best acc: %.4f | best epoch : %d' % (testacc, best_acc, best_epoch))
    return [best_acc, train_dataloader.dataset.__class__.__name__, best_epoch]

def evaluation(args, model, model2, data_loader):
    model.eval()
    samples_num = 0
    acc_sum = 0.0

    for j, a_data in enumerate(data_loader):
        _, feature = model(a_data[0], a_data[-1])
        _, feature2 = model(a_data[1], a_data[-1])
        out = model2(feature, feature2)
        samples_num += len(a_data[0])
        acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
    model.train()
    return acc_sum / samples_num


def test(args, model1, model2, data_loader):
    return evaluation(args, model1, model2, data_loader)


def dev(args, model1, model2, data_loader):
    return evaluation(args, model1, model2, data_loader)


def init_from_scrach(args):
    logging.info('No trained model provided. init model from scratch...')

    logging.info('Load the train dataset...')

    train_dataset = SNLI(args)
    test_dataset = SNLI(args, is_train = False)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = SNLI.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Train data max length : %d' % train_dataset.max_len)

    logging.info('Load the test dataset...')
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,
                                 collate_fn = SNLI.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Test data max length : %d' % test_dataset.max_len)

    logging.info('Initiating the model...')
    # model = BaseLineRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
    #                     rnn_layers = args.num_layers,
    #                     bidirection = args.bidirectional, kernel_size = args.kernel_size, stride = args.stride, num_class = train_dataset.num_class)
    # model = NGramRNN3(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
    #                     rnn_layers = args.num_layers,
    #                     bidirection = args.bidirectional, kernel_size = args.kernel_size, stride = args.stride, num_class = train_dataset.num_class)
    # model = FusionModel(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
    #                     num_layers = args.num_layers,
    #                     bidirection = args.bidirectional, num_class = train_dataset.num_class)
    model = NGramSumRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
                           rnn_layers = args.num_layers,
                           bidirection = args.bidirectional, kernel_size = args.kernel_size, stride = args.stride, num_class = train_dataset.num_class)
    model.cuda()
    model.init_optimizer()
    model2 = ClassifyNet(args = args, input_size = args.hidden_size * 2 if args.bidirectional else args.hidden_size, num_cls = train_dataset.num_class)
    model2.cuda()
    model2.init_optimizer()
    logging.info('Model {} initiate over...'.format(model.__class__.__name__))
    logger(model)
    return train_dataloader, test_dataloader, model, model2

#
# if __name__ == '__main__':
#     train()
