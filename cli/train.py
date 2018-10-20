#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
from __future__ import absolute_import
import torch
import time
import logging
import datetime
import numpy as np
from datasets.binary import *
from datasets.snli import SNLI
from utils.util import accuracy
from model.model1 import NGramRNN
from model.model2 import NGramRNN2
from model.model3 import NGramRNN3
from model.model4 import NGramConcatRNN
from datasets.imdb import IMDB
from model.baseline import BaseLineRNN
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model.fusion.fusion_model1 import FusionModel
from model.fusion.fusion_ngram import FusionNGramRNN

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def train(args):
    train_dataloader, test_dataloader, model = init_from_scrach(args)
    best_acc = 0.0
    best_epoch = 0
    iter = 0
    logger('Begin training...')
    logger_path = '../logs/log-av%s-%s-model%s-emb%d-id%s' % (
        args.activation, args.dataset, model.__class__.__name__, args.embedding_dim, str(datetime.datetime.now()))
    logger('Save log to %s' % logger_path)
    writer = SummaryWriter(log_dir = logger_path)
    for i in range(args.num_epoches):
        loss_sum = 0
        acc_sum = 0.0
        samples_num = 0
        for j, a_data in enumerate(train_dataloader):
            iter += 1
            model.optimizer.zero_grad()
            model.zero_grad()
            out, feature = model(*a_data)
            loss = model.loss(out, a_data[-1])
            loss.backward()
            if args.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            model.optimizer.step()
            loss_sum += loss.item()

            samples_num += len(a_data[0])
            acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])

            writer.add_scalar('loss', loss_sum / (j + 1), iter)
            writer.add_scalar('accuracy', acc_sum / samples_num, iter)

            if (j + 1) % args.print_every_n == 0:
                logging.info('train: Epoch = %d | iter = %d/%d | ' %
                             (i, j, len(train_dataloader)) + 'loss sum = %.2f | accuracy : %.4f' % (
                                 loss_sum * 1.0 / j, acc_sum / samples_num))

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
                        writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), j)

        logging.info("Testing...... | Model : {0} | Task : {1}".format(model.__class__.__name__, train_dataloader.dataset.__class__.__name__))
        testacc = test(args, model, test_dataloader)
        if best_acc < testacc and train_dataloader.dataset.__class__.__name__ == 'Kaggle':
            logger('Generate the result for sumbmission...')
            test_kaggle(args, model, train_dataloader.dataset.word2id)
        best_epoch = best_epoch if best_acc > testacc else i
        best_acc = best_acc if best_acc > testacc else testacc
        logging.error('Test result acc1: %.4f | best acc: %.4f | best epoch : %d' % (testacc, best_acc, best_epoch))


def test_kaggle(args, model, word2id):
    model.eval()

    def load_file(args):
        path = os.path.join(args.tmp_dir, 'Kaggle', 'r_test.txt')
        with io.open(path, 'r', encoding = 'utf-8') as f:
            data_x = list()
            data_id = list()
            f.readline()
            for i, line in enumerate(f.read().splitlines()):
                line = line.strip().split('\t')
                data_id.append(line[0])
                if len(line) < 3:
                    data_x.append([' '])
                else:
                    data_x.append(line[2].split(' '))
        return data_x, data_id

    def toid(sample):
        return [word2id[d] if word2id.get(d) else word2id['_<UNKNOW>'] for d in sample]

    result = []
    datax, dataid = load_file(args)
    for i in range(0, len(datax), args.batch_size):
        x = datax[i:i + args.batch_size]
        x = [toid(d) for d in x]
        max_len = max([len(dd) for dd in x])
        x = sequence.pad_sequences(x, maxlen = max_len, padding = 'post')
        out, _ = model(x = x, y = None)

        result.extend(out.argmax(-1).data.cpu().numpy().tolist())

    with open(os.path.join(args.tmp_dir, 'Kaggle', 'submission.tsv'), 'w+') as f:
        f.write('PhraseId' + '\t' + 'Sentiment' + '\n')
        for i, d in zip(dataid, result):
            f.write(i + '\t' + str(d) + '\n')
    model.train()


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
    if args.dataset.lower() == 'cr':
        train_dataset = CR(args)
        test_dataset = CR(args, is_train = False)
    elif args.dataset.lower() == 'mr':
        train_dataset = MR(args)
        test_dataset = MR(args, is_train = False)
    elif args.dataset.lower() == 'sst':
        train_dataset = SST(args, nclasses = 2)
        test_dataset = SST(args, is_train = False, nclasses = 2)
    elif args.dataset.lower() == 'mpqa':
        train_dataset = MPQA(args)
        test_dataset = MPQA(args, is_train = False)
    elif args.dataset.lower() == 'subj':
        train_dataset = SUBJ(args)
        test_dataset = SUBJ(args, is_train = False)
    elif args.dataset.lower() == 'trec':
        train_dataset = TREC(args)
        test_dataset = TREC(args, is_train = False)
    elif args.dataset.lower() == 'kaggle':
        train_dataset = Kaggle(args)
        test_dataset = Kaggle(args, is_train = False)
    elif args.dataset.lower() == 'imdb':
        train_dataset = IMDB(args, num_words = args.num_words, skip_top = args.skip_top)
        test_dataset = IMDB(args, train = False, num_words = args.num_words, skip_top = args.skip_top)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = IMDB.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Train data max length : %d' % train_dataset.max_len)

    logging.info('Load the test dataset...')
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,
                                 collate_fn = IMDB.batchfy_fn, pin_memory = True, drop_last = False)
    logging.info('Test data max length : %d' % test_dataset.max_len)

    logging.info('Initiating the model...')
    # model = FusionModel(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
    #                     num_layers = args.num_layers,
    #                     bidirection = args.bidirectional, num_class = train_dataset.num_class)
    model = NGramConcatRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
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
