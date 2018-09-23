#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/20

'''
SNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np
from torch.utils.data import Dataset
from utils.util import *
from tensorflow.contrib.keras.api.keras.preprocessing import sequence


class SNLI(Dataset):
    def __init__(self, args, is_train = True, nclasses = 3, seed = 11111, file_name = ''):
        super(SNLI, self).__init__()
        logging.debug('***** Transfer task : SNLI Entailment*****\n\n')
        self.seed = seed
        self.tag_dict = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        self.seed = seed
        self.args = args
        self.num_class = nclasses
        self.max_len = 0
        file = self.args.train_file if is_train else self.args.test_file
        self.data_x, self.data_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + file))

        sorted_corpus = sorted(zip(self.data_x, self.data_y),
                               key = lambda z: (len(z[0][0]), len(z[0][1]), z[1]))
        self.data_x = [x for (x, y) in sorted_corpus]
        self.data_y = [y for (x, y) in sorted_corpus]


        if args.debug:
            self.data_y = self.data_y[:1000]
            self.data_x = self.data_x[:1000]
        self.n_samples = len(self.data_x)
        self.word_file = os.path.join(args.tmp_dir, self.__class__.__name__, file_name + args.word_file)
        if os.path.isfile(self.word_file) and os.path.getsize(self.word_file) > 0:
            self.word2id = self.get_word_index(self.word_file)
        else:
            self.word2id = self.prepare_dict(self.word_file)

    def load_file(self, fpath):
        with io.open(fpath, 'r', encoding = 'utf-8') as f:
            data_x = list()
            data_y = list()
            for line in f:
                line = line.strip().split(' ')
                line2 = f.readline().strip().split(' ')
                data_x.append((line, line2))
                data_y.append(self.tag_dict[f.readline().strip()])
                self.max_len = max([len(line), len(line2), self.max_len])
        return data_x, data_y

    def prepare_dict(self, file_name):
        logger("Prepare the dictionary for the {}...".format(self.__class__.__name__))
        word2id = prepare_dictionary(data = [x[0] for x in self.data_x], dict_path = file_name)
        word2id = prepare_dictionary(data = [x[1] for x in self.data_x], dict_path = file_name, word2id = word2id)
        logger("Word2id size : %d" % len(word2id))
        return word2id

    def get_word_index(self, path = None):
        if not path:
            path = self.args.tmp_dir + self.__class__.__name__ + self.args.word_file
        word2id = dict()
        with open(path, mode = 'r', encoding = 'utf-8') as f:
            for l in f:
                word2id.setdefault(l.strip(), len(word2id))
        logger('Word2id size : %d' % len(word2id))
        return word2id

    def __getitem__(self, index):
        result = [self.word2id[d] if self.word2id.get(d) else self.word2id['_<UNKNOW>'] for d in self.data_x[index][0]]
        result2 = [self.word2id[d] if self.word2id.get(d) else self.word2id['_<UNKNOW>'] for d in self.data_x[index][1]]
        return result, result2, self.data_y[index]

    def __len__(self):
        return self.n_samples

    @staticmethod
    def batchfy_fn(data):
        x2 = [d[1] for d in data]
        x1 = [d[0] for d in data]
        y = [d[2] for d in data]
        max_len1 = max([len(x) for x in x1])
        max_len2 = max([len(x) for x in x2])

        return sequence.pad_sequences(x1, maxlen = max_len1, padding = 'post'), sequence.pad_sequences(x2, maxlen = max_len2, padding = 'post'), y
