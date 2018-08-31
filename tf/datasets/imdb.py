#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/6
from __future__ import absolute_import
import json
import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from torch.utils.data import Dataset
import logging
import os


class IMDB(Dataset):
    def __init__(self, args, train = True,
                 num_words = None,
                 skip_top = 0,
                 maxlen = None,
                 start_char = 1,
                 oov_char = 2,
                 index_from = 3, num_class = 2):
        self.args = args
        (train_data_x, train_data_y), (test_data_x, test_data_y) = self.load_data(
            path = os.getcwd() + '/' + self.args.tmp_dir + self.__class__.__name__ + '/imdb.npz',
            num_words = num_words,
            skip_top = skip_top, maxlen = maxlen, start_char = start_char,
            oov_char = oov_char, index_from = index_from)
        self.data_x = train_data_x if train else test_data_x
        self.data_y = train_data_y if train else test_data_y
        self.max_len = 1190
        self.num_class = num_class
        # sorted_corpus = sorted(zip(self.data_x, self.data_y),
        #                        key = lambda z: (len(z[0]), z[1]))
        # self.data_x = [x for (x, y) in sorted_corpus]
        # self.data_y = [y for (x, y) in sorted_corpus]

        self.data_size = len(self.data_x)  # set by the sum-class
        self.word2id = self.get_word_index(os.getcwd() + '/' + self.args.tmp_dir + self.__class__.__name__ + '/imdb_word_index.json')

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.data_size

    @staticmethod
    def batchfy_fn(data):
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        max_len = max(map(len, x))
        # if max_len % 5 != 0:
        #     max_len += 5 - (max_len % 5)
        return sequence.pad_sequences(x, maxlen = max_len, padding = 'post'), y

    def load_data(self, path = 'imdb.npz',
                  num_words = None,
                  skip_top = 0,
                  maxlen = None,
                  seed = 113,
                  start_char = 1,
                  oov_char = 2,
                  index_from = 3):
        path = get_file(
            path,
            cache_subdir = path,
            origin = 'https://s3.amazonaws.com/text-datasets/imdb.npz',
            file_hash = '599dadb1135973df5b59232a0e9a887c')
        f = np.load(path)
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']
        f.close()

        np.random.seed(seed)
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        xs = np.concatenate([x_train, x_test])
        labels = np.concatenate([labels_train, labels_test])

        if start_char is not None:
            xs = [[start_char] + [w + index_from for w in x] for x in xs]
        elif index_from:
            xs = [[w + index_from for w in x] for x in xs]

        if maxlen:
            new_xs = []
            new_labels = []
            for x, y in zip(xs, labels):
                if len(x) < maxlen:
                    new_xs.append(x)
                    new_labels.append(y)
            xs = new_xs
            labels = new_labels
            if not xs:
                raise ValueError('After filtering for sequences shorter than maxlen=' +
                                 str(maxlen) + ', no sequence was kept. '
                                               'Increase maxlen.')
        if not num_words:
            num_words = max([max(x) for x in xs])

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x]
                  for x in xs]
        else:
            new_xs = []
            for x in xs:
                nx = []
                for w in x:
                    if skip_top <= w < num_words:
                        nx.append(w)
                new_xs.append(nx)
            xs = new_xs

        x_train = np.array(xs[:len(x_train)])
        y_train = np.array(labels[:len(x_train)])

        x_test = np.array(xs[len(x_train):])
        y_test = np.array(labels[len(x_train):])
        return (x_train, y_train), (x_test, y_test)

    def get_word_index(self, path = 'imdb_word_index.json'):
        logging.info('Load the dictionaries...')
        path = get_file(path, origin = 'https://s3.amazonaws.com/text-datasets/imdb_word_index.json')
        f = open(path)
        data = json.load(f)
        f.close()
        logging.info('Load the dictionaries, size : %d' % len(data))
        return data
