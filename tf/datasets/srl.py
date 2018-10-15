#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/10/10
from __future__ import absolute_import, division, unicode_literals

import io
import os
import re
import json
import jieba
import codecs
import logging
import numpy as np
from utils.util import logger
from collections import Counter
from tensorflow.python.platform import gfile
from tf.datasets.classify_eval import ClassifierEval
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from utils.util import logger, prepare_split, write_file


class SRL():
    def __init__(self, args, num_class = 67, seed = 1111, file_name = ''):
        logging.info("Init the data set...")

        self._UNKOWN = '_<UNKNOW>'
        self._PAD = '_<PAD>'
        self._PAD_ID = 0
        self._UNKOWN_ID = 1

        self.seed = seed
        self.args = args
        self.max_len = 200  # will be updated when load the train and test file
        self.num_class = num_class

        # load the data word dict
        self.dicts = self.get_word_index(os.path.join(args.tmp_dir, self.__class__.__name__), exclude_n = self.args.skip_top, max_size = self.args.num_words)
        self.word2id, self.id2word, postag2idx, idx2postag, label2idx, idx2label = self.dicts
        self.word2id_size = len(self.word2id)
        # self.char2id_size = len(self.char2id) if self.char2id else 0

        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = self.load_data(file_name)
        self.test_x, self.test_y = self.valid_x, self.valid_y

        self.valid_nums = len(self.valid_x)
        self.train_nums = len(self.train_x)
        self.test_nums = len(self.test_x)

        self.train_idx = np.random.permutation(self.train_nums // self.args.batch_size)

    def get_embedding_matrix(self, is_char_embedding = False):
        return None

    @property
    def train_idx(self):
        return self._train_idx

    @train_idx.setter
    def train_idx(self, value):
        self._train_idx = value

    def statistic_len(self, data):
        return len(data)

    def shuffle(self):
        logger("Shuffle the dataset.")
        np.random.shuffle(self.train_idx)

    def get_word_index(self, data_dir, exclude_n, max_size = None):
        # load wordidx
        fp = open(os.path.join(data_dir, "wordidx"), mode = "r", encoding = 'utf-8')
        word2idx, idx2word = dict(), dict()
        words = [line.strip() for line in fp.readlines()]
        if max_size is not None:
            words = words[:max_size]
        for idx, word in enumerate(words):
            word2idx[word] = idx
            idx2word[idx] = word
        fp.close()

        # load labelidx
        fp = open(os.path.join(data_dir, "postagidx"), mode = "r", encoding = 'utf-8')
        postag2idx, idx2postag = dict(), dict()
        for idx, postag in enumerate([line.strip() for line in fp.readlines()]):
            postag2idx[postag] = idx
            idx2postag[idx] = postag
        fp.close()

        # load labelidx
        fp = open(os.path.join(data_dir, "labelidx"), mode = "r", encoding = 'utf-8')
        label2idx, idx2label = dict(), dict()
        for idx, label in enumerate([line.strip() for line in fp.readlines()]):
            label2idx[label] = idx
            idx2label[idx] = label
        fp.close()

        return (word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label)

    def read_data(self, data_path):
        # read in preprocessed training data
        fp = open(data_path, "r", encoding = 'utf-8')
        records = [line.strip() for line in fp.readlines()]
        data = []
        sent_data = []
        for record in records:
            if record == '':
                data.append(sent_data)
                sent_data = []
                continue
            feats = record.split("\t")
            feats[7] = int(feats[7])  # preprocess distance feature
            assert len(feats) == 9  # a valid training record should have 9 attributes
            if len(sent_data) < self.max_len:
                sent_data.append(feats)
        return data

    def _process_data(self, data):
        word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label = self.dicts
        # transform data into padded numpy array
        data_X = np.zeros(shape = (len(data), self.max_len, 8), dtype = np.int32)
        data_y = np.zeros(shape = (len(data), self.max_len), dtype = np.int32)
        for i in range(len(data)):
            for j in range(len(data[i])):
                curword, lastword, nextword, pos, lastpos, nextpos, relword, dist, label = data[i][j]
                data_X[i, j, 0] = word2idx[curword] if curword in word2idx else 2
                data_X[i, j, 1] = word2idx[lastword] if lastword in word2idx else 2
                data_X[i, j, 2] = word2idx[nextword] if nextword in word2idx else 2
                data_X[i, j, 3] = word2idx[relword] if relword in word2idx else 2
                data_X[i, j, 4] = postag2idx[pos]
                data_X[i, j, 5] = postag2idx[lastpos]
                data_X[i, j, 6] = postag2idx[nextpos]
                data_X[i, j, 7] = dist
                data_y[i, j] = label2idx[label]
        return data_X, data_y

    def load_data(self, data_path):

        # read in preprocessed training data
        train_data = self.read_data(os.path.join(self.args.tmp_dir, self.__class__.__name__, data_path + self.args.train_file))
        valid_data = self.read_data(os.path.join(self.args.tmp_dir, self.__class__.__name__, data_path + self.args.valid_file))

        # transform data into padded numpy array
        training_X, training_y = self._process_data(train_data)
        valid_X, valid_y = self._process_data(valid_data)
        return training_X, training_y, valid_X, valid_y, [], []

    def getitem(self, dataset, index):
        result = [self.word2id[d] if self.word2id.get(d) else self.word2id[self._UNKOWN] for d in dataset[index]]
        return result

    def __len__(self):
        return self.n_samples

    def get_next_batch(self, mode, idx):
        """
        return next batch of data samples
        """
        batch_size = self.args.batch_size
        if mode == "train":
            dataset_x = self.train_x
            dataset_y = self.train_y
            sample_num = self.train_nums
        elif mode == "valid":
            dataset_x = self.valid_x
            dataset_y = self.valid_y
            sample_num = self.valid_nums
        else:
            dataset_x = self.test_x
            dataset_y = self.test_y
            sample_num = self.test_nums
        if mode == "train":
            start = self.train_idx[idx] * batch_size
            stop = (self.train_idx[idx] + 1) * batch_size
        else:
            start = idx * batch_size
            stop = (idx + 1) * batch_size if start < sample_num and (idx + 1) * batch_size < sample_num else len(dataset_x)
        if start > stop:
            print(start)
            print(stop)
        features = dataset_x[start:stop]
        length = np.array([len(sent) for sent in features], dtype = np.int32)
        label = dataset_y[start:stop]
        onehot_label = np.zeros(shape = (stop - start, self.max_len), dtype = np.int32)
        masks = np.zeros(shape = (stop - start, self.max_len), dtype = np.int32)
        for i in range(stop - start):
            masks[i, 0:length[i]] = 1
            # rel_masks[i, 0:lengths[i]] = features[i, 0:lengths[i], 7] == 0
            for j in range(self.max_len):
                onehot_label[i, j] = label[i, j]
        data = {
            "curword:0": features[:, :, 0],
            "lastword:0": features[:, :, 1],
            "nextword:0": features[:, :, 2],
            "predicate:0": features[:, :, 3],
            "curpostag:0": features[:, :, 4],
            "lastpostag:0": features[:, :, 5],
            "nextpostag:0": features[:, :, 6],
            "seq_length:0": length,
            "label:0": onehot_label,
            "mask:0": masks,
            "distance:0": features[:, :, 7]
            # self.rel_mask: rel_masks,
        }
        samples = stop - start

        return data, samples
