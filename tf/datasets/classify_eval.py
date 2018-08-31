#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import logging
import numpy as np
from utils.util import prepare_dictionary, logger, prepare_split, write_file
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from utils.util import logger


class ClassifierEval():
    def __init__(self, args, num_class = 2, seed = 1111, file_name = ''):
        logging.info("Init the dataset...")
        self.seed = seed
        self.args = args
        self.num_class = num_class
        self.max_len = 0  # will be update when load the train and test file

        # load the train
        self.train_x, self.train_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.train_file))
        sorted_corpus = sorted(zip(self.train_x, self.train_y),
                               key = lambda z: (len(z[0]), z[1]))
        self.train_x = [x for (x, y) in sorted_corpus]
        self.train_y = [y for (x, y) in sorted_corpus]

        # load the test
        self.test_x, self.test_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.test_file))
        sorted_corpus = sorted(zip(self.train_x, self.train_y),
                               key = lambda z: (len(z[0]), z[1]))
        self.test_x = [x for (x, y) in sorted_corpus]
        self.test_y = [y for (x, y) in sorted_corpus]

        self.valid_nums = 0
        self.train_nums = len(self.train_x)
        self.test_nums = len(self.test_x)

        # load the data word dict
        self.word_file = os.path.join(args.tmp_dir, self.__class__.__name__, file_name + args.word_file)
        if os.path.isfile(self.word_file) and os.path.getsize(self.word_file) > 0:
            self.word2id = self.get_word_index(self.word_file)
        else:
            self.word2id = self.prepare_dict(self.word_file)
        self.word2id_size = len(self.word2id)
        self.train_idx = np.random.permutation(self.train_nums // self.args.batch_size)

    @property
    def train_idx(self):
        return self._train_idx

    @train_idx.setter
    def train_idx(self, value):
        self._train_idx = value

    def shuffle(self):
        logger("Shuffle the dataset.")
        np.random.shuffle(self.train_idx)

    def load_file(self, fpath):
        with io.open(fpath, 'r', encoding = 'utf-8') as f:
            data_x = list()
            data_y = list()
            for line in f.read().splitlines():
                line = line.strip().split(' ')
                if len(line) <= 3:
                    continue
                data_x.append(line[:-1])
                data_y.append(int(line[-1]))
                self.max_len = len(line[:-1]) if len(line[:-1]) > self.max_len else self.max_len
        return data_x, data_y

    def prepare_dict(self, file_name):
        logger("Prepare the dictionary for the {}...".format(self.__class__.__name__))
        word2id = prepare_dictionary(data = self.data_x, dict_path = file_name, exclude_n = self.args.skip_top, max_size = self.args.num_words)
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

    def getitem(self, dataset, index):
        result = [self.word2id[d] if self.word2id.get(d) else self.word2id['_<UNKNOW>'] for d in dataset[index]]
        return result

    def __len__(self):
        return self.n_samples

    def data_split(self, args):
        trainx, trainy, testx, testy = prepare_split(self.train_data_x, self.train_data_y, validation_split = 0.2)
        train = [x + [str(y)] for x, y in zip(trainx, trainy)]
        test = [x + [str(y)] for x, y in zip(testx, testy)]
        write_file(data = train, path = os.path.join(args.tmp_dir, self.__class__.__name__, args.train_file))
        write_file(data = test, path = os.path.join(args.tmp_dir, self.__class__.__name__, args.test_file))

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
        document = [self.getitem(dataset_x, i) for i in range(start, stop)]
        data = {
            "document:0": sequence.pad_sequences(document, maxlen = self.max_len, padding = "post"),
            "y_true:0": dataset_y[start:stop]
        }
        samples = stop - start
        if len(document) != len(dataset_y[start:stop]) or len(dataset_y[start:stop]) != samples:
            print(len(document), len(dataset_y[start:stop]), samples)
        return data, samples


class CR(ClassifierEval):
    def __init__(self, args, seed = 1111):
        logging.debug('***** Transfer task : CR *****\n\n')
        super(self.__class__, self).__init__(args, seed = seed)


class MR(ClassifierEval):
    def __init__(self, args, seed = 1111):
        logging.debug('***** Transfer task : MR *****\n\n')
        super(self.__class__, self).__init__(args, seed = seed)


class SUBJ(ClassifierEval):
    def __init__(self, args, seed = 1111):
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        super(self.__class__, self).__init__(args, seed)


class MPQA(ClassifierEval):
    def __init__(self, args, seed = 1111):
        logging.debug('***** Transfer task : MPQA *****\n\n')
        super(self.__class__, self).__init__(args, seed = seed)


class Kaggle(ClassifierEval):
    def __init__(self, args, seed = 1111):
        logging.debug('***** Transfer task : Kaggle *****\n\n')
        super(Kaggle, self).__init__(args, num_class = 5, seed = seed)


class TREC(ClassifierEval):
    def __init__(self, args, seed = 1111):
        logging.info('***** Transfer task : TREC *****\n\n')
        self.seed = seed
        super(TREC, self).__init__(args = args, num_class = 6, seed = seed)


class SST(ClassifierEval):
    def __init__(self, args, nclasses = 5, seed = 1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)

        tmp = 'binary/' if nclasses == 2 else 'fine/'
        super(SST, self).__init__(args, num_class = nclasses, file_name = tmp, seed = seed)
