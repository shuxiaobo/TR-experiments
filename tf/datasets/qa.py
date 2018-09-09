#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/8
from __future__ import absolute_import, division, unicode_literals

import io
import os
import json
import jieba
import logging
import numpy as np
from utils.util import logger
from collections import Counter
from tf.datasets.classify_eval import ClassifierEval
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from utils.util import logger, prepare_split, write_file


class QADataSetBase():
    def __init__(self, args, num_class = 3, seed = 1111, file_name = ''):
        logging.info("Init the data set...")

        self._UNKOWN = '_<UNKNOW>'
        self._PAD = '_<PAD>'
        self._PAD_ID = 0
        self._UNKOWN_ID = 1

        self.alt_max_len = 0

        self.seed = seed
        self.args = args
        self.max_len = 0  # will be updated when load the train and test file
        self.num_class = num_class
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data(file_name)

        # for doc
        train_max, train_mean, train_min = self.statistic_len(self.train_x[0])
        test_max, test_mean, test_min = self.statistic_len(self.test_x[0])
        self.doc_max_len = max(train_max, test_max)
        logger("Train passage data max len:%d, mean len:%d, min len:%d " % (train_max, train_mean, train_min))
        logger("Test passage data max len:%d, mean len:%d, min len:%d " % (test_max, test_mean, test_min))

        # for query
        train_max, train_mean, train_min = self.statistic_len(self.train_x[1])
        test_max, test_mean, test_min = self.statistic_len(self.test_x[1])
        self.query_max_len = max(train_max, test_max)
        logger("Train query data max len:%d, mean len:%d, min len:%d " % (train_max, train_mean, train_min))
        logger("Test query data max len:%d, mean len:%d, min len:%d " % (test_max, test_mean, test_min))

        self.valid_nums = 0
        self.train_nums = len(self.train_x[0])
        self.test_nums = len(self.test_x[0])

        # load the data word dict
        self.word2id, self.word_file = self.get_word_index(
            os.path.join(args.tmp_dir, self.__class__.__name__, file_name + args.word_file), exclude_n = self.args.skip_top, max_size = self.args.num_words)
        self.word2id_size = len(self.word2id)
        self.train_idx = np.random.permutation(self.train_nums // self.args.batch_size)

    def load_data(self, file_name = ''):
        # load the train
        train_x, train_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.train_file))
        train_x, train_y = QADataSetBase.sort_corpus(train_x, train_y)

        # load the test
        test_x, test_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.test_file))
        test_x, test_y = QADataSetBase.sort_corpus(test_x, test_y)

        return train_x, train_y, test_x, test_y

    @staticmethod
    def sort_corpus(data_x, data_y):
        sorted_corpus = sorted(zip(*data_x, data_y), key = lambda z: (len(z[0]), len(z[1])))
        doc = list()
        qry = list()
        alt = list()
        ids = list()
        yy = list()
        for kk in sorted_corpus:
            doc.append(kk[0])
            qry.append(kk[1])
            alt.append(kk[2])
            ids.append(kk[3])
            yy.append(kk[-1])
        return [doc, qry, alt, ids], yy

    @property
    def train_idx(self):
        return self._train_idx

    @train_idx.setter
    def train_idx(self, value):
        self._train_idx = value

    def statistic_len(self, data):

        lens = [len(d) for d in data]
        if len(lens) == 0:
            return 0, 0, 0
        return max(lens), sum(lens) / len(lens), min(lens)

    def shuffle(self):
        logger("Shuffle the dataset.")
        np.random.shuffle(self.train_idx)

    def load_file(self, fpath):
        """
        [data_query, data_doc, data_alt, data_query_id]

        0 : data_doc
        1 : data_query
        2 : data_alt
        3 : data_query_id

        :param fpath:
        :return:
        """
        data_query = list()
        data_query_id = list()
        data_doc = list()
        data_ans = list()
        data_alt = list()

        alt_les = list()
        if not os.path.exists(fpath + '.txt'):
            logger("Preprocess the json data . cut the chinese...")
            f1 = io.open(fpath + '.txt', mode = 'w+', encoding = 'utf-8')
            with io.open(fpath, 'r', encoding = 'utf-8') as f:
                for line in f.read().splitlines():
                    try:
                        line = json.loads(line)
                    except Exception as err:
                        logging.error("Deal the input line error" + '：' + line)
                        continue
                    passage = list(jieba.cut(line["passage"]))
                    line["passage"] = ' '.join(passage)
                    data_doc.append(passage)

                    query = list(jieba.cut(line["query"]))
                    line["query"] = ' '.join(query)
                    data_query.append(query)
                    data_query_id.append(line["query_id"])

                    alt = []

                    alt_tmp = line["alternatives"].split('|')
                    for l in alt_tmp:
                        alt.append(list(jieba.cut(l)))

                    data_alt.append(alt)
                    data_ans.append(alt_tmp.index(line["answer"]))
                    f1.write(json.dumps(line, ensure_ascii = False) + '\n')
            f1.close()
        with io.open(fpath + '.txt', mode = 'r', encoding = 'utf-8') as f:
            logger("Load the data from file: %s" % fpath + ".txt")
            for line in f.read().splitlines():
                try:
                    line = json.loads(line)
                except Exception as err:
                    logging.error("Deal the input line error" + '：' + line)
                    continue
                passage = list(line["passage"].split(' '))
                if len(passage) > 200:
                    continue

                alt = []
                alt_tmp = line["alternatives"].split('|')
                for l in alt_tmp:
                    altsss = list(jieba.cut(l))
                    alt.append(altsss)
                    alt_les.extend([len(s) for s in altsss])
                query = list(line["query"].split(' '))

                if len(query) == 0 or len(alt_tmp) == 0:
                    continue

                data_query_id.append(line["query_id"])
                data_doc.append(passage)
                data_query.append(query)

                data_alt.append(alt)
                data_ans.append(alt_tmp.index(line["answer"]))

        data_x = [data_doc, data_query, data_alt, data_query_id]
        data_y = data_ans
        self.alt_max_len = max(self.alt_max_len, max(alt_les))
        logger("Load the data over, size: %d..., alt lenght : %d" % (len(data_ans), self.alt_max_len))
        return data_x, data_y

    def prepare_dict(self, file_name, exclude_n = 10, max_size = 10000):
        logger("Prepare the dictionary for the {}...".format(self.__class__.__name__))
        data = self.train_x[0] + self.train_x[1]  # use passage words and query words
        word2id = QADataSetBase.gen_dictionary(data = data, dict_path = file_name, exclude_n = exclude_n, max_size = max_size)
        logger("Word2id size : %d" % len(word2id))
        return word2id

    @staticmethod
    def gen_dictionary(data, dict_path, exclude_n = 10, max_size = 10000, word2id = None):
        word2id = dict() if not word2id else word2id

        for i, d in enumerate(data):
            # d = jieba.cut(d)
            for j, s in enumerate(d):
                if s not in word2id.keys():
                    word2id.setdefault(s, 0)
                word2id[s] = word2id[s] + 1
        c = Counter(word2id)
        rs = c.most_common()
        word2id = dict(word2id, **{k[0]: v for v, k in enumerate(rs[exclude_n:max_size + exclude_n])})
        with open(dict_path, mode = 'w+', encoding = 'utf-8') as f:
            for d in rs:
                f.write(d[0] + '\n')
        return word2id

    def get_word_index(self, path = None, exclude_n = 10, max_size = 10000):
        if not path:
            path = self.args.tmp_dir + self.__class__.__name__ + self.args.word_file
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            word2id = {self._PAD: self._PAD_ID, self._UNKOWN: self._UNKOWN_ID}
            with open(path, mode = 'r', encoding = 'utf-8') as f:
                for l in f.readlines()[exclude_n: max_size + exclude_n]:
                    word2id.setdefault(l.strip(), len(word2id))
        else:
            word2id = self.prepare_dict(path, exclude_n = exclude_n, max_size = max_size)
            logger('Word2id size : %d' % len(word2id))
        return word2id, path

    def getitem(self, dataset, index):
        result = [self.word2id[d] if self.word2id.get(d) else self.word2id[self._UNKOWN] for d in dataset[index]]
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
            stop = (idx + 1) * batch_size if start < sample_num and (idx + 1) * batch_size < sample_num else len(dataset_x[0])
        document = [self.getitem(dataset_x[0], i) for i in range(start, stop)]
        query = [self.getitem(dataset_x[1], i) for i in range(start, stop)]
        alter = [
            sequence.pad_sequences([[self.word2id[d] if self.word2id.get(d) else self.word2id[self._UNKOWN] for d in dd] for dd in dataset_x[2][i]],
                                   maxlen = self.alt_max_len)
            for i in range(start, stop)]
        data = {
            "document:0": sequence.pad_sequences(document, maxlen = self.doc_max_len, padding = "post"),
            "answer:0": dataset_y[start:stop],
            "query:0": sequence.pad_sequences(query, maxlen = self.query_max_len, padding = "post"),
            "alternative:0": alter
        }
        samples = stop - start
        if len(document) != len(dataset_y[start:stop]) or len(dataset_y[start:stop]) != samples:
            print(len(document), len(dataset_y[start:stop]), samples)
        return data, samples


class AIC(QADataSetBase):
    def __init__(self, args, num_class = 3, seed = 1111, file_name = ''):
        super(AIC, self).__init__(args, num_class, seed, file_name)