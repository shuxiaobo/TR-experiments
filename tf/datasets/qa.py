#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/8
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

jieba.add_word('不可以')
jieba.add_word('无法确定')
jieba.add_word('不需要')
jieba.add_word('不需要')
jieba.add_word('不一样')
jieba.add_word('靠谱')
jieba.add_word('不一定')
jieba.add_word('不靠谱')
jieba.add_word('不严重')


def default_tokenizer(sentence):
    _DIGIT_RE = re.compile(r"\d+")
    sentence = _DIGIT_RE.sub("0", sentence)  # digital replace. because the answer contain the number
    _CHAR_RE = re.compile(r"[A-Za-z]+$")
    sentence = _CHAR_RE.sub("a", sentence)  # No char replace. because the answer donnot contain the char
    _NOCHINESE_RE = re.compile(r"[^\w\u4e00-\u9fff]+")
    sentence = _NOCHINESE_RE.sub("", sentence)
    # sentence = " ".join(sentence.split("|"))
    return list(jieba.cut(sentence))


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


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
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = self.load_data(file_name)
        # self.test_x, self.test_y = self.valid_x, self.valid_y
        # for doc
        train_max, train_mean, train_min = self.statistic_len(self.train_x[0])
        valid_max, valid_mean, valid_min = self.statistic_len(self.valid_x[0])
        test_max, test_mean, test_min = self.statistic_len(self.test_x[0])
        self.doc_max_len = max(train_max, valid_max, test_max)
        logger("Train passage data max len:%d, mean len:%d, min len:%d " % (train_max, train_mean, train_min))
        logger("Valid passage data max len:%d, mean len:%d, min len:%d " % (valid_max, valid_mean, valid_min))
        logger("Test passage data max len:%d, mean len:%d, min len:%d " % (test_max, test_mean, test_min))
        # for query
        train_max, train_mean, train_min = self.statistic_len(self.train_x[1])
        valid_max, valid_mean, valid_min = self.statistic_len(self.valid_x[1])
        test_max, test_mean, test_min = self.statistic_len(self.test_x[1])
        self.query_max_len = max(train_max, valid_max, test_max)
        logger("Train query data max len:%d, mean len:%d, min len:%d " % (train_max, train_mean, train_min))
        logger("Valid query data max len:%d, mean len:%d, min len:%d " % (valid_max, valid_mean, valid_min))
        logger("Test query data max len:%d, mean len:%d, min len:%d " % (test_max, test_mean, test_min))
        self.d_char_len = self.args.max_char_len
        self.q_char_len = self.args.max_char_len
        # if self.args.use_char_embedding:
        #     self.d_char_len, _, _ = self.statistic_len(self.train_x[0] + self.valid_x[0] + self.test_x[0], char = True)
        #     self.q_char_len, _, _ = self.statistic_len(self.train_x[1] + self.valid_x[1] + self.test_x[1], char = True)
        #     logger("Document char max len:%d, Query char max len:%d " % (self.d_char_len, self.q_char_len))
        self.valid_nums = len(self.valid_x[0])
        self.train_nums = len(self.train_x[0])
        self.test_nums = len(self.test_x[0])

        # load the data word dict
        self.word2id, self.char2id, self.word_file = self.get_word_index(
            os.path.join(args.tmp_dir, self.__class__.__name__), exclude_n = self.args.skip_top, max_size = self.args.num_words)
        self.word2id_size = len(self.word2id)
        self.char2id_size = len(self.char2id) if self.char2id else 0
        self.train_idx = np.random.permutation(self.train_nums // self.args.batch_size)

    def load_data(self, file_name = ''):
        # load the train
        train_x, train_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.train_file))
        train_x, train_y = QADataSetBase.sort_corpus(train_x, train_y)

        # load the valid
        valid_x, valid_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.valid_file))
        # valid_x, valid_y = QADataSetBase.sort_corpus(valid_x, valid_y)

        # load the test
        test_x, test_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + self.args.test_file))
        # test_x, test_y = QADataSetBase.sort_corpus(test_x, test_y)

        return train_x, train_y, valid_x, valid_y, test_x, test_y

    @staticmethod
    def sort_corpus(data_x, data_y):
        sorted_corpus = sorted(zip(*data_x, data_y), key = lambda z: (len(z[0]), len(z[1])))
        doc = list()
        qry = list()
        alt = list()
        ids = list()

        alt_no_cut = list()
        yy = list()
        for kk in sorted_corpus:
            doc.append(kk[0])
            qry.append(kk[1])
            alt.append(kk[2])
            ids.append(kk[3])
            alt_no_cut.append(kk[4])
            yy.append(kk[-1])
        return [doc, qry, alt, ids, alt_no_cut], yy

    @property
    def train_idx(self):
        return self._train_idx

    @train_idx.setter
    def train_idx(self, value):
        self._train_idx = value

    def statistic_len(self, data, char = False):
        lens = [len(d) for d in data] if not char else [len(c) for d in data for c in d]
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
        4 : alter_no_cut
        :param fpath:
        :return:
        """
        data_query = list()
        data_query_id = list()
        data_doc = list()
        data_ans = list()
        data_alt = list()
        data_alt_not_cut = list()

        data_anwser = list()
        data_anwser_split = list()

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
                    passage = default_tokenizer(line["passage"])
                    line["passage"] = ' '.join(passage)
                    data_doc.append(passage)

                    query = default_tokenizer(line["query"])
                    line["query"] = ' '.join(query)
                    data_query.append(query)
                    data_query_id.append(line["query_id"])

                    alt_tmp = line["alternatives"].split('|')

                    if "answer" in line:
                        ans_tmp = alt_tmp.index(line["answer"])

                        data_anwser.append(line["answer"])
                        data_anwser_split.append(list(jieba.cut(line["answer"])))
                    else:
                        ans_tmp = -1
                    data_ans.append(ans_tmp)
                    f1.write(json.dumps(line, ensure_ascii = False) + '\n')
            f1.close()
        with io.open(fpath + '.txt', mode = 'r', encoding = 'utf-8') as f:
            logger("Load the data from file: %s" % fpath + ".txt")
            for i, line in enumerate(f.read().splitlines()):
                try:
                    line = json.loads(line)
                except Exception as err:
                    logging.error("Deal the input line error" + '：' + line)
                    continue
                passage = list(line["passage"].split(' '))
                passage = passage[:100]
                alt = []
                alt_tmp = line["alternatives"].split('|')
                if len(alt_tmp) != 3:
                    alt_tmp += [alt_tmp[0]] * (3 - len(alt_tmp))
                assert len(alt_tmp) == 3
                data_alt_not_cut.append(alt_tmp)
                for l in alt_tmp:
                    altsss = default_tokenizer(l)
                    alt.append(altsss)
                    alt_les.extend([len(s) for s in altsss])
                query = line["query"].split(' ')

                if len(query) == 0 or len(alt_tmp) == 0:
                    continue

                data_query_id.append(line["query_id"])
                data_doc.append(passage)
                data_query.append(query)

                data_alt.append(alt)

                if "answer" in line:
                    ans_tmp = alt_tmp.index(line["answer"])

                    data_anwser.append(line["answer"])
                    data_anwser_split.append(list(jieba.cut(line["answer"])))
                else:
                    ans_tmp = -1
                data_ans.append(ans_tmp)

        data_x = [data_doc, data_query, data_alt, data_query_id, data_alt_not_cut]
        data_y = data_ans
        if fpath.endswith('train.json'):
            self.data_ans = data_anwser
            self.data_ans_split = data_anwser_split
        elif fpath.endswith('test.json'):
            self.data_ans_test = data_anwser
            self.data_ans_test_split = data_anwser_split
        logger("Answers class most common : %s" % str(Counter(data_ans).most_common()))
        self.alt_max_len = max(self.alt_max_len, max(alt_les))
        logger("Load the data over, size: %d..., alt length : %d" % (len(data_ans), self.alt_max_len))
        return data_x, data_y

    def prepare_dict(self, file_name, exclude_n = 10, max_size = 10000):
        logger("Prepare the dictionary for the {}...".format(self.__class__.__name__))
        data = self.train_x[0] + self.train_x[1] + self.valid_x[0] + self.valid_x[1]  # use passage words and query words
        word2id = {self._PAD: self._PAD_ID, self._UNKOWN: self._UNKOWN_ID}
        char2id = {self._PAD: self._PAD_ID, self._UNKOWN: self._UNKOWN_ID}
        word2id = QADataSetBase.gen_dictionary(data = data, word2id = word2id, dict_path = os.path.join(file_name, self.args.word_file), exclude_n = exclude_n,
                                               max_size = max_size)
        char2id = QADataSetBase.gen_char_dictionary(data = self.train_x[0] + self.train_x[1], char2id = char2id,
                                                    dict_path = os.path.join(file_name, self.args.char_file),
                                                    exclude_n = exclude_n, max_size = max_size)
        return word2id, char2id

    def get_embedding_matrix(self, is_char_embedding = False):
        """
        :param is_char_embedding: is the function called for generate char embedding
        :param vocab_file: file containing saved vocabulary.
        :return: a dict with each key as a word, each value as its corresponding embedding vector.
        """
        word_dict = self.word2id
        embedding_file = None if is_char_embedding else self.args.embedding_file
        embedding_dim = self.args.char_embedding_dim if is_char_embedding else self.args.embedding_dim
        embedding_matrix = self.gen_embeddings(word_dict,
                                               embedding_dim,
                                               embedding_file,
                                               init = np.random.uniform)
        return embedding_matrix

    @staticmethod
    def gen_embeddings(word_dict, embed_dim, in_file = None, init = np.zeros):
        """
        Init embedding matrix with (or without) pre-trained word embeddings.
        """
        num_words = max(word_dict.values()) + 1
        embedding_matrix = init(-0.05, 0.05, (num_words, embed_dim))
        logger('Embeddings: %d x %d' % (num_words, embed_dim))

        if not in_file:
            return embedding_matrix

        def get_dim(file):
            first = gfile.FastGFile(file, mode = 'r').readline()
            if len(first.split()) == 2:
                return int(first.split(' ')[-1])
            return len(first.split()) - 1

        assert get_dim(in_file) == embed_dim
        logger('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in codecs.open(in_file, encoding = "utf-8"):
            sp = line.split()
            if sp[0] in word_dict:
                pre_trained += 1
                embedding_matrix[word_dict[sp[0]]] = np.asarray([float(x) for x in sp[1:]], dtype = np.float32)
        logger("Pre-trained: {}, {:.3f}%".format(pre_trained, pre_trained * 100.0 / num_words))
        return embedding_matrix

    @staticmethod
    def gen_dictionary(data, dict_path, exclude_n = 10, max_size = 10000, word2id = None):
        word2id = dict() if not word2id else word2id
        word2id_new = dict()
        for i, d in enumerate(data):
            for j, s in enumerate(d):
                if s not in word2id_new.keys():
                    word2id_new.setdefault(s, 0)
                word2id_new[s] = word2id_new[s] + 1
        c = Counter(word2id_new)
        rs = c.most_common()
        word2id = dict(word2id, **{k[0]: v for v, k in enumerate(rs[exclude_n:max_size + exclude_n])})
        with open(dict_path, mode = 'w+', encoding = 'utf-8') as f:
            for d in rs:
                f.write(d[0] + '\n')
        return word2id

    @staticmethod
    def gen_char_dictionary(data, dict_path, exclude_n = 10, max_size = 10000, char2id = None):
        char2id = dict() if not char2id else char2id
        char2id_new = dict()
        for i, d in enumerate(data):
            for j, s in enumerate(d):
                for k, q in enumerate(s):
                    if not char2id_new.get(q):
                        char2id_new.setdefault(q, 0)
                    char2id_new[q] += 1
        c = Counter(char2id_new)
        rs = c.most_common()
        char2id = dict(char2id, **{k[0]: v for v, k in enumerate(rs[exclude_n:max_size + exclude_n])})
        with open(dict_path, mode = 'w+', encoding = 'utf-8') as f:
            for d in rs:
                f.write(d[0] + '\n')
        return char2id

    def get_word_index(self, path = None, exclude_n = 10, max_size = 10000):
        if not path:
            path = self.args.tmp_dir + self.__class__.__name__
        word2id, char2id = dict(), dict()
        word_file_exit = os.path.isfile(os.path.join(path, self.args.word_file)) and os.path.getsize(os.path.join(path, self.args.word_file)) > 0
        char_file_exit = os.path.isfile(os.path.join(path, self.args.char_file)) and os.path.getsize(
            os.path.join(path, self.args.char_file)) > 0
        if not word_file_exit or not char_file_exit:
            word2id, char2id = self.prepare_dict(path, exclude_n = exclude_n, max_size = max_size)
            logger('Word2id size : %d, Char2id size : %d' % (len(word2id), len(char2id)))
            return word2id, char2id, path
        if word_file_exit:
            word2id = {self._PAD: self._PAD_ID, self._UNKOWN: self._UNKOWN_ID}
            with open(os.path.join(path, self.args.word_file), mode = 'r', encoding = 'utf-8') as f:
                for i in range(exclude_n): f.readline()
                for i in range(exclude_n, max_size + exclude_n):
                    l = f.readline().strip()
                    word2id.setdefault(l, len(word2id))
        if self.args.use_char_embedding and char_file_exit:
            char2id = {self._PAD: self._PAD_ID, self._UNKOWN: self._UNKOWN_ID}
            with open(os.path.join(path, self.args.char_file), mode = 'r', encoding = 'utf-8') as f:
                for i in range(exclude_n): f.readline()
                for i in range(exclude_n, max_size + exclude_n):
                    l = f.readline()
                    char2id.setdefault(l.strip(), len(char2id))
        logger('Word2id size : %d, Char2id size : %d' % (len(word2id), len(char2id)))
        return word2id, char2id, path

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

    def getitem_char(self, dataset, index):
        result = [[self.char2id[d] if self.char2id.get(d) else self.char2id[self._UNKOWN] for c in d[:self.args.max_char_len]] for d in dataset[index]]
        return result

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
                                   maxlen = self.alt_max_len, padding = "post")
            for i in range(start, stop)]
        data = {
            "document:0": sequence.pad_sequences(document, maxlen = self.doc_max_len, padding = "post"),
            "answer:0": dataset_y[start:stop],
            "query:0": sequence.pad_sequences(query, maxlen = self.query_max_len, padding = "post"),
            "alternative:0": alter,
        }
        if self.args.use_char_embedding:
            d = {"document_char:0": sequence.pad_sequences(
                [sequence.pad_sequences(self.getitem_char(dataset_x[0], i), maxlen = self.d_char_len, padding = "post") for i in
                 range(start, stop)], maxlen = self.doc_max_len, padding = "post"),
                "query_char:0": sequence.pad_sequences(
                    [sequence.pad_sequences(self.getitem_char(dataset_x[1], i), maxlen = self.q_char_len, padding = "post") for i in
                     range(start, stop)], maxlen = self.query_max_len, padding = "post")
            }
            data = dict(data, **d)

        samples = stop - start
        # if len(document) != len(dataset_y[start:stop]) or len(dataset_y[start:stop]) != samples:
        #     print(len(document), len(dataset_y[start:stop]), samples)
        return data, samples


class AIC(QADataSetBase):
    def __init__(self, args, num_class = 3, seed = 1111, file_name = ''):
        super(AIC, self).__init__(args, num_class, seed, file_name)
