#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
import json
import logging
from utils.util import *
from collections import Counter

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d] %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def preprocess(filein, dir2save = '../data/processed/', voca_size = 15000, ex_top_n = 20, mode = 'w+'):
    """
    r 只能读
    r+ 可读可写，不会创建不存在的文件，从顶部开始写，会覆盖之前此位置的内容
    w+ 可读可写，如果文件存在，则覆盖整个文件，不存在则创建
    w 只能写，覆盖整个文件，不存在则创建
    a 只能写，从文件底部添加内容 不存在则创建
    a+ 可读可写 从文件顶部读取内容 从文件底部添加内容 不存在则创建
    :param filein:
    :param dir2save:
    :return:
    """
    entity2id_file = dir2save + 'entity2id.txt'
    word2id_file = dir2save + 'word2id.txt'
    relation2id_file = dir2save + 'relation2id.txt'

    # data_file = dir2save + 'data.txt'  # id    e1    e2  sentence   relation    summary1   summary2   m1&m2

    def sentence2id(word2id, sentence):
        sentence = process_tokens(default_tokenizer(sentence))
        for i, t in enumerate(sentence):
            if not word2id.get(t):
                word2id.setdefault(t, 0)
            word2id[t] = word2id.get(t) + 1
        return sentence

    entity2id = dict()
    word2id = dict()
    relation2id = dict()

    if mode == 'a+':
        entity2id = read_dictionary(entity2id_file)
        word2id = read_dictionary(word2id_file)
        relation2id = read_dictionary(relation2id_file)

    logging.info('Preparing the data now...')
    with open(filein, mode = 'r', encoding = 'utf-8') as f, open(word2id_file, mode = 'w+', encoding = 'utf-8') as f1, open(entity2id_file, mode = 'w+',
                                                                                                                            encoding = 'utf-8') as f2, open(
        relation2id_file, mode = 'w+', encoding = 'utf-8') as f3, open(filein + '.linking', encoding = 'utf-8') as f4:
        id = 0

        for line in f:
            line = line.strip().split('\t')

            m1 = line[2]
            m2 = line[3]

            relation = f.readline().strip().split('\t')

            if not entity2id.get(m1):
                entity2id.setdefault(m1, len(entity2id))
            if not entity2id.get(m2):
                entity2id.setdefault(m2, len(entity2id))
            for r in relation:
                if not relation2id.get(r):
                    relation2id.setdefault(r, len(relation2id))
            num_sen = int(f.readline().strip())
            sentence = ''
            for i in range(num_sen):
                sentence += f.readline().strip()[:-10]

            sentence2id(word2id, sentence)

            f4.readline()
            for i, l in enumerate(json.loads(f4.readline().strip())):
                sentence2id(word2id, l['summary_1'])
                sentence2id(word2id, l['summary_2'])
            id += 1
            if id % 1000 == 0:
                logging.info(str(id) + ' samples have been processed...')

            f.readline()
            f4.readline()

        word2id = [('_<PAD>', 0), ('_<UNKNOW>', 0)] + Counter(word2id).most_common(voca_size + ex_top_n)[ex_top_n:]
        for k, v in word2id:
            f1.write(str(k) + '\t' + str(v) + '\n')
        for k, v in entity2id.items():
            f2.write(str(k) + '\t' + str(v) + '\n')
        for k, v in relation2id.items():
            f3.write(str(k) + '\t' + str(v) + '\n')

    logging.info('Processed over...')


if __name__ == '__main__':
    preprocess('../data/train.filter.txt')
    preprocess('../data/test.filter.txt', mode = 'a+')
