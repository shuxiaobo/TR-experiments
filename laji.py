#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/24
import json
import logging
import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences, LineSentence


def gen_text():
    with open('data/processed/AIC/train.json.txt', mode = 'r', encoding = 'utf-8') as f:
        with open('data/processed/AIC/test.json.txt', mode = 'r', encoding = 'utf-8') as f1:
            with open('data/processed/AIC/sentext.txt', mode = 'w+', encoding = 'utf-8') as f2:
                for line in f:
                    line = json.loads(line.strip())['passage']
                    f2.write(line + '\n')
                for line in f1:
                    line = json.loads(line.strip())['passage']
                    f2.write(line + '\n')

    print('gen over')


def word2vec():
    input_file = 'data/processed/AIC/sentext.txt'
    outp1 = 'data/processed/AIC/baike.model'
    outp2 = 'data/processed/AIC/word2vec_format'
    # 训练模型 输入语料目录 embedding size 256,共现窗口大小10,去除出现次数5以下的词,多线程运行,迭代10次
    print('start train')
    model = Word2Vec(LineSentence(input_file),
                     size = 300, window = 6, min_count = 10,
                     workers = multiprocessing.cpu_count(), iter = 50)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary = False)

    print('trained over')


if __name__ == '__main__':
    gen_text()
    word2vec()
