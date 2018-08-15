#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
import json
import logging
from utils.util import *
from collections import Counter
from datasets.binary import *
logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d] %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')


def preprocess(args):
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
    logger('Processing now..')
    CR(args).data_split(args)
    MR(args).data_split(args)
    MPQA(args).data_split(args)
    SUBJ(args).data_split(args)
    logging.info('Processed over...')


if __name__ == '__main__':
    preprocess('../data/train.filter.txt')
    preprocess('../data/test.filter.txt', mode = 'a+')
