#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tensorflow.python.platform import gfile
import codecs
import numpy as np
import torch

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(filename)s[line:%(lineno)d]： %(message)s', datefmt = '%Y-%m-%d %I:%M:%S')

logger = logging.info


def gen_embeddings(word_dict, embed_dim, in_file = None, init = np.random.uniform):
    """
    Init embedding matrix with (or without) pre-trained word embeddings.
    """
    num_words = max(word_dict.values()) + 1
    embedding_matrix = init(-0.05, 0.05, (num_words, embed_dim)).astype(np.float32)
    logger('Embeddings: %d x %d' % (num_words, embed_dim))

    if not in_file:
        return embedding_matrix

    def get_dim(file):
        first = gfile.FastGFile(file, mode = 'r').readline()
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


def accuracy(out, label):
    return np.sum(np.equal(np.argmax(out, axis = -1), label))


def gather_rnnstate(data, mask):
    """
    return the last state valid
    :param data:
    :param mask:
    :return:
    """
    real_len_index = torch.sum(mask, -1) - 1
    # assert torch.max(real_len_index)[0].item() + 1 == data.shape[1]
    return torch.gather(data, 1, real_len_index.long().view(-1, 1).unsqueeze(2).repeat(1, 1, data.shape[-1])).squeeze()
