#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import codecs
import logging
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.python.platform import gfile
from tensorflow.contrib.tensorboard.plugins import projector

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


def prepare_dictionary(data, dict_path, exclude_n = 10, max_size = 10000, word2id = None):
    word2id = dict() if word2id == None else word2id
    for i, d in enumerate(data):
        for j, s in enumerate(d):
            if s not in word2id.keys():
                word2id.setdefault(s, 0)
            word2id[s] = word2id[s] + 1
    c = Counter(word2id)
    rs = [('_<PAD>', 0), ('_<UNKNOW>', 1)] + c.most_common(max_size + exclude_n)[exclude_n:]
    word2id = {k[0]: v for v, k in enumerate(rs)}
    with open(dict_path, mode = 'w+', encoding = 'utf-8') as f:
        for d in rs:
            f.write(d[0] + '\n')
    return word2id


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


def prepare_split(X, y, validation_data = None, validation_split = None):
    # Preparing validation data
    assert validation_split or validation_data
    if validation_data is not None:
        trainX, trainy = X, y
        devX, devy = validation_data
    else:
        permutation = np.random.permutation(len(X))
        trainidx = permutation[int(validation_split * len(X)):]
        devidx = permutation[0:int(validation_split * len(X))]
        trainX, trainy = np.array(X)[trainidx], np.array(y)[trainidx]
        devX, devy = np.array(X)[devidx], np.array(y)[devidx]

    return trainX.tolist(), trainy.tolist(), devX.tolist(), devy.tolist()


def write_file(path, data):
    logger('Write data to file path {}'.format(path))
    with open(path, mode = 'w+', encoding = 'utf-8') as f:
        for d in data:
            f.write(' '.join(d) + '\n')


def visualize_embedding(embedding_matrix_name, word2id, output_path = None, writer = None):
    """
    visualize the word embedding.
    :param embedding_matrix_name: numpy word embedding name
    :param word2id: word2id dict
    :param output_path: path to save the tensorboard file
    :param writer: tf.writer
    :return:
    """
    assert output_path or writer, "writer or output path could not be none in the same time"
    output_path = output_path if not writer else writer.get_logdir()
    meta_file = "embedding_matrix.tsv"
    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for word, i in word2id.items():
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    writer = tf.summary.FileWriter(output_path, sess.graph) if not writer else writer

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding_matrix_name
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    if not writer:
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


def file2dict(file_path):
    """
    file to str2id dict
    :param file_path:
    :return:
    """
    with open(file_path, mode = 'r') as f:
        result = {}
        for l in f:
            result.setdefault(l.strip(), len(result))
    return result
