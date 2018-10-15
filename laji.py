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
    size = 300
    input_file = 'data/processed/AIC/sentext.txt'
    outp1 = 'data/processed/AIC/baike.model'
    outp2 = 'data/processed/AIC/word2vec.{}d.txt'.format(size)
    # 训练模型 输入语料目录 embedding size 256,共现窗口大小10,去除出现次数5以下的词,多线程运行,迭代10次
    print('start train')
    model = Word2Vec(LineSentence(input_file),
                     size = 300, window = 5, min_count = 10,
                     workers = multiprocessing.cpu_count(), iter = 50)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary = False)
    print(model.wv.similar_by_word('无法确定'))
    # print(model.wv.similar_by_word('不可以'))
    # print(model.wv.similar_by_word('不需要'))

    print('trained over')

    return model


import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 300))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


def visualize_embedding(embedding_matrix, word2id, output_path):
    """
    visualize the word embedding.
    :param embedding_matrix: numpy word embedding
    :param word2id: word2id dict
    :param output_path: path to save the tensorboard file
    :return:
    """
    meta_file = "w2x_metadata.tsv"
    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for word, i in enumerate(word2id):
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(embedding_matrix, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


if __name__ == '__main__':
    # gen_text()
    model = word2vec()
    # model = Word2Vec.load("/Users/caiyunxin/Desktop/word2vec_model_100")
    visualize(model, "w2c")
