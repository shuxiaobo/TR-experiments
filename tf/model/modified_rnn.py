#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
from __future__ import absolute_import
import tensorflow as tf
from .layers import IndRNNCell
from .layers import ModifiedRNNCell
from base.rnn_base import ModelBase
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, GRUCell, RNNCell
from tf.model.layers import *


class ModifiedRNN(ModelBase):

    def create_model(self):
        RECURRENT_MAX_ABS = pow(2, 1 / self.max_len)
        x = tf.placeholder(name = 'document', shape = [None, self.max_len], dtype = tf.int64)
        y_true = tf.placeholder(name = 'y_true', shape = [None], dtype = tf.int64)
        keep_prob = tf.placeholder(name = 'keep_prob', dtype = tf.float32)

        embedding = tf.get_variable('embedding', initializer = tf.random_normal_initializer,
                                    shape = [self.word2id_size, self.args.embedding_dim],
                                    dtype = tf.float32)
        self.embedding = embedding
        doc_lens = tf.reduce_sum(tf.sign(tf.abs(x)), axis = -1)

        if self.args.rnn_type.lower() == 'modified':
            CELL = ContextGRUCell
        elif self.args.rnn_type.lower() == 'lstm':
            CELL = LSTMCell
        elif self.args.rnn_type.lower() == 'gru':
            CELL = GRUCell
        elif self.args.rnn_type.lower() == 'vanilla':
            CELL = RNNCell
        elif self.args.rnn_type.lower() == 'indrnn':
            CELL = IndRNNCell
        else:
            raise NotImplementedError("No rnn_type named : %s implemented. Check." % self.args.rnn_type)

        if self.args.activation == 'sigmoid':
            activation = math_ops.sigmoid
        elif self.args.activation == 'relu':
            activation = nn_ops.relu
        elif self.args.activation == 'tanh':
            activation = math_ops.tanh
        elif self.args.activation == 'log':
            activation = math_ops.log
        elif self.args.activation == 'sin':
            activation = math_ops.sin
        elif self.args.activation == 'none':
            activation = lambda yy: yy
        else:
            raise NotImplementedError("No activation named : %s implemented. Check." % self.args.rnn_type)

        with tf.variable_scope('encoder') as s:
            if self.args.rnn_type.lower() == 'modified':
                x_emb = ContextEmbedding(method = 'concat')(x = x, embedding_matrix = embedding)
            else:
                x_emb = tf.nn.embedding_lookup(params = embedding, ids = x)
            if self.args.bidirectional:
                cell_fw = MultiRNNCell([
                    CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])
                cell_bw = MultiRNNCell([
                    CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])

                outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = x_emb, sequence_length = doc_lens,
                                                                          initial_state_fw = None, initial_state_bw = None,
                                                                          dtype = tf.float32, parallel_iterations = None,
                                                                          swap_memory = False, time_major = False, scope = None)

                outputs = tf.concat(outputs, -1)
                out_maxpooled = tf.reduce_max(outputs, axis = -2)
            else:
                fw_initializer = tf.random_normal_initializer(-RECURRENT_MAX_ABS, RECURRENT_MAX_ABS)
                cell_fw = [
                    IndRNNCell(num_units = self.args.embedding_dim, recurrent_max_abs = RECURRENT_MAX_ABS, recurrent_kernel_initializer = fw_initializer)
                    for _ in range(self.args.num_layers)]

                outputs, outputs_states = tf.nn.dynamic_rnn(cell = cell_fw, inputs = x_emb, sequence_length = doc_lens, initial_state = None,
                                                            dtype = None, parallel_iterations = None, swap_memory = False,
                                                            time_major = False, scope = None)
                out_maxpooled = tf.reduce_max(outputs, axis = -2)

        with tf.variable_scope('classify', reuse = False) as f:
            w_shape = [self.args.hidden_size * 2 if self.args.bidirectional else self.args.hidden_size, self.num_class]
        cls_w = tf.get_variable('w', shape = w_shape, dtype = tf.float32, initializer = tf.random_uniform_initializer)
        # bias = tf.get_variable('bias', shape = [self.num_class], initializer = tf.random_uniform_initializer, dtype = tf.float32)
        # result = tf.nn.xw_plus_b(x = out_maxpooled, weights = cls_w, biases = bias)
        result = tf.matmul(out_maxpooled, cls_w)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = result, labels = y_true))

        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(result, -1), y_true), dtype = tf.int32)))

        self.accuracy = self.correct_prediction / tf.shape(x)[0]

        self.prediction = tf.argmax(result, -1)
