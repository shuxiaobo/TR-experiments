#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
from __future__ import absolute_import
import tensorflow as tf
from .layers import IndRNNCell
from .layers import ModifiedRNNCell
from tf.base.rnn_base import ModelBase
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import MultiRNNCell


class ModifiedRNN(ModelBase):

    def create_model(self):
        RECURRENT_MAX_ABS = pow(2, 1 / self.max_len)
        x = tf.placeholder(name = 'document', shape = [None, self.max_len], dtype = tf.int32)
        y_true = tf.placeholder(name = 'y_true', shape = [None], dtype = tf.int64)

        embedding = tf.get_variable('embedding', initializer = tf.random_normal_initializer,
                                    shape = [self.word2id_size, self.args.embedding_dim],
                                    dtype = tf.float32)

        doc_lens = tf.reduce_sum(tf.sign(tf.abs(x)), axis = -1)

        with tf.variable_scope('encoder') as s:
            x_emb = tf.nn.embedding_lookup(params = embedding, ids = x)

            if self.args.bidirectional:
                cell_fw = MultiRNNCell([
                    ModifiedRNNCell(num_units = self.args.hidden_size, activation = self.args.activation) for _ in range(self.args.num_layers)])
                cell_bw = MultiRNNCell([
                    ModifiedRNNCell(num_units = self.args.hidden_size, activation = self.args.activation) for _ in range(self.args.num_layers)])
                # fw_initializer = tf.random_normal_initializer(-RECURRENT_MAX_ABS, RECURRENT_MAX_ABS)
                # bw_initializer = tf.random_normal_initializer(-RECURRENT_MAX_ABS, RECURRENT_MAX_ABS)
                # cell_fw = MultiRNNCell([
                #     IndRNNCell(num_units = self.args.hidden_size, recurrent_max_abs = RECURRENT_MAX_ABS, recurrent_kernel_initializer = fw_initializer)
                #     for _ in range(self.args.num_layers)])
                # cell_bw = MultiRNNCell([
                #     IndRNNCell(num_units = self.args.hidden_size, recurrent_max_abs = RECURRENT_MAX_ABS, recurrent_kernel_initializer = bw_initializer)
                #     for _ in range(self.args.num_layers)])

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
