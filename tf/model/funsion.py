#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/19
from __future__ import absolute_import
import tensorflow as tf
from .layers import IndRNNCell
from .layers import ModifiedRNNCell
from base.rnn_base import ModelBase
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, GRUCell, RNNCell


class Fusion(ModelBase):

    def create_model(self):
        RECURRENT_MAX_ABS = pow(2, 1 / self.max_len)
        x = tf.placeholder(name = 'document', shape = [None, self.max_len], dtype = tf.int32)
        y_true = tf.placeholder(name = 'y_true', shape = [None], dtype = tf.int64)
        keep_prob = tf.placeholder(name = 'keep_prob', dtype = tf.float32)

        embedding = tf.get_variable('embedding', initializer = tf.random_normal_initializer,
                                    shape = [self.word2id_size, self.args.embedding_dim],
                                    dtype = tf.float32)

        x_length = tf.reduce_sum(tf.sign(tf.abs(x)), axis = -1)

        if self.args.rnn_type.lower() == 'modified':
            CELL = ModifiedRNNCell
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
            x_emb = tf.nn.embedding_lookup(params = embedding, ids = x)
            cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_')])
            cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_')])

            def reduce(i, xx):
                i = i + 1

                x_outputs, x_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = x_emb[:, i - 1:i + 1, :],
                                                                           sequence_length = tf.reduce_sum(tf.sign(tf.abs(x[:, i - 1:i + 1])), axis = -1),
                                                                           initial_state_fw = None, initial_state_bw = None,
                                                                           dtype = tf.float32, parallel_iterations = None,
                                                                           swap_memory = True, time_major = False, scope = None)

                return [i, tf.concat(x_outputs, -1)]

            # _, s = tf.scan(fn = reduce, elems = [tf.transpose(x_emb, perm = [1, 0, 2])],
            #                initializer = [tf.constant(0), tf.zeros(shape = [tf.shape(x_emb)[0], 3, tf.shape(x_emb)[-1]])], parallel_iterations = 10,
            #                back_prop = True,
            #                swap_memory = False, infer_shape = False, name = None)
            i = tf.constant(0)
            _, s = tf.while_loop(cond = lambda x, y: x < tf.shape(y)[1], body = reduce, loop_vars = [i, x_emb],
                                 shape_invariants = [tf.TensorShape([]), tf.TensorShape([None, self.max_len, 2 * self.args.hidden_size])],
                                 parallel_iterations = 10, back_prop = True, swap_memory = False,
                                 name = None, maximum_iterations = None)

            s = tf.transpose(s, perm = [1, 0, 2])
            x_last_states_concat = list()
            x_outputs_concat = list()
            x_inputs = x_emb
            for i in range(self.args.num_layers):
                x_inputs = tf.nn.relu(x_inputs)
                cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])

                x_outputs, x_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = x_inputs,
                                                                           sequence_length = x_length,
                                                                           initial_state_fw = None, initial_state_bw = None,
                                                                           dtype = tf.float32, parallel_iterations = None,
                                                                           swap_memory = True, time_major = False, scope = None)
                x_last_states_concat.extend(x_last_states)
                x_outputs_concat.extend(x_outputs)
                # layer_w0 = tf.get_variable(name = 'layer0_%d' % i, shape = [x_emb.get_shape()[-1], x_emb.get_shape()[-1]])
                # layer_w1 = tf.get_variable(name = 'layer1_%d' % i, shape = [x_emb.get_shape()[-1], x_emb.get_shape()[-1]])
                # x_emb_temp = tf.einsum('bij,jk->bik', x_emb, layer_w0)
                # x_emb_temp = tf.nn.relu(tf.einsum('bij,jk->bik', x_emb_temp, layer_w1))
                x_inputs = tf.nn.dropout(tf.concat([x_emb, s, tf.concat(x_outputs, -1)], -1), keep_prob = keep_prob)

            x_encoded = tf.squeeze(tf.concat(x_last_states_concat, -1), 0)
            x_encoded = x_encoded

        with tf.variable_scope('classify', reuse = False) as f:
            w_shape = [x_encoded.get_shape()[-1], self.num_class]
        cls_w = tf.get_variable('w', shape = w_shape, dtype = tf.float32, initializer = tf.random_uniform_initializer)
        bias = tf.get_variable('bias', shape = [self.num_class], initializer = tf.random_uniform_initializer, dtype = tf.float32)
        result = tf.nn.xw_plus_b(x = x_encoded, weights = cls_w, biases = bias)
        # result = tf.matmul(x_encoded, cls_w)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = result, labels = y_true))

        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(result, -1), y_true), dtype = tf.int32)))

        self.accuracy = self.correct_prediction / tf.shape(x)[0]
