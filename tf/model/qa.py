#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/8
import tensorflow as tf
from tf.base.rnn_base import ModelBase
from tf.model.layers import IndRNNCell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tf.model.modified_rnn import ModifiedRNNCell
from tensorflow.python.ops import special_math_ops
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, GRUCell, RNNCell


class QA(ModelBase):

    def create_model(self):
        document = tf.placeholder(dtype = tf.int64, shape = [None, self.dataset.doc_max_len], name = "document")
        query = tf.placeholder(dtype = tf.int64, shape = [None, self.dataset.query_max_len], name = "query")
        answer = tf.placeholder(dtype = tf.int64, shape = [None], name = "answer")
        alterative = tf.placeholder(dtype = tf.int64, shape = [None, 3, self.dataset.alt_max_len], name = "alternative")

        doc_length = tf.reduce_sum(tf.sign(tf.abs(document)), axis = -1)
        query_length = tf.reduce_sum(tf.sign(tf.abs(query)), axis = -1)
        alt_length = tf.reduce_sum(tf.sign(tf.abs(alterative)), axis = -1)

        embedding_matrix = tf.get_variable("embedding_matrix", shape = [self.dataset.query_max_len, self.args.embedding_dim], dtype = tf.float32)

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

        with tf.variable_scope('doc_encoder') as scp:
            doc_embed = tf.nn.embedding_lookup(embedding_matrix, document)

            cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])
            cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])
            doc_outputs, doc_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = doc_embed,
                                                                           sequence_length = doc_length,
                                                                           initial_state_fw = None, initial_state_bw = None,
                                                                           dtype = tf.float32, parallel_iterations = None,
                                                                           swap_memory = False, time_major = False, scope = None)
            doc_outputs = tf.concat(doc_outputs, axis = -1)
            doc_last_states = tf.concat(doc_last_states, axis = -1)
            doc_last_states = tf.reshape(doc_last_states, shape = [-1, doc_last_states.get_shape()[0] * doc_last_states.get_shape()[2]])
            doc_outputs_dropped = tf.nn.dropout(doc_outputs, keep_prob = self.args.keep_prob)
            doc_last_states_dropped = tf.nn.dropout(doc_last_states, keep_prob = self.args.keep_prob)

        with tf.variable_scope("query_encoder") as scp:
            query_embed = tf.nn.embedding_lookup(embedding_matrix, query)

            cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])
            cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])

            query_outputs, query_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = query_embed,
                                                                               sequence_length = query_length,
                                                                               initial_state_fw = None, initial_state_bw = None,
                                                                               dtype = tf.float32, parallel_iterations = None,
                                                                               swap_memory = False, time_major = False, scope = None)
            query_outputs = tf.concat(query_outputs, axis = -1)
            query_last_states = tf.concat(query_last_states, axis = -1)
            query_last_states = tf.reshape(doc_last_states, shape = [-1, query_last_states.get_shape()[0] * query_last_states.get_shape()[2]])
            query_outputs_dropped = tf.nn.dropout(query_outputs, keep_prob = self.args.keep_prob)
            query_last_states_dropped = tf.nn.dropout(query_last_states, keep_prob = self.args.keep_prob)
            query_outputs_max = math_ops.reduce_max(query_outputs_dropped, axis = -2)

        with tf.variable_scope("attention") as scp:
            doc_out_query_last_att = nn_ops.softmax(tf.squeeze(math_ops.matmul(doc_outputs_dropped, tf.expand_dims(query_outputs_max, axis = -1)), -1),
                                                    axis = -1)

            doc_atted = doc_outputs * tf.expand_dims(doc_out_query_last_att, -1)
            doc_atted_max = math_ops.reduce_max(doc_atted, axis = -2)

        with tf.variable_scope("alter_encoder") as scp:
            alter_embed = embedding_ops.embedding_lookup(embedding_matrix, alterative)
            alter_embed_sumed = tf.reduce_sum(alter_embed, axis = -2)
            alter_w = tf.get_variable('w', shape = [self.args.embedding_dim, self.args.hidden_size * 2])
            alter_b = tf.get_variable('b', shape = [self.args.hidden_size * 2])
            alter_embed_wxb = special_math_ops.einsum('bij,jk->bik', alter_embed_sumed, alter_w) + alter_b

        with tf.variable_scope("classify") as scp:
            result = tf.squeeze(math_ops.matmul(tf.expand_dims(doc_atted_max, -2), tf.transpose(alter_embed_wxb, perm = [0, 2, 1])), -2)

        self.correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(result, -1), answer), tf.int32))

        self.loss = tf.reduce_mean(nn_ops.sparse_softmax_cross_entropy_with_logits(logits = result, labels = answer))

        self.accuracy = self.correct_prediction / tf.shape(document)[0]