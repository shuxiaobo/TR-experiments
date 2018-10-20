#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/10/15

import math
from base.rnn_base import ModelBase
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from base.rnn_base import ModelBase
from tf.model.layers import IndRNNCell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tf.model.modified_rnn import ModifiedRNNCell
from tf.model.layers import VanillaRNNCell
from tensorflow.python.ops import special_math_ops
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, GRUCell, DropoutWrapper



class QA4(ModelBase):

    def create_model(self):
        keep_prob = tf.placeholder(name = 'keep_prob', dtype = tf.float32)
        answer = tf.placeholder(dtype = tf.int64, shape = [None], name = "answer")
        query = tf.placeholder(dtype = tf.int64, shape = [None, self.dataset.query_max_len], name = "query")
        document = tf.placeholder(dtype = tf.int64, shape = [None, self.dataset.doc_max_len], name = "document")
        alterative = tf.placeholder(dtype = tf.int64, shape = [None, 3, self.dataset.alt_max_len], name = "alternative")
        if self.args.use_char_embedding:
            q_input_char = tf.placeholder(dtype = tf.int32, shape = [None, self.dataset.query_max_len, self.dataset.q_char_len], name = 'query_char')
            d_input_char = tf.placeholder(dtype = tf.int32, shape = [None, self.dataset.doc_max_len, self.dataset.d_char_len], name = 'document_char')
            doc_char_length = tf.reduce_sum(tf.sign(tf.abs(d_input_char)), axis = -1)
            query_char_length = tf.reduce_sum(tf.sign(tf.abs(q_input_char)), axis = -1)
            doc_char_mask = tf.sequence_mask(doc_char_length, maxlen = self.dataset.d_char_len, dtype = tf.float32)
            query_char_mask = tf.sequence_mask(query_char_length, maxlen = self.dataset.q_char_len, dtype = tf.float32)

        doc_length = tf.reduce_sum(tf.sign(tf.abs(document)), axis = -1)
        doc_mask = tf.sequence_mask(doc_length, maxlen = self.dataset.doc_max_len, dtype = tf.float32)
        doc_mask = tf.expand_dims(doc_mask, -1)

        query_length = tf.reduce_sum(tf.sign(tf.abs(query)), axis = -1)
        query_mask = tf.sequence_mask(query_length, maxlen = self.dataset.query_max_len, dtype = tf.float32)
        query_mask = tf.expand_dims(query_mask, -1)

        alt_length = tf.reduce_sum(tf.sign(tf.abs(alterative)), axis = -1)
        alt_mask = tf.sequence_mask(alt_length, maxlen = self.dataset.alt_max_len, dtype = tf.float32)
        alt_mask = tf.expand_dims(alt_mask, -1)

        init_embedding = tf.constant(self.embedding_matrix, dtype = tf.float32, name = "embedding_init")
        embedding_matrix = tf.get_variable("embedding_matrix", initializer = init_embedding, dtype = tf.float32, trainable = False)

        self.embedding = embedding_matrix

        _ran = np.arange(0, self.dataset.doc_max_len, dtype = np.float32)
        div_term = np.exp(np.arange(0, self.args.embedding_dim, 2, dtype = np.float32) * (-np.log(10000.0) / self.args.embedding_dim))
        pem = np.zeros(shape = [self.dataset.doc_max_len, self.args.embedding_dim])
        pem[:, 0::2] = np.sin(np.expand_dims(_ran, -1) * np.expand_dims(div_term, 0))
        pem[:, 1::2] = np.cos(np.expand_dims(_ran, -1) * np.expand_dims(div_term, 0))
        position_embedding = tf.get_variable(name = 'position_embedding', initializer = pem.astype(np.float32), dtype = tf.float32,
                                             trainable = False)

        if self.args.rnn_type.lower() == 'modified':
            CELL = ModifiedRNNCell
        elif self.args.rnn_type.lower() == 'lstm':
            CELL = LSTMCell
        elif self.args.rnn_type.lower() == 'gru':
            CELL = GRUCell
        elif self.args.rnn_type.lower() == 'vanilla':
            CELL = VanillaRNNCell
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

        with tf.variable_scope('embedding') as scp:
            doc_embed = tf.nn.embedding_lookup(embedding_matrix, document)
            query_embed = tf.nn.embedding_lookup(embedding_matrix, query)
            alter_embed = tf.nn.embedding_lookup(embedding_matrix, alterative)

            doc_position = tf.tile(tf.expand_dims(tf.nn.embedding_lookup(position_embedding, tf.range(0, self.dataset.doc_max_len)), 0),
                                   [tf.shape(doc_embed)[0], 1, 1]) * doc_mask
            doc_embed = tf.concat([doc_embed * doc_position, doc_embed + doc_position, doc_embed, doc_position], -1)

            query_position = tf.tile(tf.expand_dims(tf.nn.embedding_lookup(position_embedding, tf.range(0, self.dataset.query_max_len)), 0),
                                     [tf.shape(doc_embed)[0], 1, 1]) * query_mask
            query_embed = tf.concat([query_embed * query_position, query_embed + query_position, query_embed, query_position], -1)
            alter_embed = tf.reduce_sum(alter_embed * alt_mask, -2)
        with tf.variable_scope('encoder') as scp:

            cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw')])
            cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_bw')])

            query_outputs, query_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = query_inputs,
                                                                               sequence_length = query_length,
                                                                               initial_state_fw = None, initial_state_bw = None,
                                                                               dtype = tf.float32, parallel_iterations = None,
                                                                               swap_memory = True, time_major = False, scope = None)
        with tf.variable_scope('attention') as scp:
            attened = list()
            for i in range(self.args.num_layers):
                with tf.variable_scope('layers_%d' % i) as scp:
                    # self_atten
                    self_atten_w = tf.get_variable(name = 'self_atten_w_%d' % i, shape = [doc_embed.get_shape()[-1], doc_embed.get_shape()[-1]])
                    doc_input = tf.einsum('bij,jk->bik', doc_embed, self_atten_w)

                    # multihead attention
                    a1 = tf.get_variable('mean_%d' % i, shape = [doc_input.get_shape()[-1]])
                    b1 = tf.get_variable('std_%d' % i, shape = [doc_input.get_shape()[-1]])
                    eps = 1e-6
                    mean, std = tf.nn.moments(doc_input, [-1], keep_dims = True)

                    doc_input = (a1 * (doc_input - mean)) / (eps + std) + b1

                    num_head = 5
                    hs = doc_input.get_shape()[-1] // num_head
                    doc_embed_view = tf.reshape(doc_input, [-1, self.dataset.doc_max_len, num_head, hs])
                    # query_embed_view = tf.reshape(query_embed, [tf.shape(query_embed)[0], -1, num_head, hs])

                    self_atten = tf.matmul(doc_embed_view, tf.transpose(doc_embed_view, perm = [0, 1, 3, 2]))
                    self_atten = self_atten / math.sqrt(self.args.embedding_dim * 2)
                    self_atten = tf.nn.softmax(self_atten, -1)
                    doc_embed_attened = tf.reshape(tf.matmul(self_atten, doc_embed_view), tf.shape(doc_input))

                    atten2 = tf.matmul(doc_input, tf.transpose(query_embed, perm = [0, 2, 1])) / math.sqrt(self.args.embedding_dim * 2)
                    qry_w = tf.get_variable('qry_w', shape = [self.dataset.query_max_len, doc_embed_attened.get_shape()[-1]])
                    atten2 = tf.nn.softmax(tf.expand_dims(tf.reduce_max(atten2), -1), -1)

                    doc_embed_attened = doc_embed_attened * atten2
                    attened.append(tf.reduce_max(doc_embed_attened, -2))
        doc_embed_attened = tf.concat(attened, -1)
        with tf.variable_scope('classify') as scp:

            alt_w = tf.get_variable('alt_w', shape = [alter_embed.get_shape()[-1], doc_embed_attened.get_shape()[-1]])
            result = tf.nn.softmax(tf.squeeze(tf.matmul(tf.einsum('bij,jk->bik', alter_embed, alt_w), tf.expand_dims(doc_embed_attened, -1))), -1)

        self.correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(result, -1), answer), tf.int32))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = result, labels = answer))

        self.accuracy = self.correct_prediction / tf.shape(document)[0]

        self.prediction = tf.argmax(result, -1)
