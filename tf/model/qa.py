#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/8
import os
import tensorflow as tf
from base.rnn_base import ModelBase
from tf.model.layers import IndRNNCell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tf.model.layers import VanillaRNNCell
from tensorflow.python.ops import embedding_ops
from tf.model.modified_rnn import ModifiedRNNCell
from tensorflow.python.ops import special_math_ops
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, GRUCell


class QA(ModelBase):

    def create_model(self):
        document = tf.placeholder(dtype = tf.int64, shape = [None, self.dataset.doc_max_len], name = "document")
        query = tf.placeholder(dtype = tf.int64, shape = [None, self.dataset.query_max_len], name = "query")
        answer = tf.placeholder(dtype = tf.int64, shape = [None], name = "answer")
        alterative = tf.placeholder(dtype = tf.int64, shape = [None, 3, self.dataset.alt_max_len], name = "alternative")
        if self.args.use_char_embedding:
            q_input_char = tf.placeholder(dtype = tf.int32, shape = [None, self.dataset.query_max_len, self.dataset.q_char_len], name = 'query_char')
            d_input_char = tf.placeholder(dtype = tf.int32, shape = [None, self.dataset.doc_max_len, self.dataset.d_char_len], name = 'document_char')

        doc_length = tf.reduce_sum(tf.sign(tf.abs(document)), axis = -1)
        query_length = tf.reduce_sum(tf.sign(tf.abs(query)), axis = -1)
        alt_length = tf.reduce_sum(tf.sign(tf.abs(alterative)), axis = -1)

        alt_mask = tf.sequence_mask(alt_length, maxlen = self.dataset.alt_max_len, dtype = tf.float32)
        embedding_matrix = tf.get_variable("embedding_matrix", shape = [self.dataset.word2id_size, self.args.embedding_dim], dtype = tf.float32)

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

        if self.args.use_char_embedding:
            char_embedding = tf.get_variable(name = 'char_embdding_matrix', shape = [self.dataset.char2id_size, self.args.char_embedding_dim],
                                             dtype = tf.float32,
                                             trainable = True)
            q_char_embed = tf.nn.embedding_lookup(char_embedding, q_input_char)
            d_char_embed = tf.nn.embedding_lookup(char_embedding, d_input_char)
            q_char_embed = tf.nn.dropout(q_char_embed, keep_prob = self.args.keep_prob)
            d_char_embed = tf.nn.dropout(d_char_embed, keep_prob = self.args.keep_prob)
            # with tf.variable_scope('char_rnn', reuse = tf.AUTO_REUSE) as scp:
            #     q_char_embed = tf.reshape(q_char_embed, [-1, self.dataset.query_max_len * self.dataset.q_char_len, self.args.char_embedding_dim])
            #     d_char_embed = tf.reshape(d_char_embed, [-1, self.dataset.doc_max_len * self.dataset.d_char_len, self.args.char_embedding_dim])
            #
            #     char_rnn_f = MultiRNNCell(
            #         cells = [DropoutWrapper(CELL(num_units = self.args.char_hidden_size, activation = activation), output_keep_prob = self.args.keep_prob)])
            #     char_rnn_b = MultiRNNCell(
            #         cells = [DropoutWrapper(CELL(num_units = self.args.char_hidden_size, activation = activation), output_keep_prob = self.args.keep_prob)])
            #
            #     d_char_embed_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = char_rnn_f, cell_bw = char_rnn_b, inputs = d_char_embed,
            #                                                           sequence_length = doc_length, initial_state_bw = None,
            #                                                           dtype = "float32", parallel_iterations = None,
            #                                                           swap_memory = True, time_major = False, scope = 'char_rnn')
            #     q_char_embed_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = char_rnn_f, cell_bw = char_rnn_b, inputs = q_char_embed,
            #                                                           sequence_length = query_length, initial_state_bw = None,
            #                                                           dtype = "float32", parallel_iterations = None,
            #                                                           swap_memory = True, time_major = False, scope = 'char_rnn')

            with tf.variable_scope('char_conv', reuse = tf.AUTO_REUSE) as scp:
                q_char_embed = tf.transpose(q_char_embed, perm = [0, 2, 3, 1])  # [batch, height, width, channels]
                filter = tf.get_variable('q_filter_w',
                                         shape = [5, 5, self.dataset.query_max_len,
                                                  self.dataset.query_max_len])  # [filter_height, filter_width, in_channels, out_channels]
                cnned_char = tf.nn.conv2d(q_char_embed, filter, strides = [1, 1, 1, 1], padding = 'VALID', use_cudnn_on_gpu = True,
                                          data_format = "NHWC",
                                          name = None)  # [B, (char_len-filter_size/stride), (word_len-filter_size/stride), d_len]

                q_char_embed_out = tf.nn.max_pool(cnned_char, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'VALID',
                                                  data_format = "NHWC",
                                                  name = None)

                char_out_size = q_char_embed_out.get_shape().as_list()[1] * q_char_embed_out.get_shape().as_list()[2]
                q_char_embed_out = tf.reshape(tf.transpose(q_char_embed_out, perm = [0, 3, 1, 2]),
                                              shape = [-1, self.dataset.query_max_len, char_out_size])

                d_char_embed = tf.transpose(d_char_embed, perm = [0, 2, 3, 1])  # [batch, height, width, channels]
                filter = tf.get_variable('d_filter_w',
                                         shape = [5, 5, self.dataset.doc_max_len,
                                                  self.dataset.doc_max_len])  # [filter_height, filter_width, in_channels, out_channels]
                cnned_char = tf.nn.conv2d(d_char_embed, filter, strides = [1, 1, 1, 1], padding = 'VALID', use_cudnn_on_gpu = True,
                                          data_format = "NHWC",
                                          name = None)  # [B, (char_len-filter_size/stride), (word_len-filter_size/stride), d_len]

                d_char_embed_out = tf.nn.max_pool(cnned_char, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'VALID',
                                                  data_format = "NHWC",
                                                  name = None)
                char_out_size = d_char_embed_out.get_shape().as_list()[1] * d_char_embed_out.get_shape().as_list()[2]
                d_char_embed_out = tf.reshape(tf.transpose(d_char_embed_out, perm = [0, 3, 1, 2]),
                                              shape = [-1, self.dataset.doc_max_len, char_out_size])

                d_char_embed_out = tf.reshape(d_char_embed_out, shape = [-1, self.dataset.doc_max_len, char_out_size])
                d_char_out = tf.concat(d_char_embed_out, -1)
                q_char_out = tf.concat(q_char_embed_out, -1)

        with tf.variable_scope("query_encoder") as scp:
            query_embed = tf.nn.embedding_lookup(embedding_matrix, query)
            if self.args.use_char_embedding:
                query_embed = tf.concat([query_embed, q_char_out], -1)
            cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])
            cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation) for _ in range(self.args.num_layers)])

            query_outputs, query_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = query_embed,
                                                                               sequence_length = query_length,
                                                                               initial_state_fw = None, initial_state_bw = None,
                                                                               dtype = tf.float32, parallel_iterations = None,
                                                                               swap_memory = True, time_major = False, scope = None)
            query_outputs = tf.concat(query_outputs, axis = -1)
            query_last_states = tf.concat(query_last_states, axis = -1)
            query_last_states = tf.reshape(query_last_states, shape = [-1, query_last_states.get_shape()[0] * query_last_states.get_shape()[2]])
            query_outputs_dropped = tf.nn.dropout(query_outputs, keep_prob = self.args.keep_prob)
            query_last_states_dropped = query_last_states
            query_outputs_max = math_ops.reduce_max(query_outputs_dropped, axis = -2)

            query_encoded = query_outputs_max
            query_encoded = tf.nn.dropout(query_encoded, keep_prob = self.args.keep_prob)

        with tf.variable_scope('doc_encoder') as scp:
            doc_embed = tf.nn.embedding_lookup(embedding_matrix, document)
            if self.args.use_char_embedding:
                doc_embed = tf.concat([doc_embed, d_char_out], -1)
            qry_encoded_dupli = tf.tile(tf.expand_dims(query_encoded, 1), multiples = [1, self.dataset.doc_max_len, 1])
            doc_embed = tf.nn.dropout(tf.concat([doc_embed, qry_encoded_dupli], -1), keep_prob = self.args.keep_prob)

            doc_inputs = doc_embed
            doc_outputs_concat = list()

            for i in range(self.args.num_layers):
                cell_fw = MultiRNNCell(
                    [CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                cell_bw = MultiRNNCell(
                    [CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                doc_outputs, doc_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = doc_inputs,
                                                                               sequence_length = doc_length,
                                                                               initial_state_fw = None, initial_state_bw = None,
                                                                               dtype = tf.float32, parallel_iterations = None,
                                                                               swap_memory = True, time_major = False, scope = None)
                doc_outputs_concat.extend(doc_outputs)
                doc_inputs = tf.concat([doc_embed, tf.concat(doc_outputs, -1)], -1)

            # ELMo s^{task}_j
            # doc_outputs_concat = [tf.expand_dims(dd, 1) for dd in doc_outputs_concat]
            # layer_norm_w = tf.get_variable(name = "layer_norm_w", shape = [self.args.num_layers * 2, 1, 1])
            # layer_norm_w = tf.nn.softmax(layer_norm_w)
            # doc_outputs = tf.concat(doc_outputs_concat, axis = 1) * layer_norm_w
            # doc_outputs = tf.reshape(doc_outputs, shape = [-1, doc_outputs.get_shape()[2], doc_outputs.get_shape()[1] * doc_outputs.get_shape()[-1]])
            doc_outputs = tf.concat(doc_outputs_concat, axis = -1)
            doc_last_states = tf.concat(doc_last_states, axis = -1)
            doc_last_states = tf.reshape(doc_last_states, shape = [-1, doc_last_states.get_shape()[0] * doc_last_states.get_shape()[2]])
            doc_last_states_dropped = tf.nn.dropout(doc_last_states, keep_prob = self.args.keep_prob)

            doc_encoded = doc_outputs
        with tf.variable_scope("attention") as scp:
            bi_att_w = tf.get_variable('bi_att_w', shape = [doc_encoded.get_shape()[-1], query_encoded.get_shape()[-1]])
            doc_out_query_last_att = nn_ops.softmax(
                tf.squeeze(math_ops.matmul(special_math_ops.einsum('bij,jk->bik', doc_encoded, bi_att_w), tf.expand_dims(query_encoded, axis = -1)),
                           -1),
                axis = -1)

            doc_atted = doc_encoded * tf.expand_dims(doc_out_query_last_att, -1)  # B * D * 2H
            doc_atted_max = math_ops.reduce_max(doc_atted, axis = -2)

        with tf.variable_scope("alter_encoder") as scp:
            alter_embed = embedding_ops.embedding_lookup(embedding_matrix, alterative)
            alter_embed_sumed = tf.reduce_max(alter_embed * tf.expand_dims(alt_mask, -1), axis = -2)
            alter_w = tf.get_variable('alter_w', shape = [self.args.embedding_dim, doc_atted.get_shape()[-1]])
            alter_b = tf.get_variable('alter_b', shape = [doc_atted.get_shape()[-1]])
            alter_embed_wxb = special_math_ops.einsum('bij,jk->bik', alter_embed_sumed, alter_w)
            # alter_embed_wxb = alter_embed_wxb * tf.expand_dims(alt_mask, -1)
            # B * 3 * 2H
            alter_encoded = tf.nn.dropout(alter_embed_wxb, keep_prob = self.args.keep_prob)

        with tf.variable_scope("classify") as scp:
            # clsy = ClassifiedNet(input_size = doc_atted_max.get_shape()[-1], output_size = 3)
            # clsy_x = tf.tile(tf.expand_dims(doc_atted_max, 1), [1, alter_encoded.get_shape()[1], 1])
            # result = tf.reduce_sum(clsy(clsy_x, alter_encoded), -2)
            # result = tf.squeeze(math_ops.matmul(tf.expand_dims(doc_atted_max, -2), tf.transpose(alter_embed_wxb, perm = [0, 2, 1])), -2)
            result = tf.reduce_sum(special_math_ops.einsum('bij,bjk->bik', alter_encoded, tf.transpose(doc_atted, perm = [0, 2, 1])), -1)

        self.correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(result, -1), answer), tf.int32))

        self.loss = tf.reduce_mean(nn_ops.sparse_softmax_cross_entropy_with_logits(logits = result, labels = answer))

        self.accuracy = self.correct_prediction / tf.shape(document)[0]

        self.prediction = tf.argmax(result, -1)

    def test_save(self, pred):
        ids = list()
        answers = list()
        for i, data in enumerate(zip(*self.dataset.test_x)):
            ids.append(data[-2])
            answers.append(data[-1][pred[i]])
        with open(os.path.join(self.args.weight_path, 'submission.txt'), mode = 'w+', encoding = 'utf-8') as f:
            for i, a in zip(ids, answers):
                f.write(str(i) + '\t' + a + '\n')
        print("Save submission over...")
