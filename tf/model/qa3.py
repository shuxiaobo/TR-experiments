#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/8
import os
import tensorflow as tf
from base.rnn_base import ModelBase
from tf.model.layers import IndRNNCell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tf.model.modified_rnn import ModifiedRNNCell
from tf.model.layers import VanillaRNNCell
from tensorflow.python.ops import special_math_ops
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, GRUCell, DropoutWrapper


# TODO: 对于一个歧义词，应该利用上下文的几个单词来区分

class QA3(ModelBase):

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
        query_length = tf.reduce_sum(tf.sign(tf.abs(query)), axis = -1)
        alt_length = tf.reduce_sum(tf.sign(tf.abs(alterative)), axis = -1)

        alt_mask = tf.sequence_mask(alt_length, maxlen = self.dataset.alt_max_len, dtype = tf.float32)

        init_embedding = tf.constant(self.embedding_matrix, dtype = tf.float32, name = "embedding_init")
        embedding_matrix = tf.get_variable("embedding_matrix", initializer = init_embedding, dtype = tf.float32, trainable = False)

        self.embedding = embedding_matrix

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
            q_char_embed = tf.nn.dropout(tf.reduce_max(q_char_embed, -1), keep_prob = keep_prob)
            d_char_embed = tf.nn.dropout(tf.reduce_max(d_char_embed, -1), keep_prob = keep_prob)
            with tf.variable_scope('char_rnn', reuse = tf.AUTO_REUSE) as scp:
                # q_char_embed = tf.reshape(q_char_embed, [-1, self.dataset.query_max_len * self.dataset.q_char_len, self.args.char_embedding_dim])
                # d_char_embed = tf.reshape(d_char_embed, [-1, self.dataset.doc_max_len * self.dataset.d_char_len, self.args.char_embedding_dim])

                char_rnn_f = MultiRNNCell(
                    cells = [DropoutWrapper(CELL(num_units = self.args.char_hidden_size, activation = activation), output_keep_prob = keep_prob)])
                char_rnn_b = MultiRNNCell(
                    cells = [DropoutWrapper(CELL(num_units = self.args.char_hidden_size, activation = activation), output_keep_prob = keep_prob)])

                d_char_embed_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = char_rnn_f, cell_bw = char_rnn_b, inputs = d_char_embed,
                                                                      sequence_length = tf.reduce_sum(tf.sign(tf.abs(doc_char_length)), -1),
                                                                      initial_state_bw = None,
                                                                      dtype = "float32", parallel_iterations = None,
                                                                      swap_memory = True, time_major = False, scope = 'char_rnn')
                q_char_embed_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = char_rnn_f, cell_bw = char_rnn_b, inputs = q_char_embed,
                                                                      sequence_length = tf.reduce_sum(tf.sign(tf.abs(query_char_length)), -1),
                                                                      initial_state_bw = None,
                                                                      dtype = "float32", parallel_iterations = None,
                                                                      swap_memory = True, time_major = False, scope = 'char_rnn')
            d_char_out = tf.concat(d_char_embed_out, -1)
            q_char_out = tf.concat(q_char_embed_out, -1)
            # d_char_out = tf.reduce_max(d_char_embed * tf.expand_dims(doc_char_mask, -1), -1)
            # q_char_out = tf.reduce_max(q_char_embed * tf.expand_dims(query_char_mask, -1), -1)

        with tf.variable_scope("query_encoder") as scp:
            query_embed = tf.nn.embedding_lookup(embedding_matrix, query, max_norm = 1.)
            if self.args.use_char_embedding:
                query_embed = tf.concat([query_embed, q_char_out], -1)

            query_inputs = query_embed
            query_last_states_concat = list()
            query_outputs_concat = list()
            for i in range(self.args.num_layers):
                query_inputs = tf.nn.relu(query_inputs)
                cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])

                query_outputs, query_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = query_inputs,
                                                                                   sequence_length = query_length,
                                                                                   initial_state_fw = None, initial_state_bw = None,
                                                                                   dtype = tf.float32, parallel_iterations = None,
                                                                                   swap_memory = True, time_major = False, scope = None)
                query_output_con = tf.concat(query_outputs, -1)
                # self_att_w = tf.get_variable('self_att_w_%d' % i, shape = [query_output_con.get_shape()[-1], 1])
                # att = nn_ops.softmax(tf.squeeze(special_math_ops.einsum('bij,jk->bik', query_output_con, self_att_w), -1), -1)
                # query_output_con = query_output_con * tf.expand_dims(att, -1)

                query_last_states_concat.extend(query_last_states)
                query_outputs_concat.extend(query_outputs)
                query_inputs = tf.concat([query_embed, query_output_con], -1)

            query_outputs = tf.concat(query_outputs_concat, axis = -1)
            query_last_states = tf.concat(query_last_states_concat, axis = -1)
            query_last_states = tf.reshape(query_last_states, shape = [-1, query_last_states.get_shape()[0] * query_last_states.get_shape()[2]])
            query_outputs_dropped = tf.nn.dropout(query_outputs, keep_prob = keep_prob)
            query_last_states_dropped = query_last_states
            query_outputs_max = math_ops.reduce_max(query_outputs, axis = -2)

            query_encoded = query_outputs_max
            query_encoded = tf.nn.dropout(query_encoded, keep_prob = keep_prob)

        with tf.variable_scope('doc_encoder', reuse = tf.AUTO_REUSE) as scp:
            doc_embed = tf.nn.embedding_lookup(embedding_matrix, document, max_norm = 1.)
            if self.args.use_char_embedding:
                doc_embed = tf.concat([doc_embed, d_char_out], -1)
            qry_encoded_dupli = tf.tile(tf.expand_dims(query_encoded, 1), multiples = [1, self.dataset.doc_max_len, 1])
            # doc_embed = tf.nn.dropout(tf.concat([doc_embed, qry_encoded_dupli], -1), keep_prob = keep_prob)

            doc_inputs = tf.nn.dropout(tf.concat([doc_embed, qry_encoded_dupli], -1), keep_prob = keep_prob)
            doc_outputs_concat = list()
            doc_last_states_concat = list()
            for i in range(self.args.num_layers):
                doc_inputs = nn_ops.relu(doc_inputs)
                cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                doc_outputs, doc_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = doc_inputs,
                                                                               sequence_length = doc_length,
                                                                               initial_state_fw = None, initial_state_bw = None,
                                                                               dtype = tf.float32, parallel_iterations = None,
                                                                               swap_memory = True, time_major = False, scope = None)

                # self attention
                # doc_output_con = tf.concat(doc_outputs, -1)
                # self_att_w = tf.get_variable('self_att_w_%d' % i, shape = [doc_output_con.get_shape()[-1], 1])
                # att = nn_ops.softmax(tf.squeeze(special_math_ops.einsum('bij,jk->bik', doc_output_con, self_att_w), -1), -1)
                # doc_output_con = tf.concat(doc_outputs, -1) * tf.expand_dims(att, -1)

                # AOA atted
                att = tf.squeeze(special_math_ops.einsum('bij,bjk->bik', tf.concat(doc_outputs, -1), tf.expand_dims(
                    tf.concat([query_last_states_concat[2 * i][0], query_last_states_concat[2 * i + 1][0]], -1), -1)), -1)
                doc_enc_qry_enc_w = tf.get_variable('doc_enc_qry_enc',
                                                    shape = [tf.concat(doc_outputs, -1).get_shape()[-1], query_outputs.get_shape()[-1]])
                att += tf.reduce_sum(
                    tf.matmul(tf.einsum('bij,jk->bik', tf.concat(doc_outputs, -1), doc_enc_qry_enc_w), tf.transpose(query_outputs, perm = [0, 2, 1])), -1)
                doc_enc_qry_emb_w = tf.get_variable('doc_enc_qry_emb', shape = [tf.concat(doc_outputs, -1).get_shape()[-1], query_embed.get_shape()[-1]])
                att += tf.reduce_sum(
                    tf.matmul(tf.einsum('bij,jk->bik', tf.concat(doc_outputs, -1), doc_enc_qry_emb_w), tf.transpose(query_embed, perm = [0, 2, 1])), -1)
                qry_enc_doc_emb = tf.get_variable('qry_enc_doc_emb', shape = [doc_embed.get_shape()[-1], query_outputs.get_shape()[-1]])
                att += tf.reduce_sum(tf.matmul(tf.einsum('bij,jk->bik', doc_embed, qry_enc_doc_emb), tf.transpose(query_outputs, perm = [0, 2, 1])), -1)

                att = nn_ops.softmax(att, -1)
                doc_output_con = tf.concat(doc_outputs, -1) * tf.expand_dims(att, -1)

                doc_outputs_concat.extend(doc_outputs)
                doc_last_states_concat.extend(doc_last_states)
                doc_inputs = tf.concat([doc_embed, doc_output_con], -1)

            # ELMo s^{task}_j
            # doc_outputs_concat = [tf.expand_dims(dd, 1) for dd in doc_outputs_concat]
            # layer_norm_w = tf.get_variable(name = "layer_norm_w", shape = [self.args.num_layers * 2, 1, 1])
            # layer_norm_w = tf.nn.softmax(layer_norm_w)
            # doc_outputs = tf.concat(doc_outputs_concat, axis = 1) * layer_norm_w
            # doc_outputs = tf.reshape(doc_outputs, shape = [-1, doc_outputs.get_shape()[2], doc_outputs.get_shape()[1] * doc_outputs.get_shape()[-1]])

            doc_outputs = tf.concat(doc_outputs_concat, axis = -1)
            doc_last_states = tf.concat(doc_last_states_concat, axis = -1)
            doc_last_states = tf.reshape(doc_last_states, shape = [-1, doc_last_states.get_shape()[0] * doc_last_states.get_shape()[2]])
            doc_last_states_dropped = tf.nn.dropout(doc_last_states, keep_prob = keep_prob)

            doc_encoded = tf.nn.dropout(doc_outputs, keep_prob = keep_prob)
        with tf.variable_scope("attention") as scp:
            bi_att_w = tf.get_variable('bi_att_w', shape = [doc_encoded.get_shape()[-1], query_encoded.get_shape()[-1]])
            doc_out_query_last_att = tf.squeeze(
                math_ops.matmul(special_math_ops.einsum('bij,jk->bik', doc_encoded, bi_att_w), tf.expand_dims(query_encoded, axis = -1)),
                -1)
            # AOA
            att = tf.reduce_sum(tf.einsum('bij,bjk->bik', doc_encoded, tf.transpose(query_outputs, perm = [0, 2, 1])), -1)
            att += tf.reduce_sum(tf.matmul(doc_embed, tf.transpose(query_embed, perm = [0, 2, 1])), -1)
            doc_enc_qry_emb_w = tf.get_variable('doc_enc_qry_emb', shape = [doc_encoded.get_shape()[-1], query_embed.get_shape()[-1]])
            att += tf.reduce_sum(tf.matmul(tf.einsum('bij,jk->bik', doc_encoded, doc_enc_qry_emb_w), tf.transpose(query_embed, perm = [0, 2, 1])), -1)
            qry_enc_doc_emb = tf.get_variable('qry_enc_doc_emb', shape = [doc_embed.get_shape()[-1], query_outputs.get_shape()[-1]])
            att += tf.reduce_sum(tf.matmul(tf.einsum('bij,jk->bik', doc_embed, qry_enc_doc_emb), tf.transpose(query_outputs, perm = [0, 2, 1])), -1)
            # vanilla
            # att = doc_out_query_last_att

            att = nn_ops.softmax(att, axis = -1)
            doc_atted = doc_encoded * tf.expand_dims(att, -1)  # B * D * 2H
            doc_atted_max = math_ops.reduce_max(doc_atted, axis = -2)

            doc_atted_max = tf.nn.dropout(doc_atted_max, keep_prob = self.args.keep_prob)

        with tf.variable_scope("query_encoder", reuse = tf.AUTO_REUSE) as scp:
            alter_embed = embedding_ops.embedding_lookup(embedding_matrix, alterative, max_norm = 1.)
            # alter_embed_sumed = tf.reduce_max(alter_embed * tf.expand_dims(alt_mask, -1), axis = -2)
            # alter_w = tf.get_variable('alter_w', shape = [self.args.embedding_dim, doc_atted.get_shape()[-1]])
            # alter_b = tf.get_variable('alter_b', shape = [doc_atted.get_shape()[-1]])
            # alter_embed_wxb = special_math_ops.einsum('bij,jk->bik', alter_embed_sumed, alter_w) + alter_b
            # # alter_embed_wxb = alter_embed_wxb * tf.expand_dims(alt_mask, -1)
            # # B * 3 * 2H
            # alter_encoded = alter_embed_wxb
            # alter_encoded = tf.transpose(alter_encoded, perm = [0, 2, 1])
            #
            num_layers = self.args.num_layers
            alt_last_states_concat = list()
            alt_outputs_concat = list()
            for j in range(3):
                alt_last_states_concat_tmp = list()
                alt_outputs_concat_tmp = list()
                alter_input = alter_embed[:, j]
                for i in range(num_layers):
                    alter_input = tf.nn.relu(alter_input)
                    cell_fw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])
                    cell_bw = MultiRNNCell([CELL(num_units = self.args.hidden_size, activation = activation, name = 'rnn_fw_%d' % i)])

                    alter_outputs, alter_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = alter_input,
                                                                                       sequence_length = alt_length[:, j],
                                                                                       initial_state_fw = None, initial_state_bw = None,
                                                                                       dtype = tf.float32, parallel_iterations = None,
                                                                                       swap_memory = True, time_major = False, scope = None)
                    alt_last_states_concat_tmp.extend(alter_last_states)
                    alt_outputs_concat_tmp.extend(alter_outputs)
                    alter_input = tf.concat([alter_embed[:, j], tf.concat(alter_outputs, -1)], -1)
                alt_last_states_concat.append(tf.concat(alt_last_states_concat_tmp, -1))
                alt_outputs_concat.append(tf.concat(alt_outputs_concat_tmp, -1))
            alter_encoded = tf.transpose(tf.concat(alt_last_states_concat, 0), perm = [1, 2, 0])  # tf.stack(alt_outputs_concat, 1)

            alter_encoded = tf.nn.dropout(alter_encoded, keep_prob = keep_prob)

        # with tf.variable_scope("max_pooled_classify") as scp:
        #     # max pooled
        #     result = tf.squeeze(math_ops.matmul(tf.expand_dims(doc_atted_max, -2), alter_encoded), -2)
        #     result = result + tf.squeeze(math_ops.matmul(tf.expand_dims(tf.reduce_max(doc_encoded, 1), 1), alter_encoded), 1)
        #     embed_w = tf.get_variable('embed_w', shape = [doc_embed.get_shape()[-1], alter_encoded.get_shape()[-2]])
        #     result = result + tf.squeeze(tf.matmul(tf.expand_dims(tf.reduce_max(tf.einsum('bij,jk->bik', doc_embed, embed_w), 1), 1), alter_encoded))
        #
        # with tf.variable_scope('hidden_classify') as scp:
        #     # last hidden state
        #     result = tf.squeeze(math_ops.matmul(tf.expand_dims(doc_atted_max, -2), alter_encoded), -2)
        #     result = result + tf.squeeze(math_ops.matmul(tf.expand_dims(doc_last_states_dropped, 1), alter_encoded), 1)
        #     embed_w = tf.get_variable('embed_w', shape = [doc_embed.get_shape()[-1], alter_encoded.get_shape()[-2]])
        #     result = result + tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(tf.reduce_max(doc_embed, 1), embed_w), 1), alter_encoded))
        #
        # with tf.variable_scope('sumed_classify') as scp:
        #     # sumed
        #     result = tf.reduce_sum(math_ops.matmul(doc_atted, alter_encoded), 1)
        #     result = result + tf.reduce_sum(math_ops.matmul(doc_encoded, alter_encoded), 1)
        #     embed_w = tf.get_variable('embed_w', shape = [doc_embed.get_shape()[-1], alter_encoded.get_shape()[-2]])
        #     result = result + tf.reduce_sum(tf.matmul(special_math_ops.einsum('bij,jk-bik', doc_embed, embed_w), alter_encoded), 1)

        with tf.variable_scope('infer_classify') as scp:
            feature = tf.concat(
                [doc_last_states_dropped, query_outputs_max, doc_last_states_dropped - query_outputs_max, doc_last_states_dropped * query_outputs_max], -1)
            embed_w = tf.get_variable('embed_w', shape = [feature.get_shape()[-1], alter_encoded.get_shape()[-2]])
            result = tf.reduce_sum(tf.matmul(tf.expand_dims(math_ops.matmul(feature, embed_w), 1), alter_encoded), 1)

        # result = tf.reduce_sum(special_math_ops.einsum('bij,bjk->bik', doc_atted, alter_encoded), 1)

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
