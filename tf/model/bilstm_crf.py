#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/10/10
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import logging
from utils.util import logger
import pickle
from base.rnn_base import ModelBase


class LSTMSRLer(ModelBase):
    def add_args(self, parser):
        parser.add_argument("--postag_dim", default = 20, type = int, help = "pos tag dim")
        parser.add_argument("--distance_dim", default = 20, type = int, help = "distance tag dim")
        parser.add_argument("--n_label", default = 67, type = int, help = "number of label")
        parser.add_argument("--use_crf", default = True, type = int, help = "use crf")

    def make_transition_rule(self):
        # make a matrix to indicate whether a transition is legal
        trans = np.ones(shape = (67, 67), dtype = "float32")
        # B flag can only be followed with I or E with same name
        trans[:17, :] = 0
        trans[:17, 17:34] = np.diag(np.ones(17))
        trans[:11, 34:45] = np.diag(np.ones(11))
        trans[12:17, 45:50] = np.diag(np.ones(5))

        # E, O, S, rel flag can be followed with any labels except E and I
        trans[17:34, 17:50] = 0
        trans[50:67, 17:50] = 0

        # I flag can only be followed with I or E with same name
        trans[34:50, :] = 0
        trans[34:50, 34:50] = np.diag(np.ones(16))
        trans[34:45, 17:28] = np.diag(np.ones(11))
        trans[45:50, 29:34] = np.diag(np.ones(5))

        self.transition_rules = trans

    def create_model(self):
        # define placeholder
        keep_prob = tf.placeholder(name = 'keep_prob', dtype = tf.float32)
        curword = tf.placeholder(name = 'curword', dtype = "int32", shape = (None, self.dataset.max_len))
        lastword = tf.placeholder(name = 'lastword', dtype = "int32", shape = (None, self.dataset.max_len))
        nextword = tf.placeholder(name = 'nextword', dtype = "int32", shape = (None, self.dataset.max_len))
        predicate = tf.placeholder(name = 'predicate', dtype = "int32", shape = (None, self.dataset.max_len))
        curpostag = tf.placeholder(name = 'curpostag', dtype = "int32", shape = (None, self.dataset.max_len))
        lastpostag = tf.placeholder(name = 'lastpostag', dtype = "int32", shape = (None, self.dataset.max_len))
        nextpostag = tf.placeholder(name = 'nextpostag', dtype = "int32", shape = (None, self.dataset.max_len))
        label = tf.placeholder(name = 'label', dtype = "int32", shape = (None, self.dataset.max_len))  # sparse encode
        mask = tf.placeholder(name = 'mask', dtype = "int32", shape = (None, self.dataset.max_len))  # 0 for padding words

        distance = tf.placeholder(name = 'distance', dtype = "int32", shape = (None, self.dataset.max_len))

        seq_length = tf.placeholder(name = 'seq_length', dtype = "int32", shape = (None,))

        # initialize embedding variables
        word_embedding = tf.get_variable("Embed_word", shape = (self.dataset.word2id_size, self.args.embedding_dim), dtype = 'float32',
                                         initializer = tf.random_normal_initializer())
        self.embedding = word_embedding
        postag_embedding = tf.get_variable("Embed_postag", shape = (34, self.args.postag_dim), dtype = 'float32',
                                           initializer = tf.random_normal_initializer())  # postag size is fixed to 32+2(eos, bos)
        distance_embedding = tf.get_variable("Embed_dist", shape = (240, self.args.distance_dim), dtype = 'float32',
                                             initializer = tf.random_normal_initializer())  # we observed in training set max dist is 240

        # initialize fully connected variables
        total_embed_dim = 4 * self.args.embedding_dim + 3 * self.args.postag_dim + self.args.distance_dim

        W = tf.get_variable("W_3", shape = (2 * self.args.hidden_size, self.args.n_label), initializer = tf.random_normal_initializer())
        b = tf.get_variable("b_3", shape = (self.args.n_label,), initializer = tf.random_normal_initializer())

        # initialize RNN cell
        fw_lstmcells = []
        bw_lstmcells = []

        for _i in range(self.args.num_layers):
            fw_lstmcells.append(tf.nn.rnn_cell.BasicLSTMCell(num_units = self.args.hidden_size))
            bw_lstmcells.append(tf.nn.rnn_cell.BasicLSTMCell(num_units = self.args.hidden_size))

        # initialize transition rule
        self.make_transition_rule()

        # get representation
        curword_emb = tf.nn.embedding_lookup(word_embedding, curword)
        lastword_emb = tf.nn.embedding_lookup(word_embedding, lastword)
        nextword_emb = tf.nn.embedding_lookup(word_embedding, nextword)
        predicate_emb = tf.nn.embedding_lookup(word_embedding, predicate)
        curpos_emb = tf.nn.embedding_lookup(postag_embedding, curpostag)
        lastpos_emb = tf.nn.embedding_lookup(postag_embedding, lastpostag)
        nextpos_emb = tf.nn.embedding_lookup(postag_embedding, nextpostag)
        dist_emb = tf.nn.embedding_lookup(distance_embedding, distance)
        embedding = tf.concat([curword_emb, lastword_emb, nextword_emb, predicate_emb, curpos_emb, lastpos_emb, nextpos_emb, dist_emb], axis = 2)

        # first fully connected layer
        total_embed_dim = 4 * self.args.embedding_dim + 3 * self.args.postag_dim + self.args.distance_dim

        # recurrent layer
        if self.args.num_layers > 1:
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_lstmcells, bw_lstmcells, embedding,
                                                                           sequence_length = seq_length,
                                                                           dtype = "float32")
            hidden_rnn = outputs[:, :, -2 * self.args.hidden_size:]
        else:
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstmcells[0], bw_lstmcells[0], embedding,
                                                         seq_length,
                                                         dtype = "float32")
            hidden_rnn = tf.concat(outputs, axis = 2)

        # output layer
        logits = tf.einsum('bij,jk->bik', hidden_rnn, W) + b  # (batch_size * max_len, n_label)
        if self.args.use_crf:
            if True:
                inputs = logits
                outputs, transition_params = tf.contrib.crf.crf_log_likelihood(inputs, label, seq_length)
                self.loss = -tf.reduce_sum(outputs)
                tf.summary.scalar('loss', self.loss)
                outputs, viterbi_score = tf.contrib.crf.crf_decode(logits, transition_params, seq_length)
                # self.train_op = self.getOptimizer(self.config['optimizer'], self.config['lrate']).minimize(self.loss)
                self.merged_summary = tf.summary.merge_all()  # record loss
            # else:
            #     outputs = tf.reshape(logits, (-1, self.dataset.max_len, self.config.n_label))
        else:
            if True:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = logits)
                outputs = tf.argmax(logits, -1)
                loss = tf.reshape(loss, (-1, self.dataset.max_len))
                loss = loss * tf.cast(mask, "float32")
                self.loss = tf.reduce_sum(loss)
                tf.summary.scalar('loss', self.loss)
                # initialize training op
                # self.train_op = self.getOptimizer(self.config['optimizer'], self.config['lrate']).minimize(self.loss)
                self.merged_summary = tf.summary.merge_all()  # record loss
            # else:
            #     outputs = tf.reshape(logits, (-1, self.dataset.max_len, self.args.n_label))

        self.correct_prediction = tf.reduce_sum(tf.cast(tf.equal(outputs, label), tf.int32))

        self.accuracy = self.correct_prediction / tf.shape(label)[0]

        self.prediction = outputs

    def modify(self, sequence):
        lastname = ""
        for i in range(len(sequence)):
            tagid = sequence[i]
            tag = self.idx2label[tagid]
            if tag[0] == "B":
                lastname = tag[2:]
                continue
            if tag[0] == "I" or tag[0] == "E":
                if tag[2:] != lastname:
                    lastname = tag[2:]
                    sequence[i] = self.label2idx["B-" + lastname]
        return sequence

    def find_best(self, scores, length, dist):
        scores = scores.T  # (n_label, max_len)
        record = np.full(shape = (67, length), fill_value = -np.Inf)
        path = np.full(shape = (67, length), fill_value = -1, dtype = np.int32)
        pred = []
        for i in range(length):
            if i == 0:
                if dist[i] != 0:
                    record[0:17, i] = scores[0:17, i]
                    record[50:66, i] = scores[50:66, i]
                else:
                    record[66, i] = scores[66, i]
            else:
                if dist[i] != 0 and dist[i - 1] != 0:
                    for j in range(66):
                        max_score = -np.Inf
                        argmax_prev = -1
                        for k in range(66):
                            if self.transition_rules[k, j] and record[k, i - 1] + scores[j, i] > max_score:
                                max_score = record[k, i - 1] + scores[j, i]
                                argmax_prev = k
                        record[j, i] = max_score
                        path[j, i] = argmax_prev
                elif dist[i] != 0 and dist[i - 1] == 0:
                    record[0:17, i] = scores[0:17, i] + record[66, i - 1]
                    path[0:17, i] = 66
                    record[50:66, i] = scores[50:66, i] + record[66, i - 1]
                    path[50:66, i] = 66
                else:
                    max_score = -np.Inf
                    argmax_prev = -1
                    for k in range(66):
                        if self.transition_rules[k, 66] and record[k, i - 1] + scores[66, i] > max_score:
                            max_score = record[k, i - 1] + scores[66, i]
                            argmax_prev = k
                    record[66, i] = max_score
                    path[66, i] = argmax_prev
                if i == length - 1:
                    record[0:17, i] = -np.Inf
                    path[0:17, i] = -1
                    record[34:50, i] = -np.Inf
                    path[34:50, i] = -1
        move = np.argmax(record[:, -1])
        pred.append(move)
        while len(pred) < length:
            assert record[move, length - len(pred)] != -np.Inf and path[move, length - len(pred)] != -1
            move = path[move, length - len(pred)]
            pred.append(move)
        pred.reverse()
        return pred

    def write_outputs(self, preds, idx2label):
        fout = open(config['output_path'], "w", encoding = 'utf-8')
        fp = open(os.path.join(self.args.tmp_dir, self.__class__.__name__, data_path + self.args.valid_file), "r", encoding = 'utf-8')
        data = fp.readlines()
        data = [line.strip().split(" ") for line in data]
        for i in range(len(data)):
            line = ''
            for j in range(len(data[i])):
                line += "/".join([data[i][j].split("/")[0], data[i][j].split("/")[1], idx2label[preds[i][j]]]) + " "
            fout.write(line + "\n")
        fout.close()
        fp.close()

    def metric(self, preds, label):
        logger('Starting evaluate custom metric...')
        case_true, case_recall, case_precision = 0, 0, 0
        assert len(label) == len(preds), "length of prediction file and gold file should be the same. Receive:%d, should %d" % (len(label), len(preds))
        for gold, pred in zip(label, preds):
            lastname = ''
            keys_gold, keys_pred = {}, {}
            for item in gold:
                word, label = item.split('/')[0], item.split('/')[-1]
                flag, name = label[:label.find('-')], label[label.find('-') + 1:]
                if flag == 'O':
                    continue
                if flag == 'S':
                    if name not in keys_gold:
                        keys_gold[name] = [word]
                    else:
                        keys_gold[name].append(word)
                else:
                    if flag == 'B':
                        if name not in keys_gold:
                            keys_gold[name] = [word]
                        else:
                            keys_gold[name].append(word)
                        lastname = name
                    elif flag == 'I' or flag == 'E':
                        assert name == lastname, "the I-/E- labels are inconsistent with B- labels in gold file."
                        keys_gold[name][-1] += ' ' + word
            for item in pred:
                word, label = item.split('/')[0], item.split('/')[-1]
                flag, name = label[:label.find('-')], label[label.find('-') + 1:]
                if flag == 'O':
                    continue
                if flag == 'S':
                    if name not in keys_pred:
                        keys_pred[name] = [word]
                    else:
                        keys_pred[name].append(word)
                else:
                    if flag == 'B':
                        if name not in keys_pred:
                            keys_pred[name] = [word]
                        else:
                            keys_pred[name].append(word)
                        lastname = name
                    elif flag == 'I' or flag == 'E':
                        assert name == lastname, "the I-/E- labels are inconsistent with B- labels in pred file."
                        keys_pred[name][-1] += ' ' + word

            for key in keys_gold:
                case_recall += len(keys_gold[key])
            for key in keys_pred:
                case_precision += len(keys_pred[key])

            for key in keys_pred:
                if key in keys_gold:
                    for word in keys_pred[key]:
                        if word in keys_gold[key]:
                            case_true += 1
                            keys_gold[key].remove(word)  # avoid replicate words
        assert case_recall != 0, "no labels in gold files!"
        assert case_precision != 0, "no labels in pred files!"
        recall = 1.0 * case_true / case_recall
        precision = 1.0 * case_true / case_precision
        f1 = 2.0 * recall * precision / (recall + precision)
        result = "recall: %s  precision: %s  F: %s" % (str(recall), str(precision), str(f1))
        logger(result)
