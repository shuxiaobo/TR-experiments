#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
import abc
import os
import sys

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
# noinspection PyUnresolvedReferences
from .nlp_base import NLPBase
from utils.log import logger, save_obj_to_json, err
from utils.util import visualize_embedding
import datetime
from tf.model.optimizer import create_optimizer


# noinspection PyAttributeOutsideInit
class ModelBase(NLPBase, metaclass = abc.ABCMeta):
    """
    Base class of TR experiments.
    Reads different TR datasets according to specific class.
    creates a model and starts training it.
    Any deep learning model should inherit from this class and implement the create_model method.
    """

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @correct_prediction.setter
    def correct_prediction(self, value):
        self._correct_prediction = value

    @property
    def accuracy(self):
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value):
        self._accuracy = value

    def metric(self, preds, label):
        """
        Implement the method to set your own metric, while use them in evaluation
        :param preds: list
        :param label: list
        :return:
        """
        pass

    def test_save(self, pred):
        """
        save the test result
        if you wanna to save the result, please override the method
        :return:
        """
        return

    def get_train_op(self):
        """
        define optimization operation
        """
        if self.args.optimizer == 'CUS':
            num_train_steps = int(self.dataset.train_nums / self.args.batch_size * self.args.num_epoches)
            num_warmup_steps = int(num_train_steps * 0.1)
            self.train_op = create_optimizer(self.loss, self.args.lr, num_train_steps, num_warmup_steps, False)
            return
        if self.args.optimizer == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.args.lr)
        elif self.args.optimizer == "ADAM":
            optimizer = tf.train.AdamOptimizer(learning_rate = self.args.lr)
        elif self.args.optimizer == 'ADAD':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate = self.args.lr, rho = 0.95, epsilon = 1e-06, )
        else:
            raise NotImplementedError("Other Optimizer Not Implemented.-_-||")

        grad_vars = optimizer.compute_gradients(self.loss)
        if self.args.grad_clipping != 0:
            grad_vars = [
                (tf.clip_by_norm(grad, self.args.grad_clipping), var)
                if grad is not None else (grad, var)
                for grad, var in grad_vars]
        for g, v in grad_vars:
            if g is not None:
                tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        self.train_op = optimizer.apply_gradients(grad_vars, self.step)
        return

    @abc.abstractmethod
    def create_model(self):
        """
        should be override by sub-class and create some operations include [loss, correct_prediction]
        as class attributes.
        """
        return

    def execute(self):
        """
        main method to train and test
        """
        # self.confirm_model_dataset_fitness()

        self.dataset = getattr(sys.modules["tf.datasets"], self.args.dataset)(self.args)

        if hasattr(self.dataset, 'get_embedding_matrix'):
            self.embedding_matrix = self.dataset.get_embedding_matrix(is_char_embedding = False)
        else:
            logger('No function named get_embedding_matrix in data set %s, use the random initialization' % self.dataset.__class__.__name__)

        self.max_len = self.dataset.max_len
        self.word2id_size = self.dataset.word2id_size
        self.train_nums, self.valid_nums, self.test_num = self.dataset.train_nums, self.dataset.valid_nums, self.dataset.test_nums

        self.num_class = self.dataset.num_class

        self.create_model()

        self.make_sure_model_is_valid()

        self.saver = tf.train.Saver(max_to_keep = 20)

        if self.args.train:
            if self.args.tensorboard:
                self.draw_graph()
            self.train()
        if self.args.test:
            self.test()

        self.sess.close()

    def draw_graph(self):
        log_file = '../logs/log-%s-%s-%s-emb%d-id%s' % (
            self.args.activation, self.args.dataset, self.args.rnn_type, self.args.embedding_dim, str(datetime.datetime.now()))
        self.writer = tf.summary.FileWriter(log_file)
        self.writer.add_graph(self.sess.graph)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = 'embedding_matrix'
        # embedding_conf.metadata_path = os.path.join(log_file, 'metadata.tsv')
        projector.visualize_embeddings(self.writer, config)

        self.merged_summary = tf.summary.merge_all()
        logger('Save log to %s' % log_file)

    def get_batch_data(self, mode, idx):
        """
        Get batch data and feed it to tensorflow graph
        Modify it in sub-class if needed.
        """
        return self.dataset.get_next_batch(mode, idx)

    def train(self):
        """
        train model
        """
        self.step = tf.Variable(0, name = "global_step", trainable = False)
        batch_size = self.args.batch_size
        epochs = self.args.num_epoches
        self.get_train_op()  # get the optimizer op
        self.sess.run(tf.global_variables_initializer())
        # self.load_weight()  # load the trained model

        # early stopping params, by default val_acc is the metric
        self.patience, self.best_val_acc = self.args.patience, 0.
        # Start training
        corrects_in_epoch, samples_in_epoch, loss_in_epoch = 0, 0, 0

        batch_num = self.train_nums // batch_size
        logger("Train on {} batches, {} samples per batch, {} total.".format(batch_num, batch_size, self.train_nums))

        step = self.sess.run(self.step)
        while step < batch_num * epochs:

            step = self.sess.run(self.step)
            # on Epoch start
            data, samples = self.get_batch_data("train", step % batch_num)
            # TODO : here can be remove if the test being desperated
            data = dict(data, **{'keep_prob:0': self.args.keep_prob})

            loss, _, corrects = self.sess.run([self.loss, self.train_op, self.correct_prediction],
                                              feed_dict = data)
            corrects_in_epoch += corrects.item()
            loss_in_epoch += loss * samples.item()
            samples_in_epoch += samples.item()

            # logger self.sess.run([self.result_s,self.result_e],data)
            if step % self.args.print_every_n == 0:
                logger("Samples : {}/{}.\tStep : {}/{}.\tLoss : {:.4f}.\tAccuracy : {:.4f}.".format(
                    samples_in_epoch, self.train_nums,
                    step % batch_num, batch_num,
                    loss_in_epoch / samples_in_epoch,
                    corrects_in_epoch / samples_in_epoch))
                if self.args.tensorboard:
                    s = self.sess.run(self.merged_summary, feed_dict = data)
                    self.writer.add_summary(s, step)
                    self.writer.flush()

            if step % batch_num == 0:
                corrects_in_epoch, samples_in_epoch, loss_in_epoch = 0, 0, 0
                logger("{}Epoch : {}{}".format("-" * 40, step // batch_num + 1, "-" * 40))
                # self.dataset.shuffle()

            if step and step % batch_num == 0:
                # evaluate on the valid set and early stopping
                val_acc, val_loss = self.validate()
                self.early_stopping(val_acc, val_loss, step)
                self.best_val_acc = max(self.best_val_acc, val_acc)
            self.sess.graph.finalize()

    def validate(self):
        batch_size = self.args.batch_size
        v_batch_num = self.valid_nums // batch_size
        # ensure the entire valid set is selected
        v_batch_num = v_batch_num + 1 if (self.valid_nums % batch_size) != 0 else v_batch_num
        # logger("Validate on {} batches, {} samples per batch, {} total."
        #        .format(v_batch_num, batch_size, self.valid_nums))
        val_num, val_corrects, v_loss = 0, 0, 0

        preds = list()
        for i in range(v_batch_num):
            data, samples = self.get_batch_data("valid", i)

            # TODO : here can be remove if the test being desperated
            data = dict(data, **{'keep_prob:0': 1.})

            if samples != 0:
                loss, v_correct, prediction = self.sess.run([self.loss, self.correct_prediction, self.prediction], feed_dict = data)
                val_num += samples
                val_corrects += v_correct
                v_loss += loss * samples
                preds.extend(prediction.tolist())

        # call the custom metric
        # self.metric(preds = preds, label = self.dataset.valid_y.tolist())

        assert (val_num == self.valid_nums)
        val_acc = val_corrects / val_num
        val_loss = v_loss / val_num
        logger("Evaluate on : {}/{}.\tVal acc : {:.4f}.\tVal Loss : {:.4f}, Best acc:{:.4f}, , Dataset:{}, ".format(val_num,
                                                                                                                    self.valid_nums,
                                                                                                                    val_acc,
                                                                                                                    val_loss, self.best_val_acc,
                                                                                                                    self.args.dataset))
        return val_acc, val_loss

    # noinspection PyUnusedLocal
    def early_stopping(self, val_acc, val_loss, step):
        if val_acc > self.best_val_acc:
            self.patience = self.args.patience
            self.best_val_acc = val_acc
            if self.args.save_val:
                self.save_weight(val_acc, step)
        elif self.patience == 1:
            logger("Oh u, stop training.")
            exit(0)
        else:
            self.patience -= 1
            logger("Remaining/Patience : {}/{} .".format(self.patience, self.args.patience))

    def save_weight(self, val_acc, step):
        path = self.saver.save(self.sess,
                               os.path.join(self.args.weight_path,
                                            "{}-val_acc-{:.4f}.models-{}".format(self.model_name, val_acc, datetime.datetime.now())),
                               global_step = step)
        if self.args.tensorboard and self.args.visualize_embedding:
            visualize_embedding(word2id = self.dataset.word2id, embedding_matrix_name = self.embedding.name, writer = self.writer)
        logger("Save models to {}.".format(path))

    def load_weight(self):
        ckpt = tf.train.get_checkpoint_state(self.args.weight_path)
        if ckpt is not None and ckpt.model_checkpoint_path.startswith(os.path.join(self.args.weight_path, self.__class__.__name__)):
            logger("Load models from {}.".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logger("No previous models. model :%s" % self.__class__.__name__)

    def test(self):
        if not self.args.train:
            self.sess.run(tf.global_variables_initializer())
            self.load_weight()
        batch_size = self.args.batch_size
        batch_num = self.test_num // batch_size
        batch_num = batch_num + 1 if (self.test_num % batch_size) != 0 else batch_num
        correct_num, total_num = 0, 0
        result = list()
        for i in range(batch_num):
            data, samples = self.get_batch_data("test", i)
            # TODO : here can be remove if the test being desperated
            data = dict(data, **{'keep_prob:0': 1.})

            if samples != 0:
                correct, pred = self.sess.run([self.correct_prediction, self.prediction], feed_dict = data)
                correct_num, total_num = correct_num + correct, total_num + samples
                result.extend(pred.tolist())
        assert (total_num == self.test_num == len(result))
        logger("Test on : {}/{}".format(total_num, self.test_num))
        test_acc = correct_num / total_num
        logger("Test accuracy is : {:.5f}".format(test_acc))
        res = {
            "model": self.model_name,
            "test_acc": test_acc
        }
        self.test_save(pred = result)
        save_obj_to_json(self.args.weight_path, res, "result.json")

    def confirm_model_dataset_fitness(self):
        # make sure the models_in_datasets var is correct
        try:
            assert (models_in_datasets.get(self.args.dataset, None) is not None)
        except AssertionError:
            err("Models_in_datasets doesn't have the specified dataset key: {}.".format(self.args.dataset))
            self.sess.close()
            exit(1)
        # make sure the model fit the dataset
        try:
            assert (self.model_name in models_in_datasets.get(self.args.dataset, None))
        except AssertionError:
            err("The model -> {} doesn't support the dataset -> {}".format(self.model_name, self.args.dataset))
            self.sess.close()
            exit(1)

    def make_sure_model_is_valid(self):
        """
        check if the model has necessary attributes
        """
        try:
            _ = self.loss
            _ = self.correct_prediction
            _ = self.prediction
            _ = self.embedding
            if self.args.visualize_embedding:
                _ = self.embedding
        except AttributeError as e:
            err("Your model {} doesn't have enough attributes.\nError Message:\n\t{}".format(self.model_name, e))
            self.sess.close()
            exit(1)
