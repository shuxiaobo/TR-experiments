#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf

from .data_file_pairs import dataset_files_pairs
from utils.log import setup_from_args_file, save_args, err


class NLPBase(object):
    """
    Base class for NLP experiments based on tensorflow environment.
    Only do some arguments reading and serializing work.
    """

    def __init__(self):
        self.model_name = self.__class__.__name__
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.sess = tf.Session(config = config)
        # get arguments
        self.args = self.get_args()

        # log set
        logging.basicConfig(filename = self.args.log_file,
                            level = logging.DEBUG,
                            format = '%(asctime)s %(message)s', datefmt = '%y-%m-%d %H:%M')

        # set random seed
        np.random.seed(self.args.random_seed)
        tf.set_random_seed(self.args.random_seed)

        # save arguments
        if not self.args.test:
            save_args(args = self.args)

    def add_args(self, parser):
        """
        If some model need more arguments, override this method.
        """
        pass

    def get_args(self):
        """
        The priority of args:
        [low]       ...    args define in the code
        [middle]    ...    args define in args_file
        [high]      ...    args define in command line
        """

        def str2bool(v):
            if v.lower() in ("yes", "true", "t", "y", "1"):
                return True
            if v.lower() in ("no", "false", "f", "n", "0", "none"):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        def str_or_none(v):
            if not v or v.lower() in ("no", "false", "f", "n", "0", "none", "null"):
                return None
            return v

        def int_or_none(v):
            if not v or v.lower() in ("no", "false", "f", "n", "0", "none", "null"):
                return None
            return int(v)

        # TODO:Implement ensemble test
        parser = argparse.ArgumentParser(description = "Reading Comprehension Experiment Code Base.")
        # -----------------------------------------------------------------------------------------------------------
        group1 = parser.add_argument_group("1.Basic options")
        # basis argument
        group1.add_argument("--debug", default = False, type = str2bool, help = "is debug mode on or off")

        group1.add_argument("--train", default = True, type = str2bool, help = "train or not")

        group1.add_argument("--test", default = False, type = str2bool, help = "test or not")

        group1.add_argument("--ensemble", default = False, type = str2bool, help = "ensemble test or not")

        group1.add_argument("--random_seed", default = 2088, type = int, help = "random seed")

        group1.add_argument("--log_file", default = None, type = str_or_none,
                            help = "which file to save the log,if None,use screen")

        group1.add_argument("--weight_path", default = "weights", help = "path to save all trained models")

        group1.add_argument("--args_file", default = 'args.json', type = str_or_none, help = "json file of current args")

        group1.add_argument("--print_every_n", default = 20, type = int, help = "print performance every n steps")

        group1.add_argument("--save_val", default = False, type = bool, help = "whether save the validation prediction result.")

        group1.add_argument("--tensorboard", default = False, type = bool, help = "whether save tensorboard result.")

        # data specific argument
        group2 = parser.add_argument_group("2.Data specific options")
        # noinspection PyUnresolvedReferences
        import tf.datasets
        group2.add_argument("--dataset", default = "AIC", choices = sys.modules['tf.datasets'].__all__, type = str, help = 'type of the dataset to load')

        group2.add_argument("--embedding_file", default = "../data/processed/AIC/word2vec.300d.txt", type = str_or_none, help = "pre-trained embedding file")

        group2.add_argument("--num_words", default = 35000, type = int, help = "the max number of words in vocabulary")

        group2.add_argument("--skip_top", default = 0, type = int, help = "the max number of words in vocabulary")

        group2.add_argument("--visualize_embedding", default = False, type = bool, help = "visualize word embedding")

        subgroup = group2.add_argument_group("Some default options related to dataset, don't change if it works")

        subgroup.add_argument("--data_root", default = "../data", help = "root path of the dataset")

        subgroup.add_argument("--tmp_dir", default = "../data/processed", help = "dataset specific tmp folder")

        subgroup.add_argument("--train_file", default = "train.txt", help = "train file, if use SQuAD, this arg will be ignore")

        subgroup.add_argument("--valid_file", default = "test.txt", help = "validation file, if use SQuAD, this arg will be ignore")

        subgroup.add_argument("--test_file", default = "test_sub.txt", help = "test file, if use SQuAD, this arg will be ignore")

        subgroup.add_argument("--max_count", default = None, type = int_or_none, help = "read n lines of data file, if Nonqe, read all data")

        subgroup.add_argument("--word_file", default = 'word2id.txt', type = str_or_none, help = "args.word_file")

        subgroup.add_argument("--char_file", default = 'char2id.txt', type = str_or_none, help = "args.char_file")

        # hyper-parameters
        group3 = parser.add_argument_group("3.Hyper parameters shared by all models")

        group3.add_argument("--use_char_embedding", default = False, type = str2bool, help = "use character embedding or not")

        group3.add_argument("--char_embedding_dim", default = 10, type = int, help = "dimension of char embeddings")

        group3.add_argument("--char_hidden_size", default = 10, type = int, help = "dimension of char embedding hidden size")

        group3.add_argument("--max_char_len", default = 20, type = int, help = "the max char length of words")

        group3.add_argument("--embedding_dim", default = 300, type = int, help = "dimension of word embeddings")

        group3.add_argument("--hidden_size", default = 180, type = int, help = "RNN hidden size")

        group3.add_argument("--grad_clipping", default = 10, type = int, help = "the threshold value of gradient clip")

        group3.add_argument("--lr", default = 5e-4, type = float, help = "learning rate")

        group3.add_argument("--keep_prob", default = 0.8, type = float, help = "dropout,percentage to keep during training")

        group3.add_argument("--l2", default = 0.005, type = float, help = "l2 regularization weight")

        group3.add_argument("--num_layers", default = 1, type = int, help = "RNN layer number")

        group3.add_argument("--rnn_type", default = "GRu", type = str_or_none,
                            help = "RNN type, use GRU, LSTM, vanilla rnn, modified rnn, indrnn")

        group3.add_argument("--batch_size", default = 256, type = int, help = "batch_size")

        group3.add_argument("--bidirectional", default = 'true', type = str2bool, help = "whether bidirectional for rnn")

        group3.add_argument("--optimizer", default = "CUS", choices = ["SGD", "ADAM", "ADAD", "CUS"],
                            help = "optimize algorithms, SGD or Adam")

        group3.add_argument("--evaluate_every_n", default = 400, type = int,
                            help = "evaluate performance on validation set and possibly saving the best model")

        group3.add_argument("--num_epoches", default = 100, type = int, help = "max epoch iterations")

        group3.add_argument("--patience", default = 100, type = int, help = "early stopping patience")
        # -----------------------------------------------------------------------------------------------------------
        group4 = parser.add_argument_group("4.model [{}] specific parameters".format(self.model_name))

        group4.add_argument("--activation", default = 'relu', type = str, help = "activation for rnn")
        self.add_args(group4)

        args = parser.parse_args()

        if args.test:
            setup_from_args_file(os.path.join(args.weight_path, args.args_file))
            args.test = True
            args.train = False
            args.keep_prob = 1.

        # set debug params
        args.max_count = 7392 if args.debug else args.max_count
        args.evaluate_every_n = 5 if args.debug else args.evaluate_every_n
        args.num_epoches = 2 if args.debug else args.num_epoches

        args = self.tune_args(args)

        return args

    @staticmethod
    def tune_args(args):
        """
        tune the dataset specific args so train_file or test_file need not be changed
        """
        try:
            files = dataset_files_pairs.get(args.dataset)
            args.data_root, args.train_file, args.valid_file, args.test_file = files
            return args
        except AssertionError:
            err("Error. Cannot find the specific key -> {} in dataset_files_pairs.".format(args.dataset))
