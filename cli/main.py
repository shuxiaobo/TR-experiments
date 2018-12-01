#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
from __future__ import absolute_import
import argparse
import torch
import sys
import os
import copy
from pprint import pprint
import random
import numpy as np
from torch.multiprocessing import Process, Pool, Process, set_start_method

sys.path.append(os.getcwd() + '/..')

from cli.train import train
from cli.preprocess import preprocess
from cli.train_snli import train as train_nli


def main():
    parser = argparse.ArgumentParser(description = "Knowledge Inference Code Base.")

    group1 = parser.add_argument_group("1.Basic options")
    # basis argument
    group1.add_argument("--mode", default = 1, type = int, help = "run mode, 0 for preprocess, 1 for train")

    group1.add_argument("--debug", default = False, type = bool, help = "is debug mode on or off")

    group1.add_argument("--train", default = True, type = bool, help = "train or not")

    group1.add_argument("--test", default = False, type = bool, help = "test or not")

    group1.add_argument("--ensemble", default = False, type = bool, help = "ensemble test or not")

    group1.add_argument("--log_file", default = None, type = str,
                        help = "which file to save the log,if None,use screen")

    group1.add_argument("--weight_path", default = "weights", help = "path to save all trained models")

    group1.add_argument("--args_file", default = None, type = str, help = "json file of current args")

    group1.add_argument("--print_every_n", default = 10, type = int, help = "print performance every n steps")

    group1.add_argument("--save_val", default = True, type = bool, help = "whether save the validation prediction result.")

    group1.add_argument('--random_seed', default = 1111, type = int)

    group1.add_argument("--gpu", default = 0, type = int, help = "GPU id be used.")



    group1.add_argument('--log_dir', default = '../logs/', type = str, help = 'tensorboard log dir')
    # data specific argument
    group2 = parser.add_argument_group("2.Data specific options")
    # noinspection PyUnresolvedReferences
    group2.add_argument("--embedding_file", default = "../data/vec.txt",
                        type = str, help = "pre-trained embedding file")
    # group2.add_argument("--embedding_file", default = "../data/glove/glove.6B.100d.txt",
    #                     type = str, help = "pre-trained embedding file")

    group2.add_argument("--data_root", default = "../data/", help = "root path of the dataset")

    group2.add_argument("--tmp_dir", default = "../data/processed/", help = "dataset specific tmp folder")

    group2.add_argument("--train_file", default = "train.txt", help = "train file path under data root")

    group2.add_argument("--valid_file", default = "dev.txt", help = "validation file path under data root")

    group2.add_argument("--test_file", default = "test.txt", help = "test file path under data root")

    group2.add_argument("--max_count", default = None, type = int, help = "read n lines of data file, if None, read all data")

    group2.add_argument("--word_file", default = "word2id.txt", type = str, help = "word file under data root")

    # train hyper-parameters
    group3 = parser.add_argument_group("3. train common hyper-parameters for model")

    group3.add_argument("--batch_size", default = 128, type = int, help = "batch size for train")

    group3.add_argument("--lr", default = 3e-3, type = float, help = "lr for model learning")

    group3.add_argument("--keep_prob", default = 0.8, type = float, help = "the keep prob")

    group3.add_argument("--optimizer", default = "ADAM", choices = ["SGD", "ADAM", "ADAD"], help = "optimize algorithms, SGD or Adam")

    group3.add_argument("--hidden_size", default = 100, type = int, help = "RNN hidden size")

    group3.add_argument("--embedding_dim", default = 200, type = int, help = "dimension of word embeddings")

    group3.add_argument("--grad_clipping", default = 0, type = int, help = "the threshold value of gradient clip")

    group3.add_argument("--num_epoches", default = 200, type = int, help = "max epoch iterations")

    group3.add_argument("--num_words", default = 30000, type = int, help = "max length of the sentences")

    group3.add_argument("--skip_top", default = 20, type = int, help = "max length of the sentences")

    # -----------------------------------------------------------------------------------------------------------
    group4 = parser.add_argument_group("4.model specific parameters")

    group4.add_argument('--kernel_size', default = 3, type = int, help = 'kernel size for n-gram.')

    group4.add_argument('--stride', default = 2, type = int, help = 'stride size for n-gram.')

    group4.add_argument('--bidirectional', default = True, type = bool, help = 'Use the bi-directional rnn.')

    group4.add_argument('--task', default = 1, type = int, help = 'task 1 for nli, 0 for classify')

    group4.add_argument('--num_layers', default = 1, type = int, help = 'number of layers')

    group4.add_argument('--activation', default = 'relu', type = str, help = 'activation function for RNN ')

    group4.add_argument('--dataset', default = 'trec', type = str, help = 'activation function for RNN ')

    group4.add_argument('--target', default = 0.0, type = float, help = '')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    pprint(vars(args), indent = 4)
    if args.mode == 1:
        if args.task == 0:
            i, d, e = train(args)
            print(' %s : acc %.4f, epoch : %d, target %.4f  \n' % (d, i, e, args.target))
        elif args.task == 1:
            print(' %s : acc %.4f, epoch : %d, target %.4f  \n' % (train_nli(args)))
        else:
            import datetime
            print(datetime.datetime.now())
            args.save_val = False
            all_dataset = [('cr', 80.5), ('mr', 75.7), ('trec', 89.0), ('mpqa', 83.7), ('sst', 81.4), ('subj', 91.4), ]
            acces = list()

            for d, t in all_dataset:
                args.dataset = d
                acces.append(train(args) + [t])
            print('===' * 10 + '***' + '===' * 10 +'\n')
            print('|name\t|accuracy\t|epoch\t|target\t|\n| :------| ------: | ------: |------: |')
            for i, d, e, t in acces:
                info = '' if i * 100 < t else 'Y'
                print('| %s | %.4f| %d| %.4f| %s |' % (d, i, e, t, info))
            print(datetime.datetime.now())

    elif args.mode == 0:
        preprocess(args)
    else:
        raise ValueError("no value of mode arg being : {}".format(str(args.mode)))


if __name__ == '__main__':
    main()
