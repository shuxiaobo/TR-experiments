'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import logging
from torch.utils.data import Dataset
from utils.util import prepare_dictionary, logger, prepare_split, write_file
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from utils.util import logger


class BinaryClassifierEval(Dataset):
    def __init__(self, args, is_train = True, num_class = 2, seed = 1111, file_name = ''):
        self.seed = seed
        self.args = args
        self.num_class = num_class
        self.max_len = 0
        file = self.args.train_file if is_train else self.args.test_file
        self.data_x, self.data_y = self.load_file(os.path.join(self.args.tmp_dir, self.__class__.__name__, file_name + file))
        sorted_corpus = sorted(zip(self.data_x, self.data_y),
                               key = lambda z: (len(z[0]), z[1]))
        self.data_x = [x for (x, y) in sorted_corpus]
        self.data_y = [y for (x, y) in sorted_corpus]

        self.n_samples = len(self.data_x)
        self.word_file = os.path.join(args.tmp_dir, self.__class__.__name__, file_name + args.word_file)
        if os.path.isfile(self.word_file) and os.path.getsize(self.word_file) > 0:
            self.word2id = self.get_word_index(self.word_file)
        else:
            self.word2id = self.prepare_dict(self.word_file)

    def load_file(self, fpath):
        with io.open(fpath, 'r', encoding = 'utf-8') as f:
            data_x = list()
            data_y = list()
            for line in f.read().splitlines():
                line = line.strip().split(' ')
                if len(line) <= 3:
                    continue
                data_x.append(line[:-1])
                data_y.append(int(line[-1]))
                self.max_len = len(line[:-1]) if len(line[:-1]) > self.max_len else self.max_len
        return data_x, data_y

    def prepare_dict(self, file_name):
        logger("Prepare the dictionary for the {}...".format(self.__class__.__name__))
        word2id = prepare_dictionary(data = self.data_x, dict_path = file_name, exclude_n = self.args.skip_top, max_size = self.args.num_words)
        logger("Word2id size : %d" % len(word2id))
        return word2id

    def get_word_index(self, path = None):
        if not path:
            path = self.args.tmp_dir + self.__class__.__name__ + self.args.word_file
        word2id = dict()
        with open(path, mode = 'r', encoding = 'utf-8') as f:
            for l in f:
                word2id.setdefault(l.strip(), len(word2id))
        logger('Word2id size : %d' % len(word2id))
        return word2id

    def __getitem__(self, index):
        result = [self.word2id[d] if self.word2id.get(d) else self.word2id['_<UNKNOW>'] for d in self.data_x[index]]
        return result, self.data_y[index]

    def __len__(self):
        return self.n_samples

    def data_split(self, args):
        trainx, trainy, testx, testy = prepare_split(self.train_data_x, self.train_data_y, validation_split = 0.2)
        train = [x + [str(y)] for x, y in zip(trainx, trainy)]
        test = [x + [str(y)] for x, y in zip(testx, testy)]
        write_file(data = train, path = os.path.join(args.tmp_dir, self.__class__.__name__, args.train_file))
        write_file(data = test, path = os.path.join(args.tmp_dir, self.__class__.__name__, args.test_file))

    @staticmethod
    def batchfy_fn(data):
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        max_len = max(map(len, x))
        return sequence.pad_sequences(x, maxlen = max_len, padding = 'post'), y


class CR(BinaryClassifierEval):
    def __init__(self, args, is_train = True, seed = 1111):
        logging.debug('***** Transfer task : CR *****\n\n')
        super(self.__class__, self).__init__(args, is_train, seed)


class MR(BinaryClassifierEval):
    def __init__(self, args, is_train = True, seed = 1111):
        logging.debug('***** Transfer task : MR *****\n\n')
        super(self.__class__, self).__init__(args, is_train, seed)


class SUBJ(BinaryClassifierEval):
    def __init__(self, args, is_train = True, seed = 1111):
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        super(self.__class__, self).__init__(args, is_train, seed)


class MPQA(BinaryClassifierEval):
    def __init__(self, args, is_train = True, seed = 1111):
        logging.debug('***** Transfer task : MPQA *****\n\n')
        super(self.__class__, self).__init__(args, is_train, seed)


class Kaggle(BinaryClassifierEval):
    def __init__(self, args, is_train = True, seed = 1111):
        logging.debug('***** Transfer task : Kaggle *****\n\n')
        super(Kaggle, self).__init__(args, is_train, num_class = 5, seed = seed)


class TREC(BinaryClassifierEval):
    def __init__(self, args, is_train = True, seed = 1111):
        logging.info('***** Transfer task : TREC *****\n\n')
        self.seed = seed
        super(TREC, self).__init__(args = args, is_train = is_train, num_class = 6)
    #     self.train = self.loadFile(os.path.join(args.tmp_dir, self.__class__.__name__, 'train_5500.label'))
    #     self.test = self.loadFile(os.path.join(args.tmp_dir, self.__class__.__name__, 'TREC_10.label'))
    #     write_file(data = self.train, path = os.path.join(args.tmp_dir, self.__class__.__name__, args.train_file))
    #     write_file(data = self.test, path = os.path.join(args.tmp_dir, self.__class__.__name__, args.test_file))
    #
    # def loadFile(self, fpath):
    #     trec_data = []
    #     tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
    #                'HUM': 3, 'LOC': 4, 'NUM': 5}
    #     with io.open(fpath, 'r', encoding = 'latin-1') as f:
    #         for line in f:
    #             target, sample = line.strip().split(':', 1)
    #             sample = sample.split(' ', 1)[1].split()
    #             assert target in tgt2idx, target
    #             trec_data.append(sample + [str(tgt2idx[target])])
    #     return trec_data
    # #


class SST(BinaryClassifierEval):
    def __init__(self, args, is_train = True, nclasses = 5, seed = 1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)

        tmp = 'binary/' if nclasses == 2 else 'fine/'
        super(SST, self).__init__(args, is_train, num_class = nclasses, file_name = tmp)
    #     self.train = self.loadFile(os.path.join(args.tmp_dir, self.__class__.__name__, tmp, 'sentiment-train'))
    #     self.dev = self.loadFile(os.path.join(args.tmp_dir, self.__class__.__name__, tmp, 'sentiment-dev'))
    #     self.test = self.loadFile(os.path.join(args.tmp_dir, self.__class__.__name__, tmp, 'sentiment-test'))
    #     self.sst_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
    #
    #     write_file(data = self.train, path = os.path.join(args.tmp_dir, self.__class__.__name__, tmp, args.train_file))
    #     write_file(data = self.test, path = os.path.join(args.tmp_dir, self.__class__.__name__, tmp, args.test_file))
    #     write_file(data = self.dev, path = os.path.join(args.tmp_dir, self.__class__.__name__, tmp, args.valid_file))
    #
    # def loadFile(self, fpath):
    #     sst_data = []
    #     with io.open(fpath, 'r', encoding = 'utf-8') as f:
    #         for line in f:
    #             if self.nclasses == 2:
    #                 sample = line.strip().split('\t')
    #                 sst_data.append(sample[0].split() + [sample[1]])
    #             elif self.nclasses == 5:
    #                 sample = line.strip().split(' ', 1)
    #                 sst_data.append(sample[1].split() + [sample[0]])
    #     return sst_data
