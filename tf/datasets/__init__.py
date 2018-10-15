#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
from __future__ import absolute_import
from .classify_eval import *
from .imdb import IMDB
from tf.datasets.qa import AIC
from tf.datasets.srl import SRL

__all__ = ["CR", "MR", "SUBJ", "MPQA", "Kaggle", "SST", "IMDB", "AIC", "SRL"]
