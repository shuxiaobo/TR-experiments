#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
# make sure the model supports the dataset you use
models_in_datasets = {
    "CBT_NE": ["AttentionSumReader", "AoAReader", "Simple_model", "Simple_modelrl", "Simple_model1"],
    "CBT_CN": ["AttentionSumReader", "AoAReader", "Simple_model", "Simple_modelrl", "Simple_model1"],
    "SQuAD": ["RNet", "SimpleModelSQuad", "BiDAF", "SimpleModelSQuad3", "SimpleModelSQuad4", "SimpleModelSQuadBiDAF", "SimpleModelSQuad5"]
}
