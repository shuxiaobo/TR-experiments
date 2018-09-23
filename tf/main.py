#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created bRy ShaneSue on 2018/8/31
import sys
import os

sys.path.append(os.getcwd() + '/..')
from base import nlp_base
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Note: if set "0,1,2,3" and the #1 GPU is using, will cause OOM Error


def get_model_class(model_name):
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            return nlp_base()
        class_obj, class_name = None, sys.argv[1]
    else:
        class_obj, class_name = None, None
    try:
        import tf.model
        class_obj = getattr(sys.modules["tf.model"], class_name if model_name == None else model_name)
        # sys.argv.pop(1)
    except AttributeError or IndexError:
        print("Model [{}] not found.\nSupported models:\n\n\t\t{}\n".format(class_name, sys.modules["models"].__all__))
        exit(1)
    return class_obj()


if __name__ == '__main__':

    model = get_model_class('QA2')
    model.execute()
