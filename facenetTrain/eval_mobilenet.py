#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES="" python ./eval_mobilenet.py

import src.facenet_eval as facenet_eval
import tensorflow as tf
import tensorflow.contrib.slim as slim

EVAL_GRAPH='/tmp/lzlu-facenet/creat-eval-graph/facenet_eval_graph.pbtxt'

if __name__ == "__main__":
    argv = ['--model_def', 'src.models.mobilenet_fb',
            '--eval_graph_dir', EVAL_GRAPH,
            '--no_store_revision_info' ]

    print(argv)
    print("-----src.facenet_eval.parse_arguments--------------")
    args = facenet_eval.parse_arguments(argv)
    print("-----src.facenet_eval.main--------------")
    facenet_eval.main(args)
