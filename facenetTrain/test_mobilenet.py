#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

# CUDA_VISIBLE_DEVICES="" python ./test_mobilenet.py

import src.facenet_quant_test as facenet_test
import tensorflow as tf
import tensorflow.contrib.slim as slim

# QUANTIZED_MODEL='/home/ydwu/project/facenet/facenetTrain/tools/transforms_graph/quantized_graph.pb'
# QUANTIZED_MODEL='/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/has-JzRequantize/quantized_graph.pb'
# QUANTIZED_MODEL='/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/has-RequantizeEight/quantized_graph.pb'
# QUANTIZED_MODEL='/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/no-mn/quantized_graph.pb'

if __name__ == "__main__":
    argv = ['--quant_model',QUANTIZED_MODEL,
            '--gpu_memory_fraction','0.3',
            '--seed','995',
            '--batch_size', '135',
            '--image_size','67',
            '--random_flip',
            '--lfw_pairs','/data/shwu/facenet_beijing/lzlu_facenettrain/data_frombj/pairs.txt',
            '--lfw_dir','/data/shwu/facenet_beijing/lzlu_facenettrain/data_frombj/lfw_67',
            '--lfw_nrof_folds', '2' ]

    print(argv)
    print("-----src.facenet_test.parse_arguments--------------")
    args = facenet_test.parse_arguments(argv)
    print("-----src.facenet_test.main--------------")
    facenet_test.main(args)

