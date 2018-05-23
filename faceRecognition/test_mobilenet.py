#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

import core.facenet_recognize as facenet
from config import config

if __name__ == "__main__":
    argv = ['--pathways','1',
            '--gpu_memory_fraction','0.3',
            '--seed','995',
            '--batch_size', '1',
            '--image_size','67',
            '--data_dir',config.SAVE_DIR,
            '--quant_model','/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/has-JzRequantize/quantized_graph.pb',
            '--database','/data/shwu/task/commit/facenet/faceRecognition/shwu_imdb',
            '--best_threshold','0.9']
    print(argv)
    print("-----core.facenet_test.parse_arguments--------------")
    args = facenet.parse_arguments(argv)
    print(args)
    print("-----core.facenet_test.main--------------")
    facenet.main(args)

#export CUDA_VISIBLE_DEVICES="" ./test_mobilenet.py
