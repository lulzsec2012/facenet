#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

import core.facenet_recognize as facenet


if __name__ == "__main__":
    argv = ['--pathways','1',
            '--gpu_memory_fraction','0.3',
            '--seed','995',
            '--batch_size', '1',
            '--image_size','67',
            '--data_dir','/data/shwu/task/facenet/faceRecognition/jz_80val_67',
            '--quant_model','/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/has-JzRequantize/quantized_graph.pb',
            '--database','/data/shwu/task/facenet/faceRecognition/jz_80val_67_txt']
    print(argv)
    print("-----core.facenet_test.parse_arguments--------------")
    args = facenet.parse_arguments(argv)
    print(args)
    print("-----core.facenet_test.main--------------")
    facenet.main(args)

#export CUDA_VISIBLE_DEVICES="" ./test_mobilenet.py
