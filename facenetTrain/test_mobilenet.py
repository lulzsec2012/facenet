#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

# CUDA_VISIBLE_DEVICES="" python ./test_mobilenet.py

import src.facenet_quant_test as facenet_test
import tensorflow as tf
import tensorflow.contrib.slim as slim


if __name__ == "__main__":
    argv = ['--logs_base_dir', '/tmp',
            '--models_base_dir','/tmp',
            '--model_def', 'src.models.mobilenet_fb',
            '--gpu_memory_fraction','0.3',#'0.97',
            '--alpha','0.40', #0.5  alpha/2 now alpha*0.70  #0.95  0.75 
            '--seed','995',
            '--epoch_size','500', ###'900',#1000',
            '--max_nrof_epochs', '1000',#'500',
            '--batch_size', '135',
            '--people_per_batch','10',
            '--images_per_person', '15', ###'9'
            '--image_size','67',
            '--random_flip',
            '--data_dir','/data/shwu/facenet_beijing/lzlu_facenettrain/data_frombj/casia_noalgin_misc',
            '--lfw_pairs','/data/shwu/facenet_beijing/lzlu_facenettrain/data_frombj/pairs.txt',
            '--lfw_dir','/data/shwu/facenet_beijing/lzlu_facenettrain/data_frombj/lfw_67',
            '--lfw_nrof_folds', '2',
            '--quant_model','/home/ydwu/project/facenet/facenetTrain/tools/transforms_graph/quantized_graph.pb',
            '--no_store_revision_info' ]
    print(argv)
    print("-----src.facenet_test.parse_arguments--------------")
    args = facenet_test.parse_arguments(argv)
    print("-----src.facenet_test.main--------------")
    facenet_test.main(args)

