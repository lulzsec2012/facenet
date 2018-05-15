#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

# CUDA_VISIBLE_DEVICES="1" python ./train_mobilenet.py

import src.facenet_train as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
if __name__ == "__main__":
    argv = ['--logs_base_dir','./models_test/mobilenet',
            '--models_base_dir','./models_test/mobilenet',
            '--data_dir','/mllib/ALG/facenet-tensorflow/casia_noalgin_misc',
            '--pretrained_model','/mllib/ALG/facenet-tensorflow/mobilenet_model_bj/20170829-172716/model-20170829-172716.ckpt-194420', 
            '--model_def', 'src.models.mobilenet_fb',
            '--gpu_memory_fraction','0.4',
            '--alpha','0.40', 
            '--seed','995',
            '--epoch_size', '900',
            '--max_nrof_epochs', '500',
            '--batch_size','135',
            '--people_per_batch','2700', 
            '--images_per_person', '9',
            '--image_size','67',
            '--learning_rate','0.005',
            '--random_flip',
            '--lfw_pairs','/mllib/ALG/facenet-tensorflow/pairs.txt',
            '--lfw_dir', '/mllib/ALG/facenet-tensorflow/lfw_67',
            '--lfw_nrof_folds', '2',
            '--no_store_revision_info' ]
    print(argv)
    print("-----src.facenet_train.parse_arguments--------------")
    args = facenet_train.parse_arguments(argv)
    print("-----src.facenet_train.main--------------")
    facenet_train.main(args)

