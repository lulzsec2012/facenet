#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


import src.facenet_train as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
if __name__ == "__main__":
    argv = ['--logs_base_dir', './models_test/resnet_67v1c4',
            '--models_base_dir', './models_test/resnet_67v1c4', 
            '--data_dir','/mllib/ALG/facenet-tensorflow/casia_noalgin_misc',
            '--pretrained_model','/mllib/ALG/facenet-tensorflow/resnet_67v1c4_model_bj/20171014-161329/model-20171014-161329.ckpt-299621',
            '--model_def', 'src.models.inception_resnet_v1_change4',
            '--gpu_memory_fraction','0.4',
            '--alpha','0.8',  
            '--seed','1896',
            '--epoch_size', '1000',
            '--max_nrof_epochs', '1000',
            '--batch_size', '126',
            '--people_per_batch','2700',
            '--images_per_person', '7',
            '--image_size','67',
            '--learning_rate','0.04',
            '--random_flip',
            '--lfw_pairs','/mllib/ALG/facenet-tensorflow/pairs.txt', 
            '--lfw_dir', '/mllib/ALG/facenet-tensorflow/lfw_67',
            '--lfw_nrof_folds', '2',
            '--no_store_revision_info' ]
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
#export CUDA_VISIBLE_DEVICES=1  ./train_resnet_67v1c4.py
