#!/usr/bin/env python
#coding:utf-8

import core.mtcnn_caffe_to_facenet_data as mtcnn
import core.facenet_recognize as facenet

if __name__ == '__main__':

    save_dir = './mtcnn_67'

    print("===================mtcnn start!!!=======================")
    argv1 = ['--net_12_prototxt','/mllib/ALG/quant_mtcnn/int8_12net.prototxt',
             '--net_12_caffemodel','/mllib/ALG/quant_mtcnn/int8_12net.caffemodel',
             '--net_24_prototxt', '/mllib/ALG/quant_mtcnn/int8_24net.prototxt',
             '--net_24_caffemodel','/mllib/ALG/quant_mtcnn/int8_24net.caffemodel',
             '--net_48_prototxt','/mllib/ALG/quant_mtcnn/int8_48net.prototxt',
             '--net_48_caffemodel','/mllib/ALG/quant_mtcnn/int8_48net.caffemodel',
             '--data_dir','/mllib/ALG/facenet-tensorflow/jz_80val',
             '--save_dir',save_dir]
    print("argv1 =",argv1)
    print("===================parse_arguments=======================")
    args1 = mtcnn.parse_arguments(argv1)
    print("args1 =",args1)
    print("===================main===================================")
    mtcnn.main(args1)
    print("===================mtcnn end!!!===========================")


    print("===================facenet start!!!=======================")
    argv2 = ['--pathways','1', ### pathways 1 compare  database ,pathways 0 create database 
             '--gpu_memory_fraction','0.3',
             '--seed','995',
             '--batch_size', '1',
             '--image_size','67',
             '--data_dir',save_dir,
             '--quant_model','/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/has-JzRequantize/quantized_graph.pb',
             '--best_threshold','0.9',
             '--database','/mllib/ALG/facenet-tensorflow/jz_80val_mtcnn_to_facenet_128_imdb']
    print(argv2)
    print("-----core.facenet_test.parse_arguments--------------")
    args2 = facenet.parse_arguments(argv2)
    print(args2)
    print("-----core.facenet_test.main--------------")
    facenet.main(args2)
    print("===================facenet end!!!=======================")
