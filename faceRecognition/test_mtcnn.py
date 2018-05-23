#!/usr/bin/env python
#coding:utf-8
import core.mtcnn_caffe_to_facenet_data as mtcnn
from config import config

if __name__ == '__main__':
    argv = ['--net_12_prototxt','/mllib/ALG/quant_mtcnn/int8_12net.prototxt',
            '--net_12_caffemodel','/mllib/ALG/quant_mtcnn/int8_12net.caffemodel',
            '--net_24_prototxt', '/mllib/ALG/quant_mtcnn/int8_24net.prototxt',
            '--net_24_caffemodel','/mllib/ALG/quant_mtcnn/int8_24net.caffemodel',
            '--net_48_prototxt','/mllib/ALG/quant_mtcnn/int8_48net.prototxt',
            '--net_48_caffemodel','/mllib/ALG/quant_mtcnn/int8_48net.caffemodel',
            '--data_dir','/data/shwu/task/commit/facenet/faceRecognition/test_per_data',
            '--save_dir',config.SAVE_DIR]
    print("argv =",argv)
    print("===================parse_arguments=======================")
    args = mtcnn.parse_arguments(argv)
    print("args =",args)
    print("===================main===================================")
    mtcnn.main(args)

