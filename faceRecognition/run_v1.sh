#!/bin/bash

echo "===================================="
echo "start!!!"
echo "===================================="

export PYTHONPATH="${HOME}/work/mtcnn_caffe/python"

echo "===================================="
echo "mtcnn start!"
echo "===================================="
/usr/bin/python mtcnn_caffe_to_facenet_data.py
echo "===================================="
echo "mtcnn end!"
echo "===================================="

export PATH="/home/shwu/anaconda2/bin:$PATH"
source activate tensorflow-facenet

echo "===================================="
echo "facenet start!"
echo "===================================="


export CUDA_VISIBLE_DEVICES=""
./test_mobilenet.py

echo "===================================="
echo "facenet end!"
echo "===================================="


source deactivate 

echo "===================================="
echo "end!!!"
echo "===================================="
