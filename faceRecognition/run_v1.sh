#!/bin/bash

rm config.py -rf
echo "from easydict import EasyDict as edict
config = edict()
config.SAVE_DIR = '/data/shwu/task/commit/facenet/faceRecognition/tes_67'" >> config.py

echo "===================================="
echo "start!!!"
echo "===================================="

export PYTHONPATH="${HOME}/work/mtcnn_caffe/python"

echo "===================================="
echo "mtcnn start!"
echo "===================================="
/usr/bin/python test_mtcnn.py
echo "===================================="
echo "mtcnn end!"
echo "===================================="

export PATH="${HOME}/anaconda2/bin:$PATH"
source activate tensorflow-facenet

echo "===================================="
echo "facenet start!"
echo "===================================="


export CUDA_VISIBLE_DEVICES=""
./test_mobilenet.py # > log1.log 2>&1

echo "===================================="
echo "facenet end!"
echo "===================================="


source deactivate 

echo "===================================="
echo "end!!!"
echo "===================================="
