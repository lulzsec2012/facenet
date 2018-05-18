#!/bin/bash

FACENET_ROOT=$(pwd)


echo "===================================="
echo "start!"
echo "===================================="
echo "FACENET_ROOT = " ${FACENET_ROOT}

# # 1) creat_training_graph
# echo "===================================="
# echo "creat_training_graph!"
# echo "===================================="

# CUDA_VISIBLE_DEVICES="" python ./train_mobilenet.py


# # 2) creat_eval_graph
echo "===================================="
echo "creat_eval_graph!"
echo "===================================="

CUDA_VISIBLE_DEVICES="" python ./eval_mobilenet.py

# # 3) freeze_graph
echo "===================================="
echo "freeze_graph!"
echo "===================================="

cd ${FACENET_ROOT}/tools/freeze_graph
./run-freeze_graph.sh

# # 4) graph_transforms
echo "===================================="
echo "graph_transforms!"
echo "===================================="

cd ${FACENET_ROOT}/tools/transforms_graph
./run-transform_quant.sh

# # 5) test-final_model
echo "===================================="
echo "test-final_model!"
echo "===================================="

cd ${FACENET_ROOT}
CUDA_VISIBLE_DEVICES="" python ./test_mobilenet.py

echo "===================================="
echo "end!"
echo "===================================="
