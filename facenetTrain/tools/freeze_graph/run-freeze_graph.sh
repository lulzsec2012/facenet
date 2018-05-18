
# echo "================================"
# echo "========= freeze_graph ========="
# echo "================================"

FACENET_ROOT=$(pwd)

CHECKPOINT_FILE_GRAPH=/mllib/ALG/facenet-tensorflow-quant/creat_eval_graph/facenet_eval_graph.pbtxt

CHECKPOINT=/tmp/lzlu-facenet/creat-training-graph/result/20180516-175949_max/model-20180516-175949.ckpt_valmax

RESULT_FILE=${FACENET_ROOT}/frozen_eval_graph.pb
OUTPUT=Bottleneck/act_quant/FakeQuantWithMinMaxVars

CUDA_VISIBLE_DEVICES="" \
python freeze_graph.py \
  --input_graph=${CHECKPOINT_FILE_GRAPH} \
  --input_checkpoint=${CHECKPOINT} \
  --output_graph=${RESULT_FILE} \
  --output_node_names=${OUTPUT}

# echo "================================"
# echo "========= OK ========="
# echo "================================"

