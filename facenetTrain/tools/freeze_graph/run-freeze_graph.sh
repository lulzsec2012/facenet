
echo "================================"
echo "========= freeze_graph ========="
echo "================================"

CHECKPOINT_FILE_GRAPH=/home/ydwu/project/facenet/facenetTrain/facenet_eval_graph.pbtxt

CHECKPOINT=/tmp/ydwu-facenet/creat-training-graph/training-models/model-20180508-193523.ckpt-177839

RESULT_FILE=/home/ydwu/project/facenet/facenetTrain/tools/freeze_graph/frozen_eval_graph.pb
OUTPUT=Bottleneck/act_quant/FakeQuantWithMinMaxVars

CUDA_VISIBLE_DEVICES="" \
python freeze_graph.py \
  --input_graph=${CHECKPOINT_FILE_GRAPH} \
  --input_checkpoint=${CHECKPOINT} \
  --output_graph=${RESULT_FILE} \
  --output_node_names=${OUTPUT}

echo "================================"
echo "========= OK ========="
echo "================================"

