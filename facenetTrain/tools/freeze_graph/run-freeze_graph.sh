
echo "================================"
echo "========= freeze_graph ========="
echo "================================"

CHECKPOINT_FILE_GRAPH=/tmp/lzlu-facenet/creat-eval-graph/facenet_eval_graph.pbtxt

#CHECKPOINT=/tmp/lzlu-facenet/creat-training-graph/result/20180516-175949/model-20180516-175949.ckpt-9536
CHECKPOINT=/tmp/lzlu-facenet/creat-training-graph/result/20180518-140509_max/model-20180518-140509.ckpt_valmax
CHECKPOINT=/tmp/lzlu-facenet/creat-training-graph/result/20180523-192837_max/model-20180523-192837.ckpt_valmax

RESULT_FILE=/tmp/lzlu-facenet/freeze_graph/frozen_eval_graph.pb
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

