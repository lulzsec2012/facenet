


# echo "========================================================================================="
# echo "=============== Quantize to quantized_graph.pb ==============="
# echo "========================================================================================="

FACENET_ROOT=$(pwd)


INPUT_MODEL=${FACENET_ROOT}/../freeze_graph/frozen_eval_graph.pb
OUTPUT_MODEL=${FACENET_ROOT}/quantized_graph.pb

# INPUT_MODEL=/home/ydwu/project/facenet/facenetTrain/tools/freeze_graph/frozen_eval_graph.pb
# OUTPUT_MODEL=/home/ydwu/project/facenet/facenetTrain/tools/transforms_graph/quantized_graph.pb


/home/ydwu/framework/tensorflow1.8/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=${INPUT_MODEL} \
  --out_graph=${OUTPUT_MODEL} \
  --inputs=Placeholder \
  --outputs=Bottleneck/act_quant/FakeQuantWithMinMaxVars \
  --transforms='
  add_default_attributes
  strip_unused_nodes(type=float, shape="1,67,67,3")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms
  squeeze_to_reshape
  quantize_weights(minimum_size=20)
  quantize_nodes
  strip_unused_nodes
  replace_conv_bias
  sort_by_execution_order
'


# rename_op(old_op_name="Requantize", new_op_name="RequantizeEight")
# rename_op(old_op_name="Requantize", new_op_name="JzRequantize")
# calculate_mn

# quantize_nodes(input_max=4.07, input_min=-3.25)


# echo "========================================================================================="
# echo "=============== OK! ==============="
# echo "========================================================================================="
