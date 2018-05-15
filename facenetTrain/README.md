
# Quantized float_facenet_model, and creating a quant_facenet_model which use calculation with int.

## 1) creat_training_graph
Running 'CUDA_VISIBLE_DEVICES="1" python ./train_mobilenet.py'.
the file named 'train_mobilenet.py' is a script for training model.
you can creat training_graph with fake-quant.

## 2) creat_eval_graph
Running 'CUDA_VISIBLE_DEVICES="" python ./eval_mobilenet.py'.
the file named 'eval_mobilenet.py' is a script for generating eval_graph. 
you can get a file named 'facenet_eval_graph.pbtxt'.

## 3) freeze_graph
Entering 'tools/freeze_graph', and Running './run-freeze_graph.sh'.
you can get a file named 'frozen_eval_graph.pb', and means that transform from ckpt-file to pb-file.

## 4) graph_transforms
Entering 'tools/transforms_graph', and Running './run-transform_quant.sh'.
you can get a file named 'quantized_graph.pb' which use calculation with int.

## 5)test-final_model
Running 'CUDA_VISIBLE_DEVICES="" python ./test_mobilenet.py'.


