#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : eval.py
## Authors    : ydwu@taurus
## Create Time: 2018-03-29:16:43:35
## Description:
## 
##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import shutil
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import src.facenet as facenet
import src.lfw as lfw
import tensorflow.contrib.slim as slim

from tensorflow.python.ops import data_flow_ops

import tensorflow as tf

def main(args):
    network = importlib.import_module(args.model_def, 'inference')
    
    g = tf.Graph()
    ydwu_train = False, # True, # False,
    with g.as_default():

        inputs = tf.placeholder(tf.float32, shape=(None, 67, 67, 3))
        
        # Build the inference graph
        prelogits, _ = network.inference(inputs, args.keep_probability, phase_train=ydwu_train, weight_decay=args.weight_decay)

        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # Only update statistics during training mode
            'is_training': False,
        }
        print("is_training ========================================== False")
        

        with slim.arg_scope([slim.batch_norm], is_training=ydwu_train):
            pre_embeddings = slim.fully_connected(prelogits, args.embedding_size, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay), normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='Bottleneck', reuse=False)
        
        embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')
        tf.reshape(embeddings, [-1,3,args.embedding_size])

        print("ydwu == quant")
        tf.contrib.quantize.create_eval_graph(g)
            
        # Save the checkpoint and eval graph proto to disk for freezing
        with open(args.eval_graph_dir, 'w') as f:
            f.write(str(g.as_graph_def()))

    print("finsh!!!!")


        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.', default='src.models.inception_resnet_v1')#'models.nn4')
    parser.add_argument('--embedding_size', type=float,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)    
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--no_store_revision_info', 
        help='Disables storing of git revision info in revision_info.txt.', action='store_true')
    parser.add_argument('--eval_graph_dir', type=str,
        help='The file is pbtxt of eval_graph.', default='/tmp/facenet_eval_graph.pbtxt')

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
