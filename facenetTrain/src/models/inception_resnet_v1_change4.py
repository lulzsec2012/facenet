# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Inception-Renset-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 48, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 48, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net 
  
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net
  
def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, reuse=reuse)

def reduction_fb(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 32, 3,stride=2,padding='VALID', scope='Conv2d_0a_1x1')

    with tf.variable_scope('Branch_1'):
        tower_avg = slim.avg_pool2d(net, 3,stride=2, padding='VALID',scope='AvgPool_0a_3x3')
        with tf.variable_scope('Branch_1_0'):
            tower_conv1_1 = slim.conv2d(tower_avg, 32, 3, scope='Conv2d_1a_3x3')
            tower_conv1_2 = slim.conv2d(tower_avg, 16, (1,3), scope='Conv2d_1a1_1x3')
            tower_conv1_3 = slim.conv2d(tower_avg, 16, (3,1), scope='Conv2d_1a2_3x1')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 64, 3, scope='Conv2d_2a_3x3')
        tower_conv3 = slim.conv2d(tower_conv2, 48, 3, scope='Conv2d_3a_3x3')
        tower_max = slim.max_pool2d(tower_conv3, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')

    net = tf.concat([tower_conv, tower_conv1_1, tower_conv1_2, tower_conv1_3,tower_max], 3)
   
    return net 

def reduction_fb2(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 128, 3,stride=2,padding='VALID', scope='Conv2d_0a_1x1')

    with tf.variable_scope('Branch_1'):
        tower_avg = slim.avg_pool2d(net, 3,stride=2, padding='VALID',scope='AvgPool_0a_3x3')
        with tf.variable_scope('Branch_1_0'):
            tower_conv1_1 = slim.conv2d(tower_avg, 64, 3, scope='Conv2d_1a_3x3')
            tower_conv1_2 = slim.conv2d(tower_avg, 32, (1,3), scope='Conv2d_1a1_1x3')
            tower_conv1_3 = slim.conv2d(tower_avg, 32, (3,1), scope='Conv2d_1a2_3x1')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 32, 3, scope='Conv2d_0a_1x1')
        tower_max = slim.max_pool2d(tower_conv2, 3, stride=2, padding='VALID',
                                      scope='MaxPool_1a_3x3')

    net = tf.concat([tower_conv, tower_conv1_1, tower_conv1_2, tower_conv1_3,tower_max], 3)
    net = slim.conv2d(net, 144, 1, scope='Conv2d_0a_1x1')
    return net 

def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV1'):
    #inputs 299 x 299 x 32
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {} 
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                #input 67 x 67 x 3
                #33 x 33 x 144
                with tf.variable_scope('Mixed_0a'):
                    net = reduction_fb(inputs)
                print(net)
                # 33 x 33 x 96
                net = slim.conv2d(net, 96, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                print(net)
                end_points['Conv2d_3b_1x1'] = net
                # 31 x 31 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                print(net)
                # 15 x 15 x 288   ##15 x 15 x 144
                with tf.variable_scope('Mixed_4a'):
                    net = reduction_fb2(net)
                end_points['Mixed_4a'] = net
                print(net)
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 6, block35, scale=0.7) ##0.17
                print(net)
                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                print(net)
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 11, block17, scale=0.5) ##0.1
                print(net)
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net
                print(net)
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 6, block8, scale=0.7) ##0.20
                print(net)
                net = block8(net, activation_fn=None)
                print(net)
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_3x3')

                    #print(net)
                    #net = slim.conv2d(net, 768, 1, scope='Conv2d_2a_1x1')
                    #net = slim.conv2d(net, 256, 1, scope='Conv2d_3a_1x1')
  
                    print(net)
                    net = slim.flatten(net)
                    print(net)
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
                    print(net)
                    end_points['PreLogitsFlatten'] = net
  
    return net, end_points
