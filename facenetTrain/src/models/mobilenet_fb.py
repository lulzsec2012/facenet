from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):

  logits, end_points = mobilenet(images, is_training=phase_train, weight_decay=weight_decay, reuse=None)
  #return end_points['squeeze'],logits #end_points['squeeze']
  return logits, end_points
def mobilenet(inputs,
          num_classes=1000,
          is_training=True,
          width_multiplier=1,
          reuse=None,
          weight_decay=0.0,
          scope='MobileNet'):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  normalizer_fn=slim.batch_norm,
                                                  scope=sc+'/depthwise_conv')

    bn = depthwise_conv #slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        normalizer_fn=slim.batch_norm,
                                        scope=sc+'/pointwise_conv')
    bn = pointwise_conv #slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn


  print(is_training)
  if is_training == True:  
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.995,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
      # Only update statistics during training mode
      'is_training': True,
    }
    print("is_training ========================================== True")
  else:
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.995,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # Only update statistics during training mode
      'is_training': False,
    }
    print("is_training ========================================== False")

  end_points = {} 
  with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=tf.nn.relu):#, 
                        #outputs_collections=[end_points_collection]):
    with tf.variable_scope(scope,[inputs], reuse=reuse):
    #end_points_collection = sc.name + '_end_points'

      with slim.arg_scope([slim.batch_norm],
                          **batch_norm_params): #,
                          # is_training=is_training,
                          # activation_fn=None):

        
        print(inputs) #batch_size*67*67*3 or  65~80
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=1,padding='SAME',scope='conv_1')# padding='SAME', padding='VALID',
        print(net)
        #net = slim.batch_norm(net, scope='conv_1/batch_norm')
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
        print(net)
        net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
        print(net)
        net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
        net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5') #downsample=True,
        print(net)
        net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_7')
        net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_8')
        print(net)
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
        net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True,  sc='conv_ds_13')
        print(net)
        end_points['avg_pool_13'] = net
        net = _depthwise_separable_conv(net, 1024, width_multiplier,sc='conv_ds_14')
        net = slim.avg_pool2d(net, [5, 5], scope='avg_pool_15')
        print(net)
        end_points['avg_pool_15'] = net
    logits = tf.squeeze(net, [1, 2], name='logits')
    end_points['logits'] = logits

  return logits, end_points

mobilenet.default_image_size = 224


def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc
