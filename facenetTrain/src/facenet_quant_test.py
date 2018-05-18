"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from tensorflow.python.framework import importer
from tensorflow.python.framework import graph_util


from tensorflow.python.ops import data_flow_ops

def main(args):

    np.random.seed(seed=args.seed)    
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    g = tf.Graph()
    with g.as_default(), tf.device(tf.train.replica_device_setter(0)):

        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
    
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        
        nrof_preprocess_threads = 1
        images_and_labels = []
    
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            fb_count=0
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                print('filename:%s'%filename )
                image = tf.image.decode_png(file_contents)
                
                if args.random_crop:
                    print('args.random_crop') 
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    print('args.random_flip')
                    image = tf.image.random_flip_left_right(image)
                if 1 :
                    #Random brightness transformation
                    image = tf.image.random_brightness(image, max_delta=0.2)
                    #Random contrast transformation
                    image = tf.image.random_contrast(image, lower=0.2, upper=1.0)
                fb_count+=1
                image.set_shape((args.image_size, args.image_size, 3))

                # # ydwu
                # image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
                # images.append(image)

                # ## origine
                images.append(tf.image.per_image_standardization(image))

            images_and_labels.append([images, label])
            print('fb_count:%d'%fb_count)


        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'input')
                
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=8))

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        batch_size = args.batch_size
        embedding_size = args.embedding_size
        image_paths = lfw_paths
        
        with sess.as_default():
            start_time = time.time()
            # Run forward pass to calculate embeddings
            print('Running forward pass on LFW images: ', end='')

            nrof_images = len(actual_issame)*2
            assert(len(image_paths)==nrof_images)
            print('nrof_images:%d'%nrof_images)
            labels_array = np.reshape(np.arange(nrof_images),(-1,3))
            image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
            emb_array = np.zeros((nrof_images, embedding_size))
            print('nrof_images:%d,batch_size:%d'%(nrof_images,batch_size) )
            nrof_batches = int(np.ceil(nrof_images / batch_size))
            label_check_array = np.zeros((nrof_images,))
            
            for i in xrange(nrof_batches):
                batch_size = min(nrof_images-i*batch_size, batch_size)
                pre_input, lab = sess.run([image_batch, labels_batch], feed_dict={batch_size_placeholder: batch_size})

                g2 = tf.Graph()
                with g2.as_default():

                    quant_graph_def = tf.GraphDef()
                    with tf.gfile.FastGFile(args.quant_model,'rb') as f:        
                        quant_graph_def.ParseFromString(f.read())
                        _ = importer.import_graph_def(quant_graph_def, name="")
                        
                    quant_sess = tf.Session(graph=g2, config=tf.ConfigProto(allow_soft_placement=True))
                    quant_sess.run(tf.global_variables_initializer())
                    quant_sess.run(tf.local_variables_initializer())

                    with quant_sess.as_default():                
                        quant_graph_input = pre_input

                        float_output = quant_sess.graph.get_tensor_by_name('Bottleneck/act_quant/FakeQuantWithMinMaxVars:0')
                        float_embeddings = quant_sess.run(float_output, feed_dict={tf.get_default_graph().get_operation_by_name('Placeholder').outputs[0]: quant_graph_input})
                        
                        regularization = tf.nn.l2_normalize(float_embeddings, 1, 1e-10, name='l2_embeddings')
                        embeddings = quant_sess.graph.get_tensor_by_name('l2_embeddings:0')
                        emb = quant_sess.run(embeddings, feed_dict={tf.get_default_graph().get_operation_by_name('l2_embeddings').inputs[0]:float_embeddings})

                        emb_array[lab,:] = emb
                        label_check_array[lab] = 1            
            
            print('time:%.3f' % (time.time()-start_time))
            assert(np.all(label_check_array==1))
            print('embeddings1.shape[0]:%d,embeddings2:%d'%(emb_array[0::2].shape[0],emb_array[1::2].shape[0] ))
            _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=args.lfw_nrof_folds)
            
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            lfw_time = time.time() - start_time
            # Add validation loss and accuracy to summary
            summary = tf.Summary()

            summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
            summary.value.add(tag='lfw/val_rate', simple_value=val)
            summary.value.add(tag='time/lfw', simple_value=lfw_time)
            
            print("acc = ", np.mean(accuracy))
            print("val = ", val)            
            
    sess.close()
    quant_sess.close()
    return 1

  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # quantized model
    parser.add_argument('--quant_model', type=str,
                        help='quantized model.', default='/mllib/ALG/facenet-tensorflow-quant/graph_transforms/has-JzRequantize/quantized_graph.pb')

    # Some evalution parameters
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.3)    
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--embedding_size', type=float,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='~/fbtian/soft/facenet/data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='~/fbtian/soft/facenet/dataset/lfw_96')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=2)

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
