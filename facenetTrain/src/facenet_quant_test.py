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
  
    network = importlib.import_module(args.model_def, 'inference')

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    subdirmaxlin= subdir+'_lin_max' #fbtian_max
    maxlin_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdirmaxlin)#fbtian_max
    if not os.path.exists(maxlin_dir):#fbtian_max
        os.makedirs(maxlin_dir)#fbtian_max

    subdirmax= subdir+'_max' #fbtian_max
    modelmax_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdirmax)#fbtian_max
    if not os.path.exists(modelmax_dir):#fbtian_max
        os.makedirs(modelmax_dir)#fbtian_max

    # Store some git revision info in a text file in the log directory
    if not args.no_store_revision_info:
        src_path,_ = os.path.split(os.path.realpath(__file__))
        facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    
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
        
        # nrof_preprocess_threads = 4
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
                    print('args.random_crop') #fbtian_add
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    #print('else not args.random_crop') #come in fbtian_add
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    print('args.random_flip')
                    image = tf.image.random_flip_left_right(image)
                if 1 : # 1
                    image = tf.image.random_brightness(image, max_delta=0.2)  #Random brightness transformation
                    image = tf.image.random_contrast(image, lower=0.2, upper=1.0)#Random contrast transformation
                fb_count+=1
                #pylint: disable=no-member# fbtian_add
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
        image_batch = tf.identity(image_batch, 'input') ##fbtian
                
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)###
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
        config = tf.ConfigProto(allow_soft_placement=True) ###########################################
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)###################################
        config.gpu_options.allow_growth = True###########################################
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=8))    
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=8))

        # Initialize variables
        sess.run(tf.global_variables_initializer()) # , feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer()) # , feed_dict={phase_train_placeholder:True})
        tf.summary.FileWriter("/tmp/tf-ydwu/create-g", g)

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
                ydwu_pre_in, lab = sess.run([image_batch, labels_batch], feed_dict={batch_size_placeholder: batch_size})

                g2 = tf.Graph()
                with g2.as_default():

                    ydwu_INPUT_MODEL = args.quant_model
                    ydwu_graph_def = tf.GraphDef()
                    with tf.gfile.FastGFile(ydwu_INPUT_MODEL,'rb') as f:        
                        ydwu_graph_def.ParseFromString(f.read())
                        _ = importer.import_graph_def(ydwu_graph_def, name="")
                        
                    sess_ydwu = tf.Session(graph=g2, config=tf.ConfigProto(allow_soft_placement=True))
                    sess_ydwu.run(tf.global_variables_initializer())
                    sess_ydwu.run(tf.local_variables_initializer())

                    with sess_ydwu.as_default():                
                        ydwu_in = ydwu_pre_in

                        float_out = sess_ydwu.graph.get_tensor_by_name('Bottleneck/act_quant/FakeQuantWithMinMaxVars:0')
                        float_embeddings = sess_ydwu.run(float_out, feed_dict={tf.get_default_graph().get_operation_by_name('Placeholder').outputs[0]: ydwu_in})
                        
                        regularization = tf.nn.l2_normalize(float_embeddings, 1, 1e-10, name='l2_embeddings')
                        embeddings = sess_ydwu.graph.get_tensor_by_name('l2_embeddings:0')
                        emb = sess_ydwu.run(embeddings, feed_dict={tf.get_default_graph().get_operation_by_name('l2_embeddings').inputs[0]:float_embeddings})

                        emb_array[lab,:] = emb
                        label_check_array[lab] = 1            
            
            print('time:%.3f' % (time.time()-start_time))

            print("ydwu ============ emb_array = ", emb_array)
            assert(np.all(label_check_array==1))
            print('embeddings1.shape[0]:%d,embeddings2:%d'%(emb_array[0::2].shape[0],emb_array[1::2].shape[0] ))
            _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=args.lfw_nrof_folds)
            
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            lfw_time = time.time() - start_time
            # Add validation loss and accuracy to summary
            summary = tf.Summary()
            #pylint: disable=maybe-no-member
            summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
            summary.value.add(tag='lfw/val_rate', simple_value=val)
            summary.value.add(tag='time/lfw', simple_value=lfw_time)
            
            print("acc = ", np.mean(accuracy))
            print("val = ", val)
            
            
    sess.close()
    sess_ydwu.close()
    return model_dir

  
      

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.3)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.', default='src.models.inception_resnet_v1')#'models.nn4')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=float,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0) ##1.0
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)

    parser.add_argument('--no_store_revision_info', 
        help='Disables storing of git revision info in revision_info.txt.', action='store_true')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='~/fbtian/soft/facenet/data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='~/fbtian/soft/facenet/dataset/lfw_96')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=2)

    parser.add_argument('--quant_model', type=str,
                        help='quantized model.', default='/home/ydwu/project/facenet/facenetTrain/tools/transforms_graph/quantized_graph.pb')

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
