"""Test a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
import os.path
import time
import sys
import cv2
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
from tensorflow.python.framework import importer
from tensorflow.python.ops import data_flow_ops
from scipy import interpolate

def dict2score(dist_target):
    dist=   [0,   0.076,   0.1 ,  0.2 ,  0.3 , 0.45 , 0.55 , 0.59 , 0.69, 0.73, 0.76, 0.82, 0.85, 0.88 , 0.90, 0.92,  1 , 1.07,1.17 ,1.20, 1.25, 1.30 , 1.35 ,1.5 ,  1.6,1.7,1.8,1.9, 2.5 , 3.0,3.5,4.0 ,4.5]
    score=[100,    100,     100,  100 , 99.9 , 99   , 98   ,  96  ,  94 ,  90 ,   88 ,  85,   80,  75  ,  70 , 65  ,  60, 55  , 50  , 20 ,  2  ,  0.01,  0   ,  0 ,    0,  0, 0 , 0 ,   0 ,   0,  0,  0,  0 ]
    f = interpolate.interp1d(dist, score, kind='slinear')
    score_target= f(dist_target)
    return score_target


def getMinIndex(my_list):
    min_index = []
    flag = min(my_list)
    for i in enumerate(my_list):
        if i[1]==flag:
           min_index.append(i[0])
    return min_index

def main(args):
  
    lfw_paths = []
    input_dir0 = args.data_dir
    classes1 = os.listdir(input_dir0)
    for cls1 in classes1 :
        path = os.path.join(input_dir0, cls1)
        for im_name in os.listdir(path):
            imgpath = os.path.join(path, im_name)
            lfw_paths.append(imgpath)
            try :
                img = cv2.imread(imgpath)
                caffe_img = img.copy()
            except Exception as e:
                print("e =",e)
                continue      
    g = tf.Graph()
    with g.as_default(), tf.device(tf.train.replica_device_setter(0)):
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
    
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(1,), (1,)],
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
                image = tf.image.decode_png(file_contents)
                fb_count+=1
                #pylint: disable=no-member# fbtian_add
                image.set_shape((args.image_size, args.image_size, 3))
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
            print('===============Running forward pass on data_dir images==============')
            nrof_images = len(lfw_paths)
            assert(len(image_paths)==nrof_images)
            labels_array = np.reshape(np.arange(nrof_images),(-1,1))
            image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,1))
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
            emb_array = np.zeros((nrof_images, embedding_size))
            nrof_batches = int(np.ceil(nrof_images / batch_size))
            label_check_array = np.zeros((nrof_images,))
            if args.pathways == 1:
                print('===============Compare emb database start !!====================')
            elif args.pathways == 0:
                print('===============Collection emb database start !!====================')
            else:
                pass

            data = dict()
            emb_all = []
            emb_name = []
            emb_name = lfw_paths
            for i in xrange(nrof_batches):
                name_path = lfw_paths[i]
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
                        emb_all.append(emb)
                        if args.pathways == 1:
                            print ('===========One test start==============')
                            input_dir1 = args.database
                            save_txt = input_dir1 + "/temp.imdb"
                            fid = open(save_txt,'r')
                            data = pickle.load(fid)
                            fid.close()
                            dist_list = []
                            for emb1 in data['emb_all']:
                                 dist = np.sqrt(np.sum(np.square(np.subtract(emb[:], emb1[:]))))
                                 dist_list.append(dist)
                            min_dist = min(dist_list)
                            min_index = []
                            min_index = getMinIndex(dist_list)
                            for j in min_index:
                                imgpath = data['emb_name'][j]
                                file_name_ = data['emb_name'][j].split('/')
                                file_name_ = file_name_[-1].rstrip('.jpg')
                                
                                test_name_ = name_path.split('/')
                                test_name_ = test_name_[-1].rstrip('.jpg')
                                flag = 0
                                if min_dist < 0.8955:
                                    flag += 1
                                    print('The test %s is like %s on dist: %1.4f  ' %(test_name_,file_name_,min_dist))
                                    print('The test %s is like %s on score: %1.4f  ' %(test_name_,file_name_,dict2score(min_dist)))
                                    test_img = cv2.imread(name_path)
                                    database_img = cv2.imread(imgpath)
                                    cv2.imshow(file_name_, database_img)
                                    cv2.imshow("test", test_img)
                                    cv2.waitKey(0)
                            if flag == 0:
                                print("The test one isn't exist database")
                                
                            print ('===========One test end================')
                        else:
                            pass
            if args.pathways == 0:
                save_dir = input_dir0 + "_txt/"
                if not os.path.isdir(save_dir):  
                    os.makedirs(save_dir)
                data['emb_all'] = emb_all
                data['emb_name'] = emb_name
                save_dir_txt = save_dir+'temp.imdb'
                f = open(save_dir_txt,'w')
                pickle.dump(data, f)
                f.close()
                print ('===========emb  collection end================')
            else:
                pass
        if args.pathways == 1:
            print('===============Compare emb database end !!======================')
        elif args.pathways == 0:
            print('===============Collection emb database end !!======================')
        else:
            pass
    sess.close()
    sess_ydwu.close()

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathways', type=int,
                        help='if 1 run compare database else 0 run create database', default=1)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.3)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/data/shwu/facenet_beijing/lzlu_faceDetectRec/test_per_data_67')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=1)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=67)
    parser.add_argument('--embedding_size', type=float,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--database', type=str,
        help='Compare database.', default='./mtcnn_to_facenet_data_67_txt')
    parser.add_argument('--quant_model', type=str,
                        help='quantized model.', default='/mllib/ALG/facenet-tensorflow-quant/based-beijing/graph_transforms/has-JzRequantize/quantized_graph.pb')

    return parser.parse_args(argv)
  
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
