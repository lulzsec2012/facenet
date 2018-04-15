import numpy as np
import tensorflow as tf
import faceRecognize
import sys
import os

class FcnRecognize(object):
    def __init__(self, model_path, data_size=160, batch_size=4):
        graph=tf.Graph()
        with graph.as_default():
            print('fbtian_fcn_recognize Model directory: %s' % model_path)
            meta_file, ckpt_file = faceRecognize.get_model_filenames(os.path.expanduser(model_path))
            print('Metagraph file: %s,Checkpoint file: %s' % (meta_file,ckpt_file) )
            self.sess=tf.Session(config=tf.ConfigProto( allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            
            model_dir_exp = os.path.expanduser(model_path)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))
            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")#("input:0")
            self.embeddings =  tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        self.data_size = data_size
        self.batch_size = batch_size


    def recognize(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur+batch_size, n), :, :, :])
            cur += batch_size
        emb_list=[]
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            emb = self.sess.run(self.embeddings, feed_dict={ self.images_placeholder:data, self.phase_train_placeholder:False})
            emb_list.append(emb[:real_size])

        return np.concatenate(emb_list,axis=0)
