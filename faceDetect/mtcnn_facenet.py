#!/usr/bin/env python
#coding:utf-8
import numpy as np
import argparse
import sys
import os
import cv2
import time
from scipy import misc
import core.faceRecognize as faceRecognize  
os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
from core.model import P_Net, R_Net, O_Net

from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.MtcnnDetector import MtcnnDetector
from core.fcn_recognize import FcnRecognize


from scipy import interpolate

def dict2score(dist_target):
    dist=   [0,   0.076,   0.1 ,  0.2 ,  0.3 , 0.45 , 0.55 , 0.59 , 0.69, 0.73, 0.76, 0.82, 0.85, 0.88 , 0.90, 0.92,  1 , 1.07,1.17 ,1.20, 1.25, 1.30 , 1.35 ,1.5 ,  1.6,1.7,1.8,1.9, 2.5 , 3.0,3.5,4.0 ,4.5]
    score=[100,    100,     100,  100 , 99.9 , 99   , 98   ,  96  ,  94 ,  90 ,   88 ,  85,   80,  75  ,  70 , 65  ,  60, 55  , 50  , 20 ,  2  ,  0.01,  0   ,  0 ,    0,  0, 0 , 0 ,   0 ,   0,  0,  0,  0 ]
    f = interpolate.interp1d(dist, score, kind='slinear')
    score_target= f(dist_target)  
    return score_target


def test_net( dataset_path, prefix, faceRecognize_model,epoch,
             batch_size, test_mode="onet",
             thresh=[0.6, 0.6, 0.7], min_face_size=24,margin=44,
             stride=2, slide_window=False):

    detectors = [None, None, None]

    model_path=['%s-%s'%(x,y) for x,y in zip(prefix,epoch)]
    if slide_window: 
        PNet = Detector(P_Net, 12, batch_size[0],model_path[0]) 
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    #load rnet model
    if test_mode in ["rnet", "onet"]:
        detectors[1]  = Detector(R_Net, 24, batch_size[1], model_path[1]) 
    # load onet model
    if test_mode == "onet":
        detectors[2] = Detector(O_Net, 48, batch_size[2], model_path[2]) 

    # load faceRecognize model
    face_rec=FcnRecognize(faceRecognize_model,data_size=67, batch_size=16)
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    ######load data
    input_dir0='/mllib/ALG/facenet-tensorflow/jz_80val' 
    classes1 = os.listdir(input_dir0)
    message_path=[]
    for cls1 in classes1 :
      classes2_path = os.path.join(input_dir0, cls1 )
      try :
          classes2 = os.listdir(classes2_path)
      except Exception as e:
          print e
          continue
      key=cv2.waitKey(1)
      if key=='q' :  ### when keycode is q 
          print('====------======\n')
      img_list_tmp = []
      num_id=0
      image_message = {}
      message=[]
      meg_name=['exit_id','img_read','img_detect','face_num','face_roi','exit_person','score','person_name']
      ##########0 exit_id :Is there an id   ; 1 ,0
      ##########1 img_read :Whether to read successfully image  ; 1 ,0
      ##########2 img_detect :Whether to detect successfully image ; 1 ,0
      ##########3 face_num :The face amount  ; 0, 1, 2, 3, ...
      ##########4 face_roi :The face of a coordinate  ; (x1,y1,w1,h1),(x2,y2,w2,h2) ...
      ##########5 exit_person :The person amount ; 0, 1, 2, 3, ...
      ##########6 score :The person score ; 0, 1, 2, 3, ...
      ##########7 score :The person name ; name0, name1, name2, name3, ...
      
      for cls2 in classes2 :
        classes3_path = os.path.join(classes2_path, cls2 )
        print(classes3_path)
        if 1: #classes2_path == "emb_check" : 
            image_message[meg_name[0]]=1
        else :
            num_id=0
            image_message[meg_name[0]]=0
            continue
        try :
            img = cv2.imread(classes3_path)
        except Exception as e:
            print e
            image_message[meg_name[1]]=0
            continue
        #img = cv2.imread(image_name)#('test01.jpg')
        #img = cv2.imread(input_test+'.jpg')
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        image_message[meg_name[0]]=1 
        t1 = time.time()
        try :
            boxes, boxes_c = mtcnn_detector.detect_pnet(img)
        except Exception as e:
            image_message[meg_name[2]]=0
            print e
            print(classes3_path )
            continue
        image_message[meg_name[2]]=1  
        #message.append("img_available")
        if  boxes_c is  None:
            image_message[meg_name[3]]=0
            continue
        boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
        if  boxes_c is  None:
            image_message[meg_name[3]]=0
            continue
        boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)
        if  boxes_c is  None:
            image_message[meg_name[3]]=0
            continue
        image_message[meg_name[3]]=len(boxes_c)
        print('box_num:%d',len(boxes_c))
        print 'time: ',time.time() - t1
        message.append("have_face")
        num_box=[]
        if boxes_c is not None:
            img0= misc.imread(classes3_path) 
            img_size = np.asarray(img0.shape)[0:2]

            nrof_samples = len(boxes_c)
  
            for det in boxes_c:
                bb = np.zeros(4, dtype=np.int32)
                margin_tmp=( (det[3]-det[1])-(det[2]-det[0]))/2
                if margin_tmp>0 :
                    size_tmp=(det[3]-det[1])*0
                    bb[0] = np.maximum(det[0]-margin_tmp-size_tmp, 0)
                    bb[1] = np.maximum(det[1]-size_tmp, 0)
                    bb[2] = np.minimum(det[2]+margin_tmp+size_tmp, img_size[1])
                    bb[3] = np.minimum(det[3]+size_tmp, img_size[0])
                else :
                    size_tmp=(det[2]-det[0])*0
                    bb[0] = np.maximum(det[0]-size_tmp, 0)
                    bb[1] = np.maximum(det[1]+margin_tmp-size_tmp, 0)
                    bb[2] = np.minimum(det[2]+size_tmp, img_size[1])
                    bb[3] = np.minimum(det[3]-margin_tmp+size_tmp, img_size[0])
                cropped = img0[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2]),:]
                num_box.append("%d,%d,%d,%d"%(int(bb[1]),int(bb[3]),int(bb[0]),int(bb[2]) ))
                #misc.imsave('hebei/%d.png'%i, cropped)
                #cropped=misc.imread('/home/ssd/fb_data/casic_cluster_china67/Jin Jong-oh/Jin Jong-oh_8.png')
                #cv2.imshow("cropped", cropped)
                #cv2.waitKey(500) ###
                aligned = misc.imresize(cropped, (67, 67), interp='bilinear')
                #cv2.imshow("aligned", aligned)
                #cv2.waitKey(500) ###
                #misc.imsave('hebei/%d.png'%i, aligned)
                prewhitened = faceRecognize.prewhiten(aligned)
                img_list_tmp.append( prewhitened )
                key=cv2.waitKey(1)
        image_message[meg_name[4]]=num_box
        print(image_message)

      if len(img_list_tmp) <2 : #1 :
          continue
      img_list = [None] * len(img_list_tmp)
      for i in range(len(img_list_tmp)) :
          img_list[i] =img_list_tmp[i]
      images = np.stack(img_list)
      t_emb = time.time()
      emb=face_rec.recognize(images)
      
      np.save("./emb_check.txt",emb)
      print 'emb_1_time: ',time.time() - t_emb
    
      for i in range(len(img_list_tmp)):
          print('%1d ' % i)
          score_tmp=65
          score2person=-1
          for j in range(len(img_list_tmp)):
              dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
              print('  %1.4f  ' % dict2score(dist) )
              print('') 
              if dict2score(dist)> score_tmp:
                  score_tmp=dict2score(dist)
                  score2person=j
        
          if score_tmp>65 :
              image_message[meg_name[6]]=score_tmp
              image_message[meg_name[7]]='num_0'

          else:
              image_message[meg_name[6]]=0
              image_message[meg_name[7]]='num_0'
      print('-----====end compare !!===-----') 
      print(image_message)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='onet', type=str)
    parser.add_argument('--detect_model', dest='detect_model', help='detect_model of model name', nargs="+",
                        default=['/mllib/ALG/facenet-tensorflow/wider_model/pnet', '/mllib/ALG/facenet-tensorflow/wider_model/rnet', '/mllib/ALG/facenet-tensorflow/wider_model/onet'], type=str)

    parser.add_argument('--faceRecognize_model', type=str, 
                        help='Directory containing the meta_file and ckpt_file',
                        default='/mllib/ALG/facenet-tensorflow/mobilenet_model_bj/20170829-172716')
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[1024, 128, 8], type=int)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=0)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+", default=[0.7, 0.8, 0.99], type=float)
                       
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=48, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    test_net(args.dataset_path, args.detect_model, args.faceRecognize_model,
             args.epoch, args.batch_size, args.test_mode,
             args.thresh, args.min_face, args.stride,
             args.slide_window)

#export CUDA_VISIBLE_DEVICES=1  ./mtcnn_facenet.py  
