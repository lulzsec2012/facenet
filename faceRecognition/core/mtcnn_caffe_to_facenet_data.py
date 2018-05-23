import sys,pprint
#pprint.pprint(sys.path)
sys.path.append('../')
sys.path.append('./')
sys.path.append('${HOME}/work/mtcnn_caffe/python')

import caffe, os
import cv2
import time
import os.path
import argparse
import numpy as np
import tools_matrix as tools

def detectFace(img_path,threshold,args):
    deploy = args.net_12_prototxt
    caffemodel = args.net_12_caffemodel
    net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
     
    deploy = args.net_24_prototxt
    caffemodel = args.net_24_caffemodel
    net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)
          
    deploy = args.net_48_prototxt
    caffemodel = args.net_48_caffemodel
    net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

    print ("img_path:",img_path)
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/127.5
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    pnet_caffe_img = img.copy()
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(pnet_caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
	caffe.set_device(0)
	caffe.set_mode_cpu()        
	out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        roi = roi / 30.2 / 858.73 #####8bit 30.200505602063224, 192.7469009510095, 858.7324507745616
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)

    rectangles = tools.NMS(rectangles,0.7,'iou')
    print "==================pnet_rectangles_num: ", len(rectangles)

    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    rnet_caffe_img = img.copy()
    for rectangle in rectangles:
        crop_img = rnet_caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']

    roi_prob = roi_prob / 78.99 / 393.3528 ###8bit 78.99526980055604, 347.1242616474836, 393.3528946049652
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
  
    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    onet_caffe_img = img.copy()
    
    for rectangle in rectangles:
        crop_img = onet_caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img        
        crop_number += 1
 
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    roi_prob = roi_prob / 98.75 / 499.455 ###8bit 98.7522345220119, 354.8245526748337, 499.45565604371944        
    rectangles = tools.filter_face_48net_nopts(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[2])        
    return rectangles

def main(args):

    threshold = args.threshold
    sum,t = 0,0
    origin_start = time.time()
    input_dir0 = args.data_dir
    classes1 = os.listdir(input_dir0)
    for cls1 in classes1 :
        path1 = os.path.join(input_dir0, cls1)
        if not(os.path.isdir(path1)):
            continue
        path = os.path.join(input_dir0, cls1)
        facenet_save_dir = os.path.join(args.save_dir, cls1)     
        if not os.path.isdir(facenet_save_dir):  
            os.makedirs(facenet_save_dir)                    
        faceimg_size = 67                                 
        d_idx = 0                                        
        for im_name in os.listdir(path):
            sum += 1
            imgpath = os.path.join(path, im_name)
            try :
                img = cv2.imread(imgpath)
                caffe_img = (img.copy()-127.5)/127.5
            except Exception as e:
                print ("Exception =",e)
                continue
            start = time.time()
            rectangles = detectFace(imgpath,threshold,args)
            print (sum, 'time = ', time.time() - start, 's')
            img = cv2.imread(imgpath)
            draw = img.copy()
            for rectangle in rectangles:
                crop_img = draw[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]        
                resized_im = cv2.resize(crop_img, (faceimg_size, faceimg_size), interpolation=cv2.INTER_LINEAR)  
                save_file = os.path.join(facenet_save_dir, cls1+"_%s.jpg"%d_idx)                                       
                cv2.imwrite(save_file, resized_im)                                                               
                d_idx += 1                                                                                      
                cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(0,255,0),2)
            if(len(rectangles)==0):
                t+=1
            print ('online Recall : ', sum - t, sum)
            #cv2.imshow("src-test",draw)
            #cv2.waitKey(500)
    recall = float(sum - t)/float(sum)
    print ('Recall : ', sum - t, sum, recall)
    print ("sum_time: ", time.time() - origin_start, "s")

  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_12_prototxt', type=str,
                        help='int8 12net.prototxt.', default='/mllib/ALG/quant_mtcnn/int8_12net.prototxt')
    parser.add_argument('--net_12_caffemodel', type=str,
                        help='int8 12net.caffemodel.', default='/mllib/ALG/quant_mtcnn/int8_12net.caffemodel')
    parser.add_argument('--net_24_prototxt', type=str,
                        help='int8 12net.prototxt.', default='/mllib/ALG/quant_mtcnn/int8_24net.prototxt')
    parser.add_argument('--net_24_caffemodel', type=str,
                        help='int8 24net.caffemodel.', default='/mllib/ALG/quant_mtcnn/int8_24net.caffemodel')
    parser.add_argument('--net_48_prototxt', type=str,
                        help='int8 48net.prototxt.', default='/mllib/ALG/quant_mtcnn/int8_48net.prototxt')
    parser.add_argument('--net_48_caffemodel', type=str,
                        help='int8 48net.caffemodel.', default='/mllib/ALG/quant_mtcnn/int8_48net.caffemodel')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory.',default='./test_per_data')
    parser.add_argument('--save_dir', type=str,
        help='Path to save the data directory.',default='./test_per_data_67')
    parser.add_argument('--threshold', dest='threshold', help='list of threshold for pnet, rnet, onet', nargs="+", default=[0.8, 0.8, 0.8], type=float)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
