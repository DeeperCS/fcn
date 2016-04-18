# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:14:38 2016

@author: joe
"""
import sys
sys.path.insert(0, '/home/joe/github/caffe-4-17/caffe/python/')

import caffe
import numpy as np
from PIL import Image
import pylab

if __name__=='__main__':
    
    weights = 'snapshot/train_iter_16000.caffemodel'

    # init
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    inputIm = '2007_000129.jpg'
    im = Image.open(inputIm)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    
    # load net
    net = caffe.Net('deploy.prototxt', weights, caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    pylab.imshow(out)
    
    
    pallete = [ 0,0,0,
            128,0,0,
            0,128,0,
            128,128,0,
            0,0,128,
            128,0,128,
            0,128,128,
            128,128,128,
            64,0,0,
            192,0,0,
            64,128,0,
            192,128,0,
            64,0,128,
            192,0,128,
            64,128,128,
            192,128,128,
            0,64,0,
            128,64,0,
            0,192,0,
            128,192,0,
            0,64,128 ]
            
    out_img = Image.fromarray(out.astype(np.uint8))
    out_img.putpalette(pallete)
    out_img.save(inputIm.replace('jpg', 'png'))