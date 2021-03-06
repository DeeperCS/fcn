# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:46:08 2016

@author: joe
"""

import sys
sys.path.insert(0, '/home/joe/github/caffe-4-17/caffe/python/')
import caffe


#---------------------Model 1------------------------#
# Load conv layers 
net_full_conv = caffe.Net('VGG_ILSVRC_16_layers_ful_conv_deploy.prototxt', 'vgg16-full-conv.caffemodel', caffe.TEST)

# Because the layer name has changed, so these layers below will ignored to be load
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']

conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)



#---------------------Model 2------------------------#
net_fc = caffe.Net('vgg16fc.prototxt', 'vgg16-full-conv.caffemodel', caffe.TEST)

params = ['fc6', 'fc7', 'fc8']

fc_params = {pr:(net_fc.params[pr][0].data, net_fc.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)



#---------------------Transplant------------------------#

# transplant the parameters
for pr, pr_conv in zip(params, params_full_conv):
    fc_params[pr][0].flat = conv_params[pr_conv][0].flat  # flat unrolls the arrays
    fc_params[pr][1][...] = conv_params[pr_conv][1]
    
net_fc.save('vgg16fc.caffemodel')