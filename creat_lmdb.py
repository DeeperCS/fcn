import sys
sys.path.insert(0,'/home/cl/caffe-with_crop/python')
# Script to Convert the Data set and ground truth to the required sizes
import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle

# Initialize the Image set
NumberTrain = 2#1464 # Number of Training Images

NumberTest = 2#1449 # Number of Testing Images

Rheight = 200 # Required Height

Rwidth = 200 # Required Width

RheightLabel = 200 # Height for the label

RwidthLabel = 200 # Width for the label

LabelWidth = 200 # Downscaled width of the label

LabelHeight = 200 # Downscaled height of the label

# Read the files in the Data Folder
inputs_data_train = sorted(glob.glob("/home/cl/pascal/image/*.jpg"))
inputs_data_valid = sorted(glob.glob("/home/cl/pascal/image/*.jpg"))
inputs_label = sorted(glob.glob("/home/cl/pascal/groundtruth/*.png"))

shuffle(inputs_data_train) # Shuffle the DataSet
shuffle(inputs_data_valid) # Shuffle the DataSet

inputs_Train = inputs_data_train[:NumberTrain] # Extract the training data from the complete set

inputs_Test = inputs_data_valid[:NumberTest] # Extract the testing data from the complete set
print len(inputs_Train)

# Creating LMDB for Training Data
print("Creating Training Data LMDB File ..... ")

in_db = lmdb.open('TrainVOC_Data_lmdb',map_size=int(1e14))

with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs_Train):
        print in_idx
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        Dtype = im.dtype
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
        im = np.array(im,Dtype)     
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())
in_db.close()

# Creating LMDB for Training Labels
print("Creating Training Label LMDB File ..... ")

in_db = lmdb.open('TrainVOC_Label_lmdb',map_size=int(1e14))

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_label):
        print in_idx
        #in_label = '/home/cl/pascal/'+in_[0:2]+'png'
	L = np.array(Image.open(in_)) # or load whatever ndarray you need
	Dtype = L.dtype
	Limg = Image.fromarray(L)
	Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
	L = np.array(Limg,Dtype)
	L = L.reshape(L.shape[0],L.shape[1],1)
	L = L.transpose((2,0,1))
	L[L==255]=21
	L_dat = caffe.io.array_to_datum(L)
	in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())
in_db.close()

# Creating LMDB for Testing Data
print("Creating Testing Data LMDB File ..... ")

in_db = lmdb.open('TestVOC_Data_lmdb',map_size=int(1e14))

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_Test):
        print in_idx    
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        Dtype = im.dtype
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
        im = np.array(im,Dtype)     
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())
in_db.close()

# Creating LMDB for Testing Labels
print("Creating Testing Label LMDB File ..... ")

in_db = lmdb.open('TestVOC_Label_lmdb',map_size=int(1e14))

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_label):
        print in_idx    
        #in_label = '/home/cl/pascal/'+in_[0:2]+'png'
        L = np.array(Image.open(in_)) # or load whatever ndarray you need
        Dtype = L.dtype
        Limg = Image.fromarray(L)
        Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
        L = np.array(Limg,Dtype)
        L = L.reshape(L.shape[0],L.shape[1],1)
        L = L.transpose((2,0,1))
        L[L==255]=21
        L_dat = caffe.io.array_to_datum(L)
        in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())
in_db.close()

