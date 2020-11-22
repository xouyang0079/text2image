#coding:utf-8
# the input is the 400 dimension random noise
import sys
sys.path.append('..')
import glob

import os
import json
import cPickle
from matplotlib import pyplot as plt
import time
import numpy as np
from collections import OrderedDict
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano import config
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



import os, os.path

import vgg16_gpu
import lasagne
import pickle
from PIL import Image



k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
batchsize = 4    # # of examples in batch
nlen = 15         # # of words in sentence
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 25        # # of iter at starting learning rate
nepochs = 10000   # # of epochs in training
niter_decay = nepochs   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntest = 1020     # number of testing data

X = T.tensor4()     # (batchsize*len)*3*64*64

model = pickle.load(open('vgg16.pkl'))

nnet_fc7d = vgg16_gpu.build_model()

X7d = lasagne.layers.get_output(nnet_fc7d, X, deterministic=True)

print 'COMPILING'
t = time.time()
_gen = theano.function([X], X7d)
print '%.2f seconds to compile theano functions'%(time.time()-t)

path  = "./pdf_11/test/gen"
filepath = os.path.join(path, '*.png')

path1  = "./pdf_11/test/real"
filepath1 = os.path.join(path1, '*.png')


image = np.empty((ntest, 3, 64, 64), dtype='float32')
image1 = np.empty((ntest, 3, 64, 64), dtype='float32')
i = 0
for files1, files2 in zip(sorted(glob.glob(filepath)), sorted(glob.glob(filepath1))):
    img = Image.open(files1)
    img = np.asarray(img, dtype="float32")
    img = np.reshape(img, (3, 64, 64))/255.0
    image[i] = img

    img1 = Image.open(files2)
    img1 = np.asarray(img1, dtype="float32")
    img1 = np.reshape(img1, (3, 64, 64))/255.0
    image1[i] = img1

    i += 1

imagev1 = np.empty((ntest, 1000), dtype='float32')
imagev3 = np.empty((ntest, 1000), dtype='float32')


for j in range(ntest):
    # fake
    x1v = _gen(image[j][None,:,:,:])
    imagev1[j] = x1v[0]


    # real
    x1v1 = _gen(image1[j][None,:,:,:])
    imagev3[j] = x1v1[0]


cPickle.dump([imagev1, imagev3], open("./test_11f1.pkl", 'wb'))






