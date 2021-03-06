# -*- coding: utf-8 -*-
"""pix2pix_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1prb4RcGjgsweVTZF5G1URcqMxKSMX4Sk
"""

#!pip3 install scipy==1.1.0
#!pip install texttable

#from google.colab import drive
#drive.mount('/gdrive')
# %cd /gdrive

import configparser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from six.moves import cPickle
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stats
import glob
from texttable import Texttable

#import matplotlib
#matplotlib.use('Agg')

import copy
from sklearn import metrics
import seaborn
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.engine.topology import Input
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from vis.visualization import visualize_activation, visualize_saliency, visualize_cam
from vis.utils import utils
from keras import activations

from IPython.display import HTML, display
import tqdm
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix
import datetime
from astropy.time import Time
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import normalize as tf_norm

import io
import gzip
from astropy.io import fits
#from bson.json_util import loads, dumps

import matplotlib.pyplot as plt
# plt.style.use(['dark_background'])
from pandas.plotting import register_matplotlib_converters, scatter_matrix
register_matplotlib_converters()
#%matplotlib inline

x=np.load("x_test.npy")
y=np.load("y_test.npy")
preds=np.load("preds.npy")
prob=np.load("preds_proba.npy")

##def load_model_helper(path, model_base_name):
##    """
##        Build keras model using json-file with architecture and hdf5-file with weights
##    """
##    with open(os.path.join(path, f'{model_base_name}.architecture.json'), 'r') as json_file:
##        loaded_model_json = json_file.read()
##    m = model_from_json(loaded_model_json)
##    m.load_weights(os.path.join(path, f'{model_base_name}.weights.h5'))
##
##    return m
##
##model = load_model_helper(path='G:\\Caltech SURF 2018\\Desktop_2019\\braai\\', model_base_name='d6_m7')
##print(model.summary())

def vgg6(input_shape=(63, 63, 3), n_classes: int = 1):
    """
        VGG6
    :param input_shape:
    :param n_classes:
    :return:
    """

    model = keras.models.Sequential(name='VGG6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D(pool_size=(4, 4)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256, activation='relu', name='fc_1'))
    model.add(keras.layers.Dropout(0.5))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(keras.layers.Dense(n_classes, activation=activation, name='fc_out'))

    return model

loss = 'sparse_categorical_crossentropy'
optimizer = 'adam'

image_shape = x.shape[1:]

binary_classification = True if loss == 'sparse_categorical_crossentropy' else False
n_classes = 2 if binary_classification else 1

model = vgg6(input_shape=image_shape, n_classes=n_classes)
# Swap softmax with linear
#layer_idx = utils.find_layer_idx(model, 'fc_out')
#model.layers[layer_idx].activation = activations.linear
#model = utils.apply_modifications(model)

# set up optimizer:
if optimizer == 'adam':
    optimzr = keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,
                                       epsilon=None, decay=0.0, amsgrad=False)
elif optimizer == 'sgd':
    optimzr = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
else:
    print('Could not recognize optimizer, using Adam')
    optimzr = keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,
                                       epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=optimzr, loss=loss, metrics=['accuracy'])
model.load_weights('model.h5')
print(model.summary())

# x=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/X_test.npy")
# y=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/y_test.npy")
# preds=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/preds.npy")
# prob=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/preds_proba.npy")

#classes=[1,2,4,5,6,8,13]
classes=[0, 1]
filterid={0:0, 1:1}
pd={0:'bogus', 1:'real'}
#pd={1:'EW',2:'EA',4:'RRab',5:'RRc',6:'RRd',8:'RSCVn',13:'LPV'}

#dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
#dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

#xloc=np.arange(25)
#yloc=np.arange(23)
#yloc=yloc[::-1]
#for i in range(len(dtints)):
#    dtints[i]=round(dtints[i],3)
#yloc=yloc-0.5
#xloc=xloc-0.5

thres=np.where(prob>=0.8)
ind_thres=thres[0]
class_thres=thres[1]

#plt.rcParams['figure.figsize'] = (18, 6)
layer_idx = utils.find_layer_idx(model, 'fc_out')
penultimate_layer_idx = utils.find_layer_idx(model, 'conv4')

# os.mkdir("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/keras-vis/grad_CAM/dmdt/")
# os.mkdir("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/keras-vis/grad_CAM/gradcam/")

##os.mkdir("keras-vis/")

###os.mkdir("keras-vis/grad_CAM/")
###os.mkdir("keras-vis/grad_CAM/triplet/")
###os.mkdir("keras-vis/grad_CAM/gradcam/")

###os.mkdir("keras-vis/real/saliency/")

for i in range(len(classes)):
   # This is the output node we want to maximize.
   filter_idx = filterid[classes[i]]
   
   ###os.mkdir("keras-vis/"+pd[classes[i]]+"/")
   ###os.mkdir("keras-vis/"+pd[classes[i]]+"/grad_CAM/")
   ###os.mkdir("keras-vis/"+pd[classes[i]]+"/triplet/")
   os.mkdir("keras-vis/"+pd[classes[i]]+"/saliency/")
   #input_range= (0,255)
   #image=np.random.random_sample((22,24,1))
   x_ind=np.where(class_thres==filter_idx)[0]
   ###init=np.where(ind_thres[x_ind]==8882)[0][0]
   init=0
   for ii in range(init,x_ind.shape[0]):
       #image=x[ind_thres[x_ind[ii]]].transpose((1,2,0))
       image=x[ind_thres[x_ind[ii]]]
       ###img_cam = visualize_cam(model, layer_idx, filter_indices=filter_idx, seed_input=image, 
       ###                             penultimate_layer_idx=penultimate_layer_idx)
       img_sal = visualize_saliency(model, layer_idx, filter_indices=filter_idx, seed_input=image) 
       #                      penultimate_layer_idx=penultimate_layer_idx
       #plt.figure()
       #fig, ax=plt.subplots(1,2)
       #fig, ax=plt.subplots(1,1)
       #im1=ax.imshow(image[:,:,0])
       ###plt.imshow(image[:,:,0])
       
       ###fig = plt.figure(figsize=(8, 2), dpi=100)
       ###ax = fig.add_subplot(131)
       ###ax.axis('off')
       ###ax.imshow(image[:, :, 0], origin='upper', cmap=plt.cm.bone)
       ###ax2 = fig.add_subplot(132)
       ###ax2.axis('off')
       ###ax2.imshow(image[:, :, 1], origin='upper', cmap=plt.cm.bone)
       ###ax3 = fig.add_subplot(133)
       ###ax3.axis('off')
       ###ax3.imshow(image[:, :, 2], origin='upper', cmap=plt.cm.bone)
       #plt.imshow()
       ###plt.axis('off')
##       plt.xticks(xloc,dtints,rotation=90)
##       plt.yticks(yloc,dmints)
       #plt.set_xticks(xloc)
       #ax.set_xticks(xloc)
       #plt.set_xticklabels(dtints,rotation=90)
       #plt.set_yticks(yloc)
       #plt.set_yticklabels(dmints)
       #ax.set(xlabel="dt(days)",ylabel="dm(mag)")
       ###plt.savefig("keras-vis/"+pd[classes[i]]+"/triplet/"+str(ind_thres[x_ind[ii]])+".png", bbox_inches='tight', transparent="True", pad_inches=0)
       ###plt.close()
       #plt.clf()
       
       plt.figure()
       #fig, ax=plt.subplots(1,2)
       #fig, ax=plt.subplots(1,1)
       ###plt.imshow(img_cam)
       plt.imshow(img_sal)
       plt.axis('off')
##       plt.xticks(xloc,dtints,rotation=90)
##       plt.yticks(yloc,dmints)
       #plt.set_xticks(xloc)
       #plt.set_xticklabels(dtints,rotation=90)
       #plt.set_yticks(yloc)
       #plt.set_yticklabels(dmints)
       #ax.set(xlabel="dt(days)",ylabel="dm(mag)")
       ###plt.savefig("keras-vis/"+pd[classes[i]]+"/grad_CAM/"+str(ind_thres[x_ind[ii]])+".png", bbox_inches='tight', transparent="True", pad_inches=0)
       plt.savefig("keras-vis/"+pd[classes[i]]+"/saliency/"+str(ind_thres[x_ind[ii]])+".png", bbox_inches='tight', transparent="True", pad_inches=0)
       ###plt.savefig("keras-vis/real/saliency/"+str(ind_thres[x_ind[ii]])+".png", bbox_inches='tight', transparent="True", pad_inches=0)
       plt.close()
       #plt.clf()
        
       #im2=ax[1].imshow(img_cam)
       #divider1 = make_axes_locatable(ax[0])
       #cax1 = divider1.append_axes("right", size="5%", pad=0.1)
       #divider2 = make_axes_locatable(ax[1])
       #cax2 = divider2.append_axes("right", size="5%", pad=0.1)
       #ax[0].set_xticks(xloc)
       #ax[0].set_xticklabels(dtints,rotation=90)
       #ax[1].set_xticks(xloc)
       #ax[1].set_xticklabels(dtints,rotation=90)
       #ax[0].set_yticks(yloc)
       #ax[0].set_yticklabels(dmints)
       #ax[1].set_yticks(yloc)
       #ax[1].set_yticklabels(dmints)
       #ax[0].set(xlabel="dt(days)",ylabel="dm(mag)")
       #ax[1].set(xlabel="dt(days)",ylabel="dm(mag)")
       #fig.colorbar(im1, cax=cax1)
       #fig.colorbar(im2, cax=cax2)
       #plt.tight_layout()
       #plt.suptitle("Class: "+pd[classes[i]]+", grad_CAM,\n"+"X_id: "+str(ind_thres[x_ind[ii]])+"\nPred_Prob: "+str(round(prob[ind_thres[x_ind[ii]],class_thres[x_ind[ii]]],4)))
       #plt.savefig("keras-vis/grad_CAM/"+pd[classes[i]]+"/"+str(ind_thres[x_ind[ii]])+".png")
       #plt.close()
