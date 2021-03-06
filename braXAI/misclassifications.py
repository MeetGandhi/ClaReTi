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

x_misclassified = np.load("x_misclassified.npy")
id_misclassified = np.load("id_misclassified.npy")
shap_values = np.load("shap_values_misclassified_2.5k.npy")
diff_prob = np.load("diff_prob_misclassified_blanking.npy")
preds = np.load("preds.npy")
y_test = np.load("y_test.npy")

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

image_shape = x_misclassified.shape[1:]

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

layer_idx = utils.find_layer_idx(model, 'fc_out')
penultimate_layer_idx = utils.find_layer_idx(model, 'conv4')
pd={0:'Bogus',1:'Real'}

for i in range(x_misclassified.shape[0]):
  img = x_misclassified[i]
  img_cam = visualize_cam(model, layer_idx, filter_indices=preds[id_misclassified[i]], seed_input=img, penultimate_layer_idx=penultimate_layer_idx)
  img_sal = visualize_saliency(model, layer_idx, filter_indices=preds[id_misclassified[i]], seed_input=img)

  fig = plt.figure()
  ax1 = fig.add_subplot(331)
  ax1.axis('off')
  ax1.imshow(x_misclassified[i][:, :, 0], origin='upper', cmap=plt.cm.bone)
  ax1.title.set_text("SCI")
  ax2 = fig.add_subplot(332)
  ax2.axis('off')
  ax2.imshow(x_misclassified[i][:, :, 1], origin='upper', cmap=plt.cm.bone)
  ax2.title.set_text("REF")
  ax3 = fig.add_subplot(333)
  ax3.axis('off')
  ax3.imshow(x_misclassified[i][:, :, 2], origin='upper', cmap=plt.cm.bone)
  ax3.title.set_text("DIFF")

  ax4 = fig.add_subplot(334)
  ax4.axis('off')
  ax4.imshow(img_cam)
  ax4.title.set_text("grad_CAM")
  ax5 = fig.add_subplot(335)
  ax5.axis('off')
  ax5.imshow(img_sal)
  ax5.title.set_text("Saliency")
  ax6 = fig.add_subplot(336)
  ax6.axis('off')
  ax6.imshow(diff_prob[i])
  ax6.title.set_text("Blanking exp.")

  ax7 = fig.add_subplot(337)
  ax7.axis('off')
  ax7.imshow(shap_values[preds[id_misclassified[i]]][i][:,:,0])
  ax7.title.set_text("SHAP plot of SCI")
  ax8 = fig.add_subplot(338)
  ax8.axis('off')
  ax8.imshow(shap_values[preds[id_misclassified[i]]][i][:,:,1])
  ax8.title.set_text("SHAP plot of REF")
  ax9 = fig.add_subplot(339)
  ax9.axis('off')
  ax9.imshow(shap_values[preds[id_misclassified[i]]][i][:,:,2])
  ax9.title.set_text("SHAP plot of DIFF")

  plt.suptitle("Prediction: "+pd[preds[id_misclassified[i]]]+"  Class: "+pd[y_test[id_misclassified[i]]])
  plt.savefig("misclassifications/"+str(id_misclassified[i])+".png")

  plt.close()



















