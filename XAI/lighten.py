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

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", 
                 #data_format = "channels_first",
                 kernel_regularizer=regularizers.l2(0.01), 
                 input_shape=(22,24,1),
                 padding='valid',
                 name="conv2d1"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', 
                 #data_format = "channels_first",
                 name="maxpool2d1",
                 strides=(2,2)))
#model.add(Dropout(rate=0.1))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", 
                 kernel_regularizer=regularizers.l2(0.01),
                 padding='valid', 
                 #data_format = "channels_first",
                 name="conv2d2"))
#model.add(Conv2D(filters=256, kernel_size=(5, 5), activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Flatten(
    #data_format = "channels_first", 
    name="flatten"))
#model.add(Dense(units=512, kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(rate=0.5))
#model.add(Dense(units=512, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=7, activation="softmax", name="preds"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.load_weights('/gdrive/My Drive/workflow/periodic/code/experiments/cnn/model.h5')
model.load_weights('model.h5')
model.summary()

# x=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/X_test.npy")
# y=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/y_test.npy")
# preds=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/preds.npy")
# prob=np.load("/gdrive/My Drive/workflow/periodic/code/experiments/cnn/preds_proba.npy")

x=np.load("X_test.npy")
y=np.load("y_test.npy")
preds=np.load("preds.npy")
prob=np.load("preds_proba.npy")

classes=[1,2,4,5,6,8,13]

p_1=np.zeros((22,24))
p_2=np.zeros((22,24))
p_4=np.zeros((22,24))
p_5=np.zeros((22,24))
p_6=np.zeros((22,24))
p_8=np.zeros((22,24))
p_13=np.zeros((22,24))

for j in range(22):
    for k in range(24):
        x=np.zeros((22,24))
        x[j,k]=255.0
        [p_1[j,k],p_2[j,k],p_4[j,k],p_5[j,k],p_6[j,k],p_8[j,k],p_13[j,k]]=model.predict(x.reshape((1,22,24,1)))[0]

os.mkdir("lighten")
np.save("lighten/lighten_EW",p_1)
np.save("lighten/lighten_EA",p_2)
np.save("lighten/lighten_RRab",p_4)
np.save("lighten/lighten_RRc",p_5)
np.save("lighten/lighten_RRd",p_6)
np.save("lighten/lighten_RSCVn",p_8)
np.save("lighten/lighten_LPV",p_13)
np.save("lighten/x",x)


dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

xloc=np.arange(25)
yloc=np.arange(23)
yloc=yloc[::-1]
for i in range(len(dtints)):
    dtints[i]=round(dtints[i],3)
yloc=yloc-0.5
xloc=xloc-0.5

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_1)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class1: EW"+"\nMin: "+str(round(p_1.min(),5))+" Violet"+"\n Max: "+str(round(p_1.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_EW.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_2)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class2: EA"+"\nMin: "+str(round(p_2.min(),5))+" Violet"+"\n Max: "+str(round(p_2.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_EA.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_4)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class4: RRab"+"\nMin: "+str(round(p_4.min(),5))+" Violet"+"\n Max: "+str(round(p_4.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_RRab.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_5)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class5: RRc"+"\nMin: "+str(round(p_5.min(),5))+" Violet"+"\n Max: "+str(round(p_5.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_RRc.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_6)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class6: RRd"+"\nMin: "+str(round(p_6.min(),5))+" Violet"+"\n Max: "+str(round(p_6.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_RRd.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_8)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class8: RSCVn"+"\nMin: "+str(round(p_8.min(),5))+" Violet"+"\n Max: "+str(round(p_8.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_RSCVn.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_13)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
#ax.set_title("Activation Probabilities for class13: LPV"+"\nMin: "+str(round(p_13.min(),5))+" Violet"+"\n Max: "+str(round(p_13.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/lighten_LPV.png")
plt.close()
