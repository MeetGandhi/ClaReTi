import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

x_train = np.load("X_train.npy")
x_test = np.load("X_test.npy")
x = np.concatenate((x_train, x_test))

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
y = np.concatenate((y_train, y_test))

classes = [1,2,4,5,6,8,13]
pd={1:'EW',2:'EA',4:'RRab',5:'RRc',6:'RRd',8:'RSCVn',13:'LPV'}

dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

xloc=np.arange(25)
yloc=np.arange(23)
yloc=yloc[::-1]
for i in range(len(dtints)):
    dtints[i]=round(dtints[i],3)
yloc=yloc-0.5
xloc=xloc-0.5

os.mkdir("stacked_dmdt_plots/")
#plt.rcParams['figure.figsize'] = (8, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams.update({'font.size': 14})

for cls in classes:
    id_ = np.where(y==cls)[0]
    length = id_.shape[0]

    fig = plt.figure(frameon=False)
    
    for j in range(length):
        plt.imshow(x[id_[j]][:,:,0])

    ax = plt.gca()
    ax.set_xticks(xloc)
    ax.set_xticklabels(dtints,rotation=90)
    ax.set_yticks(yloc)
    ax.set_yticklabels(dmints)
    ax.set(xlabel="dt(days)",ylabel="dm(mag)")
    #fig.colorbar(im1, cax=cax1)
    #plt.suptitle("Class: "+pd[cls]+"  X_id: "+str(i))
    plt.tight_layout()
    plt.savefig("stacked_dmdt_plots/"+pd[cls]+".png")
    plt.cla()
    plt.clf()
    plt.close()

        
        
