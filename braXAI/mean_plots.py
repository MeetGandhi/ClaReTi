import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
x = np.concatenate((x_train, x_test))

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
y = np.concatenate((y_train, y_test))

classes = [0, 1]
pd={0:'bogus', 1:'real'}

##dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
##dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]
##
##xloc=np.arange(25)
##yloc=np.arange(23)
##yloc=yloc[::-1]
##for i in range(len(dtints)):
##    dtints[i]=round(dtints[i],3)
##yloc=yloc-0.5
##xloc=xloc-0.5

os.mkdir("mean_plots/")
#plt.rcParams['figure.figsize'] = (18, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams.update({'font.size': 12})

for cls in classes:
    id_ = np.where(y==cls)[0]
    length = id_.shape[0]

    x_sum_1 = np.zeros((63,63))
    x_sum_2 = np.zeros((63,63))
    x_sum_3 = np.zeros((63,63))
    
    for j in range(length):
        x_sum_1 = np.add(x_sum_1, x[id_[j],:,:,0])
        x_sum_2 = np.add(x_sum_2, x[id_[j],:,:,1])
        x_sum_3 = np.add(x_sum_3, x[id_[j],:,:,2])

    x_mean = np.zeros((63,63,3))

    x_mean[:,:,0] = x_sum_1/length
    x_mean[:,:,1] = x_sum_2/length
    x_mean[:,:,2] = x_sum_3/length
    
    mtype = {0:'SCI',1:'REF',2:'DIFF'}

    for m in range(3):
        
        plt.figure()
        fig, ax = plt.subplots(1,1)
        im1 = ax.imshow(x_mean[:,:,m], origin='upper', cmap=plt.cm.bone)
        divider1 = make_axes_locatable(ax)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        #ax.set_xticks(xloc)
        #ax.set_xticklabels(dtints,rotation=90)
        #ax.set_yticks(yloc)
        #ax.set_yticklabels(dmints)
        #ax.set(xlabel="dt(days)",ylabel="dm(mag)")
        fig.colorbar(im1, cax=cax1, boundaries=np.linspace(0,x_mean[:,:,m].max(),10))
        ax.axis("off")
        #plt.suptitle("Class: "+pd[cls]+"  X_id: "+str(i))
        plt.tight_layout()
        plt.savefig("mean_plots/"+pd[cls]+"_"+mtype[m]+".png")
        plt.cla()
        plt.clf()
        plt.close()

        
        
