import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib
import numpy as np
import glob
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def changecolormap(image, origin_cmap, target_cmap):
    r = np.linspace(0,1, 256)
    norm = matplotlib.colors.Normalize(0,1)
    mapvals = origin_cmap(norm(r))[:,:3]

    def get_value_from_cm(color):
        color=matplotlib.colors.to_rgb(color)
        #if color is already gray scale, dont change it
        if np.std(color) < 0.1:
            return color
        #otherwise return value from colormap
        distance = np.sum((mapvals - color)**2, axis=1)
        return target_cmap(r[np.argmin(distance)])[:3]

    newim = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            c = image[i,j,:3]
            newim[i,j, :3] =  get_value_from_cm(c)
    return newim


dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

for i in range(len(dtints)):
    dtints[i]=round(dtints[i],3)

classes = ['EW', 'EA', 'RRab', 'RRc', 'RSCVn', 'LPV']
pd={'EW': 0, 'EA': 1, 'RRab': 2, 'RRc': 3, 'RRd': 4, 'RSCVn': 5, 'LPV':6}
prob = np.load("..\\preds_proba.npy")
num = 5

os.mkdir("grad_CAM_ticked_plots_viridify_sqrt\\")

for cls in classes:
    
    os.mkdir("grad_CAM_ticked_plots_viridify_sqrt\\"+cls+"\\")
    blank = glob.glob("grad_CAM\\gradcam\\"+cls+"\\*.png")
    plt.rcParams['figure.figsize'] = (18, 8)
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    random_blank = np.random.choice(blank, num, False)

    for r in random_blank:
        random_id = int(r.split("\\")[-1].split(".")[0])
        
        img_arr_random_blank = plt.imread(r)[:,:,:3]
        img_arr_random_dmdt = plt.imread("dmdt_sqrt\\"+cls+"\\"+str(random_id)+".png")
        
        xloc = np.arange(0, img_arr_random_blank.shape[1], img_arr_random_blank.shape[1]/24)
        xloc = np.append(xloc, xloc[-1] + xloc[1])
        yloc = np.arange(0, img_arr_random_blank.shape[0], img_arr_random_blank.shape[0]/22)
        yloc = np.append(yloc, yloc[-1] + yloc[1])
        yloc = yloc[::-1]

        fig = plt.figure()
        
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(img_arr_random_dmdt)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        ax1.set_xticks(xloc)
        ax1.set_xticklabels(dtints,rotation=90)
        ax1.set_yticks(yloc)
        ax1.set_yticklabels(dmints)
        ax1.set(xlabel="dt(days)",ylabel="dm(mag)")
        
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(changecolormap(img_arr_random_blank, plt.cm.jet, plt.cm.viridis))
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        ax2.set_xticks(xloc)
        ax2.set_xticklabels(dtints,rotation=90)
        ax2.set_yticks(yloc)
        ax2.set_yticklabels(dmints)
        ax2.set(xlabel="dt(days)",ylabel="dm(mag)")

        fig.colorbar(im1, cax=cax1, boundaries=np.linspace(img_arr_random_dmdt.min(),img_arr_random_dmdt.max(),10))
        fig.colorbar(im2, cax=cax2, boundaries=np.linspace(img_arr_random_blank.min(),img_arr_random_blank.max(),10))
        plt.tight_layout()
        #plt.suptitle("Class: "+cls+",  Grad-CAM\n"+"X_id: "+str(random_id)+",  Pred_Prob: "+str(round(prob[random_id, pd[cls]],4)))
        #plt.savefig("grad_CAM_ticked_plots_viridify\\"+cls+"\\"+str(random_id)+".png")
        plt.savefig("grad_CAM_ticked_plots_viridify_sqrt\\"+cls+"\\"+str(random_id)+"_"+str(round(prob[random_id, pd[cls]],4))+".png")
        plt.close()

##        fig, ax=plt.subplots(1,2)
##        im1=ax[0].imshow(img_arr_random_dmdt)
##        im2=ax[1].imshow(changecolormap(img_arr_random_blank, plt.cm.jet, plt.cm.viridis))
##        divider1 = make_axes_locatable(ax[0])
##        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
##        divider2 = make_axes_locatable(ax[1])
##        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
##        ax[0].set_xticks(xloc)
##        ax[0].set_xticklabels(dtints,rotation=90)
##        ax[1].set_xticks(xloc)
##        ax[1].set_xticklabels(dtints,rotation=90)
##        ax[0].set_yticks(yloc)
##        ax[0].set_yticklabels(dmints)
##        ax[1].set_yticks(yloc)
##        ax[1].set_yticklabels(dmints)
##        ax[0].set(xlabel="dt(days)",ylabel="dm(mag)")
##        ax[1].set(xlabel="dt(days)",ylabel="dm(mag)")
##        fig.colorbar(im1, cax=cax1)
##        fig.colorbar(im2, cax=cax2)
##        plt.tight_layout()
##        plt.suptitle("Class: "+cls+",  Grad-CAM\n"+"X_id: "+str(random_id)+",  Pred_Prob: "+str(round(prob[random_id, pd[cls]],4)))
##        plt.savefig("grad_CAM_ticked_plots_viridify\\"+cls+"\\"+str(random_id)+".png")
##        plt.close()

        
##        fig, ax = plt.subplots(1,1)
##        img = ax.imshow(img_arr)
##        ax.set_xticks(xloc)
##        ax.set_xticklabels(dtints, rotation=90)
##        ax.set_xlabel("dt(days)")
##        ax.set_yticks(yloc)
##        ax.set_yticklabels(dmints)
##        ax.set_ylabel("dm(mag)")
##        plt.title(l[i].split(".")[0])
##        plt.savefig(l[i].split(".")[0]+"_ticked"+".png")
##        plt.close()
