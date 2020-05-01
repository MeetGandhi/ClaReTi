import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

for i in range(len(dtints)):
    dtints[i]=round(dtints[i],3)

classes = ['EW', 'EA', 'RRab', 'RRc', 'RSCVn', 'LPV']
pd={'EW': 0, 'EA': 1, 'RRab': 2, 'RRc': 3, 'RRd': 4, 'RSCVn': 5, 'LPV':6}
prob = np.load("..\\preds_proba.npy")
num = 5

os.mkdir("blanking_ticked_plots_sqrt\\")

for cls in classes:
    
    os.mkdir("blanking_ticked_plots_sqrt\\"+cls+"\\")
    blank = glob.glob(cls+"\\blank\\*.png")
    plt.rcParams['figure.figsize'] = (18, 8)
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    random_blank = np.random.choice(blank, num, False)

    for r in random_blank:
        random_id = int(r.split("\\")[-1].split(".")[0])
        
        img_arr_random_blank = plt.imread(r)
        img_arr_random_dmdt = plt.imread("..\\keras-vis\\dmdt_sqrt\\"+cls+"\\"+str(random_id)+".png")
        
        xloc_blank = np.arange(0, img_arr_random_blank.shape[1], img_arr_random_blank.shape[1]/24.0)
        xloc_blank = np.append(xloc_blank, xloc_blank[-1] + xloc_blank[1])
        yloc_blank = np.arange(0, img_arr_random_blank.shape[0], img_arr_random_blank.shape[0]/22.0)
        yloc_blank = np.append(yloc_blank, yloc_blank[-1] + yloc_blank[1])
        yloc_blank = yloc_blank[::-1]

        xloc_dmdt = np.arange(0, img_arr_random_dmdt.shape[1], img_arr_random_dmdt.shape[1]/24.0)
        xloc_dmdt = np.append(xloc_dmdt, xloc_dmdt[-1] + xloc_dmdt[1])
        yloc_dmdt = np.arange(0, img_arr_random_dmdt.shape[0], img_arr_random_dmdt.shape[0]/22.0)
        yloc_dmdt = np.append(yloc_dmdt, yloc_dmdt[-1] + yloc_dmdt[1])
        yloc_dmdt = yloc_dmdt[::-1]
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(img_arr_random_dmdt)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        ax1.set_xticks(xloc_dmdt)
        ax1.set_xticklabels(dtints,rotation=90)
        ax1.set_yticks(yloc_dmdt)
        ax1.set_yticklabels(dmints)
        ax1.set(xlabel="dt(days)",ylabel="dm(mag)")
        
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(img_arr_random_blank)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        ax2.set_xticks(xloc_blank)
        ax2.set_xticklabels(dtints,rotation=90)
        ax2.set_yticks(yloc_blank)
        ax2.set_yticklabels(dmints)
        ax2.set(xlabel="dt(days)",ylabel="dm(mag)")

        fig.colorbar(im1, cax=cax1, boundaries=np.linspace(img_arr_random_dmdt.min(),img_arr_random_dmdt.max(),10))
        fig.colorbar(im2, cax=cax2, boundaries=np.linspace(img_arr_random_blank.min(),img_arr_random_blank.max(),10))
        plt.tight_layout()
        #plt.suptitle("Class: "+cls+",  Grad-CAM\n"+"X_id: "+str(random_id)+",  Pred_Prob: "+str(round(prob[random_id, pd[cls]],4)))
        #plt.savefig("grad_CAM_ticked_plots_viridify\\"+cls+"\\"+str(random_id)+".png")
        plt.savefig("blanking_ticked_plots_sqrt\\"+cls+"\\"+str(random_id)+"_"+str(round(prob[random_id, pd[cls]],4))+".png")
        plt.close()
