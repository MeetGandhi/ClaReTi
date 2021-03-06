import numpy as np
import os
import glob

classes=["bogus", "real"]

y=np.load("y_test.npy")
pr=np.load("preds.npy")
prob=np.load("preds_proba.npy")
#index = {1:0, 2:1, 4:2, 5:3, 6:4, 8:5, 13:6}
index = {0:0, 1:1}

for clss in classes:
    #l=glob.glob("trainingdata\\saliency\\"+clss+"\\*.png")
    l=glob.glob("keras-vis\\"+clss+"\\shap_sub\\*.png")
    #l=glob.glob("keras-vis\\saliency\\"+clss+"\\*.png")
    for loc in range(len(l)):
        i=int(l[loc].split("\\")[-1][:-4])
        if y[i]!=pr[i] or prob[i,index[pr[i]]]<0.8:
            os.remove(l[loc])
        
