import numpy as np
import os
import glob

classes=["EW","EA","RRab","RRc","RRd","RSCVn","LPV"]

y=np.load("y_test.npy")
pr=np.load("preds.npy")

for clss in classes:
    #l=glob.glob("trainingdata\\saliency\\"+clss+"\\*.png")
    l=glob.glob("keras-vis\\dmdt_sqrt\\"+clss+"\\*.png")
    #l=glob.glob("keras-vis\\saliency\\"+clss+"\\*.png")
    for loc in range(len(l)):
        i=int(l[loc].split("\\")[-1][:-4])
        if y[i]!=pr[i]:
            os.remove(l[loc])
        
