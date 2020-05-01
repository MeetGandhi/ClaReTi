import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.load("y_test.npy")
y_prediction = np.load("preds.npy")
cnf_matrix = confusion_matrix(y_true, y_prediction)
print(cnf_matrix)

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("Sensitivity, hit rate, recall, or true positive rate:\n", TPR)

TNR = TN/(TN+FP)
print("Specificity or true negative rate:\n", TNR)

PPV = TP/(TP+FP)
print("Precision or positive predictive value:\n", PPV)

NPV = TN/(TN+FN)
print("Negative predictive value:\n", NPV)

FPR = FP/(FP+TN)
print("Fall out or false positive rate:\n", FPR)

FNR = FN/(TP+FN)
print("False negative rate:\n", FNR)

FDR = FP/(TP+FP)
print("False discovery rate:\n", FDR)

ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall accuracy:\n", ACC)
