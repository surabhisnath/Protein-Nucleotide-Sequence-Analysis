
# coding: utf-8

# In[35]:


import numpy as np
import os
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import sys
import warnings
warnings.filterwarnings("ignore")

# In[36]:


aatoind = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 
           'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 
           'M':10, 'N':11, 'P':12, 'Q':13, 'R':14,
           'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}


# In[53]:


def getres(pred, actual, classlab, notclasslab):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        if(pred[i] == actual[i] == classlab):
            tp += 1
        elif(pred[i] == actual[i] == notclasslab):
            tn += 1
        elif(pred[i] == classlab and actual[i] == notclasslab):
            fp += 1
        elif(pred[i] == notclasslab and actual[i] == classlab):
            fn += 1
    #print(tp)
    #print(tn)
    #print(fp)
    #print(fn)
    sens = tp/(tp + fn)
    spec = tn/(tn + fp)
    acc = (tp + tn)/(tp + tn + fp + fn)
    if(tp+fp == 0 or tp+fn == 0 or tn+fp == 0 or tn+fn == 0):
        mcc = 0
    else:
        mcc = ((tp*tn - fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
    return sens, spec, acc, mcc


# In[38]:


def computeaac(seq):
    toret = [0]*20
    den = len(seq)
    for aa in seq:
        toret[aatoind[aa]] += 1
    toret = np.array(toret)
    toret = toret/den
    #print(toret)
    return toret


# In[39]:


filpos = open(sys.argv[2],'r')
filneg = open(sys.argv[4], 'r')
fo = open(sys.argv[6], 'w')


# In[40]:


#Each contains \n
pos = filpos.readlines()
pos = np.array(pos)
neg = filneg.readlines()
neg = np.array(neg)
#print(len(pos))
#print(len(neg))


# In[41]:


x = []
for seq in pos:
    try:
        aac = computeaac(seq[:-1])
        x.append(aac)
    except:
        #print("Except Pos")
        continue

for seq in neg:
    try:
        aac = computeaac(seq[:-1])
        x.append(aac)
    except:
        #print("Except Neg")
        continue

x = np.array(x)
#print(len(x))


# In[42]:


# pos - 0, neg - 1
labelspos = [0] * (len(pos) - 1)
labelsneg = [1] * (len(neg) - 1)
labels = np.concatenate((labelspos, labelsneg), axis = 0)
labels = np.array(labels)
#print(len(labels))


# In[56]:


partitions = KFold(5, shuffle = True)
partitions.get_n_splits(x)
cnt = 1
for train_index, test_index in partitions.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    fo.write("FOLD" + str(cnt) + "\n")
    cnt += 1

    #SVM
    fo.write("SVM\n")
    clf1 = SVC(gamma='auto')
    clf1.fit(x_train, y_train)
    pred1 = clf1.predict(x_test)
    fo.write("Class 0\n")
    sens, spec, acc, mcc = getres(pred1, y_test, 0, 1)
    fo.write("Sensitivity: " + str(sens) + "\n")
    fo.write("Specificity: " + str(spec) + "\n")
    fo.write("Accuracy: " + str(acc) + "\n")
    fo.write("MCC: " + str(mcc) + "\n")
    fo.write("Class 1\n")
    sens, spec, acc, mcc = getres(pred1, y_test, 1, 0)
    fo.write("Sensitivity: " + str(sens) + "\n")
    fo.write("Specificity: " + str(spec) + "\n")
    fo.write("Accuracy: " + str(acc) + "\n")
    fo.write("MCC: " + str(mcc) + "\n")
    #print(classification_report(pred1, y_test))
    
    #ANN
    fo.write("ANN\n")
    clf2 = MLPClassifier()
    clf2.fit(x_train, y_train)
    pred2 = clf2.predict(x_test)
    fo.write("Class 0\n")
    sens, spec, acc, mcc = getres(pred2, y_test, 0, 1)
    fo.write("Sensitivity: " + str(sens) + "\n")
    fo.write("Specificity: " + str(spec) + "\n")
    fo.write("Accuracy: " + str(acc) + "\n")
    fo.write("MCC: " + str(mcc) + "\n")
    fo.write("Class 1\n")
    sens, spec, acc, mcc = getres(pred2, y_test, 1, 0)
    fo.write("Sensitivity: " + str(sens) + "\n")
    fo.write("Specificity: " + str(spec) + "\n")
    fo.write("Accuracy: " + str(acc) + "\n")
    fo.write("MCC: " + str(mcc) + "\n")
    #print(classification_report(pred2, y_test))
    
    #RF
    fo.write("Random Forest\n")
    clf3 = RandomForestClassifier()
    clf3.fit(x_train, y_train)
    pred3 = clf2.predict(x_test)
    fo.write("Class 0\n")
    sens, spec, acc, mcc = getres(pred3, y_test, 0, 1)
    fo.write("Sensitivity: " + str(sens) + "\n")
    fo.write("Specificity: " + str(spec) + "\n")
    fo.write("Accuracy: " + str(acc) + "\n")
    fo.write("MCC: " + str(mcc) + "\n")
    fo.write("Class 1\n")
    sens, spec, acc, mcc = getres(pred3, y_test, 1, 0)
    fo.write("Sensitivity: " + str(sens) + "\n")
    fo.write("Specificity: " + str(spec) + "\n")
    fo.write("Accuracy: " + str(acc) + "\n")
    fo.write("MCC: " + str(mcc) + "\n")
    #print(classification_report(pred3, y_test))

fo.close()