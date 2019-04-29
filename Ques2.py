
# coding: utf-8

# In[24]:


import numpy as np
import math
import os
import re
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import sys

# In[15]:


aatoind = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 
           'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 
           'M':10, 'N':11, 'P':12, 'Q':13, 'R':14,
           'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}


# In[16]:


def computeaac(seq):
    toret = [0]*20
    den = len(seq)
    for aa in seq:
        toret[aatoind[aa]] += 1
    toret = np.array(toret)
    toret = toret/den
    #print(toret)
    return toret


# In[17]:


def computeatc(seq):
    toret = [0]*5 # C, H, N, O, S
    den = 0
    for aa in seq:
        toret[0] += d_carbon[aa]
        toret[1] += d_hydrogen[aa]
        toret[2] += d_nitrogen[aa]
        toret[3] += d_oxygen[aa]
        toret[4] += d_sulphur[aa]
        den += d_atoms[aa]
    toret = np.array(toret)
    toret = toret/den
    #print(toret)
    return toret


# In[18]:


d_atoms = {'A':13, 'R':26, 'N':17, 'D':16, 'C':14, 
           'Q':20, 'E':19, 'G':10, 'H':20, 'I':22, 
           'L':22, 'K':24, 'M':20, 'F':23, 'P':17,
           'S':14, 'T':17, 'W':27, 'Y':24, 'V':19}
d_carbon = {'A':3, 'C':3, 'D':4, 'E':5, 'F':9, 
           'G':2, 'H':6, 'I':6, 'K':6, 'L':6, 
           'M':5, 'N':4, 'P':5, 'Q':5, 'R':6,
           'S':3, 'T':4, 'V':5, 'W':11, 'Y':9}
d_hydrogen = {'A':7, 'C':7, 'D':7, 'E':9, 'F':11, 
           'G':5, 'H':9, 'I':13, 'K':14, 'L':13, 
           'M':11, 'N':8, 'P':9, 'Q':10, 'R':14,
           'S':7, 'T':9, 'V':11, 'W':12, 'Y':11}
d_nitrogen = {'A':1, 'C':1, 'D':1, 'E':1, 'F':1, 
           'G':1, 'H':3, 'I':1, 'K':2, 'L':1, 
           'M':1, 'N':2, 'P':1, 'Q':2, 'R':4,
           'S':1, 'T':1, 'V':1, 'W':2, 'Y':1}
d_oxygen = {'A':2, 'C':2, 'D':4, 'E':4, 'F':2, 
           'G':2, 'H':2, 'I':2, 'K':2, 'L':2, 
           'M':2, 'N':3, 'P':2, 'Q':3, 'R':2,
           'S':3, 'T':3, 'V':2, 'W':2, 'Y':3}
d_sulphur = {'A':0, 'C':1, 'D':0, 'E':0, 'F':0, 
           'G':0, 'H':0, 'I':0, 'K':0, 'L':0, 
           'M':1, 'N':0, 'P':0, 'Q':0, 'R':0,
           'S':0, 'T':0, 'V':0, 'W':0, 'Y':0}


# In[19]:

folderpath = sys.argv[2]
fo = open(sys.argv[4], 'w')

lines = ""
for file in os.listdir(folderpath):
    fil = open(folderpath + file, 'r')
    lines += fil.read()
    #print(len(lines))
    #print(lines)


# In[20]:


starts = [m.start() for m in re.finditer('::', lines)]
ends = [n.start() for n in re.finditer('>', lines)]
breaks = [o.start() for o in re.finditer('\n', lines)]
ends = ends[1:]
starts = np.array(starts)
starts = starts + 2
ends = np.array(ends)
ends = ends - 1
breaks = np.array(breaks)
#print(starts)
#print(ends)
#print(breaks)


# In[21]:


x = []
for i in range(len(starts)):
    seq = lines[starts[i] : breaks[i]]
    #print(seq)
    aac = computeaac(seq)
    #aac is a vector of 20
    atc = computeatc(seq)
    #atc is a vector of 5
    fv = np.concatenate((aac, atc), axis = 0)
    x.append(fv)


# In[9]:


x_train = x[:500]
x_test = x[500:]


# In[27]:


clusters = [2, 3, 4, 5, 10, 15, 20, 25]


# In[33]:

sc = 0
num_clust = 0
for numclust in clusters:
    kmeans = KMeans(n_clusters=numclust, random_state=0).fit(x)
    labels = kmeans.labels_
    score = silhouette_score(x, labels)
    if(score > sc):
        sc = score
        num_clust = numclust

fo.write("Kmeans\n")
fo.write("The optimal number of clusters is " + str(num_clust) + " with a Silhoutte score of " + str(sc) + "\n")


# In[32]:


#To do heirarchical clustering
sc = 0
num_clust = 0
for numclust in clusters:
    heir = AgglomerativeClustering(n_clusters=numclust).fit(x)
    labels = heir.labels_
    score = silhouette_score(x, labels)
    if(score > sc):
        sc = score
        num_clust = numclust

fo.write("Heirarchical Clustering\n")
fo.write("The optimal number of clusters is " + str(num_clust) + " with a Silhoutte score of " + str(sc) + "\n")
fo.close()