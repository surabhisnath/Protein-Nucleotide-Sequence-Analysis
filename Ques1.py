
# coding: utf-8

# In[2]:


#Dot plot


# In[3]:


import numpy as np
import cv2
import sys


# In[4]:


#Given a sequence, find dotplot matrix and find repeats and inverse repeats


# In[5]:


def evalmat(seq, mat, img):
    for i in range(len(seq)):
        for j in range(len(seq)):
            if(seq[i] == seq[j] and i != j):
                mat[i,j] = 1
                img[i,j] = 255


# In[6]:


def check4ones(arr):
    num = 0
    flag = False
    for i in range(len(arr)):
        #print(arr[i])
        if flag == False and arr[i] == 1:
            #print("hi1")
            flag = True
            cnt = 1
            continue
        
        if flag == True and i == len(arr) - 1 and arr[i] == 1:
            cnt += 1
            #print("hi2")
            flag = False
            #print(cnt)
            if cnt >= 4:
                num += 1
            continue
        
        if flag == True and arr[i] == 0:
            #print("hi3")
            flag = False
            #print(cnt)
            if cnt >= 4:
                num += 1
            continue
            
        if flag == True and arr[i] == 1:
            #print("hi4")
            cnt += 1
            continue
    return num


# In[7]:


def computerepeats(mat):
    ans = 0
    for i in range(len(mat) - 1):
        arr = []
        for j in range(i + 1):
            arr.append(mat[len(mat) - (j + 1), i - j])
        #print(arr)
        ans += check4ones(arr)
    
    return ans


# In[8]:


def computeinvrepeats(mat):
    ans = 0
    
    for k in range(len(mat)):
        arr = []
        arr.append(mat[k, k])
        i = k
        j = k
        while(i + 1 <= len(mat) - 1 and j - 1>= 0):
            i = i + 1
            j = j - 1
            arr.append(mat[i, j])
        #print(arr)
        ans += check4ones(arr)

    for k in range(len(mat) - 1):
        arr = []
        arr.append(mat[k + 1, k])
        i = k + 1
        j = k
        while(i + 1 <= len(mat) - 1 and j - 1>= 0):
            i = i + 1
            j = j - 1
            arr.append(mat[i, j])
        #print(arr)
        ans += check4ones(arr)
    return ans


# In[17]:
fi = open(sys.argv[2], 'r')
sequ = fi.read().splitlines()

fo = open(sys.argv[4], 'w')

for seq in sequ:

    #seq = "ATGTGTGTCATGCTACGGTCAGGGGTGCATGCTACGTCGTGTCATGTACTG"
    #seq = "ABCDABCDLABCD"


    # In[18]:


    mat = np.zeros((len(seq), len(seq)))
    img = np.zeros((len(seq), len(seq)))


    # In[19]:


    evalmat(seq, mat, img)


    # In[20]:


    # mat = [[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #        [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #        [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
    #        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
    #        [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
    #        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #        [0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
    #        [0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
    #        [0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
    #        [1., 0., 0., 0., 1., 0., 0., 0., 1., 0.]]
    # mat = np.array(mat)


    # In[21]:

    str1 = seq
    fo.write(seq+"\n")
    str2 = "Number of repeats is " + str(computerepeats(mat))
    fo.write(str2+"\n")

    # In[22]:

    str3 = "Number of inverse repeats is " + str(computeinvrepeats(mat))
    fo.write(str3+"\n")

fi.close()
fo.close()