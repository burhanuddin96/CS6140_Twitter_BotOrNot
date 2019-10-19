#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd


# In[3]:


#Read data from file
df = pd.read_csv('project_data.csv');

#Dropping unnecessary columns
to_drop = ['Unnamed: 0', 'bot_or_not']
df.drop(to_drop, inplace = True, axis = 1)
df.set_index('id', inplace = True);


# In[4]:


#Converting data frame to ndarray
X = df.values
X = np.transpose(X)

assert X.shape == (13, 1377), "The matrix dimensions are not (13, 1377)."


# In[5]:


#Calculate mean vector for X
mean_vector = []
for i in range(13):
    mean_vector.append([np.mean(X[i,:])])
mean_vector = np.asarray(mean_vector, dtype = np.float32)
X = X - mean_vector;

print("Mean Vector: \n", mean_vector)


# In[6]:


#Computing Covariance Matrix of X
X_cov = np.cov(X)
print("Covariance Matrix: \n", X_cov);


# In[7]:


#Computing eigenvalues and eigenvectors
eig_val, eig_vec = np.linalg.eig(X_cov);


# In[8]:


#Rearranging pairs of eigenvalues in eigenvectors in 
#descending order to get top d values
eig_vv = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_vv.sort(key = lambda x: x[0], reverse = True)


# In[11]:


plt.plot(eig_val*10)
plt.show();


# In[32]:


#Computing weights to get low dimensional points
d = 3 #Change it according to dimension space required

W = eig_vv[0][1].reshape(X.shape[0],1);

for i in range (1,d):
    W = np.hstack((W, eig_vv[i][1].reshape(X.shape[0],1)))

print('Weights :\n', W)


# In[33]:


#Computing final low dimensional representation of X
Z = np.dot(np.transpose(W), X);
print(Z.shape);


# In[34]:


np.savetxt("PCA-Data.csv", (np.transpose(Z)), delimiter=",", header="X1,X2,X3");

