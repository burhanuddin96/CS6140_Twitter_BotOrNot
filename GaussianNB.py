#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd


# In[119]:


#Get Full dimensional data from csv files
def get_normal_data():
    #Read data from file
    trn_data = pd.read_csv('Data_Train.csv');
    tst_data = pd.read_csv('Data_Test.csv');

    #Extracting output for dataset
    Y_trn = trn_data.bot_or_not.values.reshape(1101,1);
    Y_tst = tst_data.bot_or_not.values.reshape(276,1);

    #Dropping unnecessary columns
    to_drop = ['bot_or_not']
    trn_data.drop(to_drop, inplace = True, axis = 1)
    trn_data.set_index('id', inplace = True);
    tst_data.drop(to_drop, inplace = True, axis = 1)
    tst_data.set_index('id', inplace = True);

    X_trn = trn_data.values;
    X_tst = tst_data.values;

    return X_trn, Y_trn, X_tst, Y_tst;

X_trn, Y_trn, X_tst, Y_tst = get_normal_data();


# In[120]:


#Get PCA data from csv files
def get_PCA_data():
    #Read data from file
    trn_data = pd.read_csv('PCA_Data_Train.csv');
    tst_data = pd.read_csv('PCA_Data_Test.csv');

    #Extracting output for dataset
    Y_trn = trn_data.bot_or_not.values.reshape(1101,1);
    Y_tst = tst_data.bot_or_not.values.reshape(276,1);

    #Dropping unnecessary columns
    to_drop = ['bot_or_not']
    trn_data.drop(to_drop, inplace = True, axis = 1)
    trn_data.set_index('id', inplace = True);
    tst_data.drop(to_drop, inplace = True, axis = 1)
    tst_data.set_index('id', inplace = True);

    X_trn = trn_data.values;
    X_tst = tst_data.values;

    return X_trn, Y_trn, X_tst, Y_tst;

X_trn, Y_trn, X_tst, Y_tst = get_PCA_data();


# In[121]:


def generateDataFrame(X, Y):
    dataX = pd.DataFrame(data=X[0:, 0:])
    dataY = pd.DataFrame(data=Y[0:, 0:], columns=['C'])
    frames = [dataX, dataY]
    return pd.concat(frames, axis=1);

trn_data = generateDataFrame(X_trn, Y_trn)
tst_data = generateDataFrame(X_tst, Y_tst)


# In[128]:


#Calculating priors for Y
def calc_priors(data):
    class_count = data['C'].value_counts().to_dict()
    priors = {(clas, count / data.shape[0]) for (clas, count) in class_count.items()}
    return priors;

priors = calc_priors(trn_data)


# In[123]:


#Calculating mean and variance
def calc_mean_variance(data):
    mean_var = {}
    for c in data['C'].unique():
        class_set = data[(data['C'] == c)]
        mv = {}
        for i in range(0, data.shape[1]-1):
            mv[i] = []
            mv[i].append(class_set[i].mean())
            mv[i].append(math.pow(class_set[i].std(), 2))
            mean_var[c] = mv
    return mean_var

mean_var = calc_mean_variance(trn_data)


# In[124]:


#Calculate Gaussian PDF
def calc_gaussian_pdf(xi, mean, var):
    exp = math.exp(-1 * (math.pow(xi - mean, 2) / (2 * var)))
    pdf = (1 / (math.sqrt(2 * math.pi * var))) * exp
    return pdf


# In[125]:


#Returns Predictions on Data
def predict(data, priors, mean_var):
    Y_hat = {}
    for index, row in data.iterrows():
        res = {}
        for C, val in priors:
            p = 0
            for i in range(0, data.shape[1]-1):
                prob = calc_gaussian_pdf(row[i], mean_var[C][i][0], mean_var[C][i][1])
                if prob > 0:
                    p += math.log(prob)
            res[C] = math.log(val) + p
        Y_hat[index] = max([key for key in res.keys() if res[key] == res[max(res, key=res.get)]])
    return Y_hat

Y_trn_pred = predict(trn_data, priors, mean_var)
Y_tst_pred = predict(tst_data, priors, mean_var)


# In[126]:


def error(data, Y_hat):
    err = 0
    for index, row in data.iterrows():
        if row['C'] != Y_hat[index]:
            err += 1
    return (err / len(data)) * 100


# In[127]:


print("Training Error: ", error(trn_data, Y_trn_pred))
print("Testing Error: ", error(tst_data, Y_tst_pred))
