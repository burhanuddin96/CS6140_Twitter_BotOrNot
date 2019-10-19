#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd


# In[4]:


#Get Full dimensional data from csv files
def get_normal_data():
    #Read data from file
    trn_data = pd.read_csv('Data_Train.csv');
    tst_data = pd.read_csv('Data_Test.csv');

    #Extracting output for dataset
    Y_trn = trn_data.bot_or_not.values.reshape(1101,1);
    Y_tst = tst_data.bot_or_not.values.reshape(276,1);

    #Replacing 0's with -1
    Y_trn = Y_trn.astype(int);
    np.putmask(Y_trn, Y_trn == 0, -1)

    Y_tst = Y_tst.astype(int);
    np.putmask(Y_tst, Y_tst == 0, -1)

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


# In[5]:


#Get PCA data from csv files
def get_PCA_data():
    #Read data from file
    trn_data = pd.read_csv('PCA_Data_Train.csv');
    tst_data = pd.read_csv('PCA_Data_Test.csv');

    #Extracting output for dataset
    Y_trn = trn_data.bot_or_not.values.reshape(1101,1);
    Y_tst = tst_data.bot_or_not.values.reshape(276,1);

    #Replacing 0's with -1
    Y_trn = Y_trn.astype(int);
    np.putmask(Y_trn, Y_trn == 0, -1)

    Y_tst = Y_tst.astype(int);
    np.putmask(Y_tst, Y_tst == 0, -1)

    #Dropping unnecessary columns
    to_drop = ['bot_or_not']
    trn_data.drop(to_drop, inplace = True, axis = 1)
    trn_data.set_index('id', inplace = True);
    tst_data.drop(to_drop, inplace = True, axis = 1)
    tst_data.set_index('id', inplace = True);

    X_trn = trn_data.values;
    X_tst = tst_data.values;

    return X_trn, Y_trn, X_tst, Y_tst;

#X_trn, Y_trn, X_tst, Y_tst = get_PCA_data();


# In[6]:


#Linear Kernel
def linear_kernel(X, Y):
    return np.dot(X, Y);


# In[7]:


#Select random j not equal to i
def random_j(i, rows):
    j = i;
    while(j == i):
        j = int(np.random.uniform(0, rows));

    return j;


# In[8]:


#Calculate weight vector
def calculate_W(X, Y, alpha):
    W = np.zeros((X.shape[1], 1));
    for i in range(X.shape[0]):
        W += np.multiply(Y[i] * alpha[i], X[i].T).T

    return W


# In[13]:


def classify(X, w, b):
    Y = np.dot(X, w) + b;

    np.putmask(Y, Y < 0, -1)
    np.putmask(Y, Y >= 0, 1)
    return Y


# In[14]:


def error_percent(Y, Y_hat):
    assert Y_hat.shape == Y.shape , "Not equal"
    count = 0;
    for i in range(Y.shape[0]):
        if(Y[i] == Y_hat[i]):
            continue
        else:
            count+=1

    return (count / Y.shape[0]) * 100;


# In[10]:


#SMO implementation for SVM
def SVM(X, Y, MAX_ITERATIONS, C, tolerance, threshold):
    #Initialize alphas and b
    alpha = np.mat(np.zeros((X.shape[0], 1)));
    b = np.mat([[0]]);

    #Initialize number of iterations
    iterations = 0;

    while(iterations < MAX_ITERATIONS):
        number_changed_alphas = 0;
        for i in range(X.shape[0]):
            #Calculate Error Ei
            Ei = np.multiply(Y, alpha).T * linear_kernel(X, X[i].reshape(X.shape[1], 1)) + b - Y[i];

            #Check if alphas violate KKT conditions
            if(alpha[i] > 0 and np.abs(Ei) < tolerance) or (alpha[i] < C and np.abs(Ei) > tolerance):
                j = random_j(i, X.shape[0]);
                #Calculate Error Ej
                Ej = np.multiply(Y, alpha).T * linear_kernel(X, X[j].reshape(X.shape[1], 1)) + b - Y[j];

                #Store the old alphas
                old_alphaI = alpha[i].copy();
                old_alphaJ = alpha[j].copy();

                #Calculate L and H
                if(Y[i] != Y[j]):
                    L = max(0, alpha[j] - alpha[i]);
                    H = min(C, alpha[j] - alpha[i] + C);
                else:
                    L = max(0, alpha[i] + alpha[j] - C);
                    H = min(C, alpha[i] + alpha[j]);

                #Continue to next i if L==H
                if(L == H):
                    continue;

                #Calculate ETA
                eta = 2.0 * linear_kernel(X[i].reshape(X.shape[1], 1).T, X[j].reshape(X.shape[1], 1)) - linear_kernel(X[i].reshape(X.shape[1], 1).T, X[i].reshape(X.shape[1], 1)) - linear_kernel(X[j].reshape(X.shape[1], 1).T, X[j].reshape(X.shape[1], 1));

                #Continue to next i if eta>=0
                if(eta >= 0):
                    continue;

                #Compute alpha[j]
                alpha[j] -= Y[j] * (Ei - Ej) / eta;

                #Clip alpha[j]
                if(alpha[j] < L):
                    alpha[j] = L;
                elif(alpha[j] > H):
                    alpha[j] = H;

                if(abs(alpha[j] - old_alphaJ) < threshold):
                    continue;

                #Optimize alpha[i]
                alpha[i] += Y[j] * Y[i] * (old_alphaJ - alpha[j]);

                #Calculate b1 & b2
                b1 = b - Ei - Y[i] * (alpha[i] - old_alphaI) * linear_kernel(X[i].reshape(X.shape[1], 1).T, X[i].reshape(X.shape[1], 1)) - Y[j] * (alpha[j] - old_alphaJ) * linear_kernel(X[i].reshape(X.shape[1], 1).T, X[j].reshape(X.shape[1], 1));

                b2 = b - Ej - Y[i] * (alpha[i] - old_alphaI) * linear_kernel(X[i].reshape(X.shape[1], 1).T, X[j].reshape(X.shape[1], 1)) - Y[j] * (alpha[j] - old_alphaJ) * linear_kernel(X[j].reshape(X.shape[1], 1).T, X[j].reshape(X.shape[1], 1));

                #Calculate b
                if(alpha[i] > 0 and alpha[i] < C):
                    b = b1;
                elif(alpha[j] > 0 and alpha[j] < C):
                    b = b2;
                else:
                    b = (b1 + b2) / 2.0;

                number_changed_alphas += 1;

        if(number_changed_alphas == 0):
            iterations += 1;
        else:
            iterations = 0;

    W = calculate_W(X, Y, alpha);
    print(alpha)
    return W, b;


# In[42]:


#Calculate aggregate errors to get optimum C
C = []
trn_errors = []
tst_errors = []
min = 99
minC = -1

for i in range(1, 100):

    SVM(X_trn, Y_trn, 50, i, 0.001, 0.00001);

    Yh_trn = classify(X_trn, W, b);
    Yh_tst = classify(X_tst, W, b);

    trn_error = error_percent(Y_trn.ravel(), Yh_trn)
    tst_error = error_percent(Y_tst.ravel(), Yh_tst)

    if((trn_error + tst_error) / 2 < min):
        min = (trn_error + tst_error) / 2;
        minC = i


    print("Training Error: ", trn_error)
    print("Testing Error: ", tst_error)

    C.append(i)
    trn_errors.append(trn_error)
    tst_errors.append(tst_error)

    print('Minimum Error C: ', minC)
