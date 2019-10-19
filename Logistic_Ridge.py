
# coding: utf-8

# In[3]:


import numpy as np
from numpy import genfromtxt
import scipy.io
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[26]:


data = genfromtxt('Data_Train.csv', delimiter=',', skip_header=1)
X_trn = data[:,1:-1]
Y_trn = data[:,-1:]

data = genfromtxt('Data_Test.csv', delimiter=',', skip_header=1)
X_tst = data[:,1:-1]
Y_tst = data[:,-1:]

pca_data = genfromtxt('PCA_Data_Train.csv', delimiter=',', skip_header=1)
X_trn_pca = pca_data[:,1:-1]
Y_trn_pca = pca_data[:,-1:]

pca_data = genfromtxt('PCA_Data_Test.csv', delimiter=',', skip_header=1)
X_tst_pca = pca_data[:,1:-1]
Y_tst_pca = pca_data[:,-1:]



# In[6]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()


def gradient_descent(X, Y, lrn_rate = 0.001, iterations=100000):
    theta = np.random.random_sample(X[0].shape) # initialize theta

    for i in range(iterations):
        z = np.dot(X, theta)
        a = sigmoid(z)
        gradient = np.dot(X.T, (a.ravel() - Y.ravel())) / Y.size
        theta -= lrn_rate * gradient

    return theta


# In[7]:


def predict(X, Y, theta, threshold):
    predict_prob = sigmoid(np.dot(X, theta))
    prediction = (predict_prob >= threshold)

    return np.mean(prediction == Y)*100.


# In[9]:


theta = gradient_descent(X_trn, Y_trn)

trn_accuracy = predict(X_trn, Y_trn, theta, 0.5)
tst_accuracy = predict(X_tst, Y_tst, theta, 0.5)


theta_pca = gradient_descent(X_trn_pca, Y_trn_pca)

trn_accuracy_pca = predict(X_trn_pca, Y_trn_pca, theta_pca, 0.5)
tst_accuracy_pca = predict(X_tst_pca, Y_tst_pca, theta_pca, 0.5)

print 'Regular Logistic Regression'
print 'full dimensionality'
print 'Training accuracy: ', np.round(trn_accuracy, 2)
print 'Testing accuracy: ', np.round(tst_accuracy, 2)
print
print 'PCA'
print 'Training accuracy: ', np.round(trn_accuracy_pca, 2)
print 'Testing accuracy: ', np.round(tst_accuracy_pca, 2)



# In[13]:


def gradient_descent_ridge(X, Y, C, iterations=100000):
    theta = np.random.random_sample(X[0].shape) # initialize theta
    lrn_rate = 0.001

    for i in range(iterations):
        z = np.dot(X, theta)
        a = sigmoid(z)
        gradient = np.dot(X.T, (a.ravel() - Y.ravel())) / Y.size
        theta -= lrn_rate * gradient + C * theta


    return theta


# In[14]:


def k_groups(X, Y, k, group):
        size = len(X)/k

        X_k = X[group*size:(group+1)*size]
        Y_k = Y[group*size:(group+1)*size]

        X_rest = np.concatenate((X[0:group*size], X[(group+1)*size:]), axis=0)
        Y_rest = np.concatenate((Y[0:group*size], Y[(group+1)*size:]), axis=0)

        return X_k, Y_k, X_rest, Y_rest


# In[44]:


def k_fold_cross_val_GD(X, Y, k, C):

    thetas, accuracy = [], []
    for group in range(k):
        X_k, Y_k, X_rest, Y_rest = k_groups(X, Y, k, group)
        theta = gradient_descent_ridge(X_rest, Y_rest, C) # train on train set
        k_acc = predict(X_k, Y_k, theta.T, 0.5)  # evaluate on holdout set
        thetas.append(theta)
        accuracy.append(k_acc) # record evaluation score
    return np.mean(thetas, axis=0) , np.mean(accuracy)


# In[45]:


Ks = [10]
Cs = [0.000001, 0.00001, 0.0001, 0.001, 0.1, 0.5, 1, 1.5, 10]

print "Gradient descent with Ridge Regression"
for k in Ks:

    theta_list, accuracy_list, C_list = [], [], []
    print 'k = ', k
    for c in Cs:
        theta_l, accuracy_l = k_fold_cross_val_GD(X_trn_pca, Y_trn_pca, k, c)

        theta_list.append(theta_l)
        accuracy_list.append(accuracy_l)
        C_list.append(c)

    # find best regularization lambda
    opt = accuracy_list.index(max(accuracy_list))
    print 'For k = %s, the optimal lambda value is %s' %(k, Cs[opt])

accuracy_list_pca = accuracy_list
theta_list_pca = theta_list


# In[46]:


Ks = [10]
Cs = [0.000001, 0.00001, 0.0001, 0.001, 0.1, 0.5, 1, 1.5, 10]

print "Gradient descent with Ridge Regression"
for k in Ks:

    theta_list, accuracy_list, C_list = [], [], []
    print 'k = ', k
    for c in Cs:
        theta_l, accuracy_l = k_fold_cross_val_GD(X_trn, Y_trn, k, c)

        theta_list.append(theta_l)
        accuracy_list.append(accuracy_l)
        C_list.append(c)

    # find best regularization lambda
    opt = accuracy_list.index(max(accuracy_list))
    print 'For k = %s, the optimal lambda value is %s' %(k, Cs[opt])

accuracy_list_regular = accuracy_list
theta_list_regular = theta_list


# In[56]:


plt.plot(np.log(Cs), accuracy_list_regular, label = 'full feature space', c='r')
plt.plot(np.log(Cs), accuracy_list_pca, label = 'pca', c='c')
plt.title('Ridge Regression: Cross validation accuracy', fontsize=18)
plt.ylabel('accuracy', fontsize=15)
plt.xlabel('logC value', fontsize=15)
plt.legend()
plt.savefig('ridge_cross_val_k10.png')
plt.show()


# In[55]:


acc_trns, acc_tsts, acc_trns_pca, acc_tsts_pca = [],[],[],[]

for theta in theta_list_pca:
    acc_trn = predict(X_trn_pca, Y_trn_pca, np.array([theta]).T, 0.5)
    acc_tst = predict(X_tst_pca, Y_tst_pca, np.array([theta]).T, 0.5)

    acc_trns.append(acc_trn)
    acc_tsts.append(acc_tst)


for theta in theta_list_regular:
    acc_trn = predict(X_trn, Y_trn, np.array([theta]).T, 0.5)
    acc_tst = predict(X_tst, Y_tst, np.array([theta]).T, 0.5)

    acc_trns_pca.append(acc_trn)
    acc_tsts_pca.append(acc_tst)


plt.plot(np.log(Cs), acc_trns, label = 'regular train', c='r')
plt.plot(np.log(Cs), acc_tsts, label = 'regular test', c='r', linestyle=":")
plt.plot(np.log(Cs), acc_trns_pca, label = 'PCA train', c='c')
plt.plot(np.log(Cs), acc_tsts_pca, label = 'PCA test', linestyle=":")
plt.title('Gradient Descent with Ridge Regression', fontsize=18)
plt.ylabel('accuracy', fontsize=15)
plt.xlabel('logC value', fontsize=15)
plt.legend()
plt.savefig('ridge_results_k10.png')
plt.show()


# In[58]:


opt = accuracy_list.index(max(accuracy_list))
acc_trn = predict(X_trn, Y_trn, np.array([theta_list[opt]]).T, 0.5)
acc_tst = predict(X_tst, Y_tst, np.array([theta_list[opt]]).T, 0.5)

print 'Full feature space best theta: train and test accuracy'
print acc_trn, acc_tst
print

opt_pca = accuracy_list_pca.index(max(accuracy_list_pca))
acc_trn_pca = predict(X_trn_pca, Y_trn_pca, np.array([theta_list_pca[opt_pca]]).T, 0.5)
acc_tst_pca = predict(X_tst_pca, Y_tst_pca, np.array([theta_list_pca[opt_pca]]).T, 0.5)

print 'PCA best theta: train and test accuracy'
print acc_trn_pca, acc_tst_pca
