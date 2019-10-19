

import numpy as np
from numpy import genfromtxt
import numpy.linalg as la
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
sns.set()



data = genfromtxt('Data_Train.csv', delimiter=',', skip_header=1)
X_trn = data[:,1:]
Y_trn = data[:,-1:]

data = genfromtxt('Data_Test.csv', delimiter=',', skip_header=1)
X_tst = data[:,1:-1]
Y_tst = data[:,-1:]

pca_data = genfromtxt('PCA_Data_Train.csv', delimiter=',', skip_header=1)
X_trn_pca = pca_data[:,1:]
Y_trn_pca = pca_data[:,-1:]


def euclidian_distance(x1, x2):
    distance = np.mean((x1[:-1] - x2[:-1])**2)
    return np.sqrt(distance)


def find_neighbors(x1, X, Y, k):

    distances = []
    for example in X:
        distances.append(euclidian_distance(x1, example))

    distances = np.array(distances)
    idx = distances.argsort()[:k+1] #k+1 because smallest distance is point itself

    votes = []
    for neighbor in idx:
        votes.append(Y[neighbor][0])

    majority_vote = max(set(votes), key=votes.count)

    return majority_vote


def error(Y, predictions):
    return np.float64(np.sum(Y == predictions))/np.float64(len(Y))*100


def KNN(X, Y, Ks):

    errors = []
    for k in Ks:
        predictions = []
        for x in range(len(X)):
            prediction = find_neighbors(X[x], X, Y, k)
            predictions.append(prediction)
        e = error(Y.flatten(), predictions)
        errors.append(e)

    return errors


Ks = [1, 3, 5, 7, 9, 11, 13, 15, 17]

errors = KNN(X_trn, Y_trn, Ks)
errors_pca = KNN(X_trn_pca, Y_trn_pca, Ks)


plt.plot(Ks, errors, label='full feature space')
plt.plot(Ks, errors_pca, label='pca')
plt.xlabel('k value', fontsize=18)
plt.ylabel('classification accuracy (%)', fontsize=18)
plt.title('K-Nearest Neighbors', fontsize=20)
plt.legend()
plt.savefig('knn.png')
plt.show()
