import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

def initialisation(X):
    W = np.array([[X],[X]])
    b = np.array([X])
    return (W, b)

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
#plt.show()
W = initialisation(X)[0]
b = initialisation(X)[1]

print(W.shape)
print(b.shape)


def model(X, W, b):
    Z = X * W + b
    A = 1 / (1 +  math.e-Z)
    return A

A = model(X, W, b)

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

print(log_loss(A, y))
