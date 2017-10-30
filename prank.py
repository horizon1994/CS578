import numpy as np
import csv
#import random
import math
#import pandas as pd


'''L: number of iterations'''
'''k: number of labels'''
def prank(L, k, X, y):
    (n, d) = X.shape
    if n < k:
        raise Exception("number of samples should be at least number of labels")
    theta = np.zeros((d,1), dtype = float)
    b = np.zeros((k-1, 1), dtype = float)
    S = np.zeros((n, k-1), dtype = int)
    
    '''generate label array'''
    for l in range(k-1):
        b[l, 0] = l/2
    
    '''compute label matrix of X'''
    for t in range(n):
        for l in range(k-1):
            S[t, l] = -1 if y[t, 0] <= b[l, 0] else 1
    
    '''compute theta and b(thretholds for each classifier)'''
    for iter in range(L):
        for t in range(n):
            E = np.array([])
            for l in range(k-1):
                if S[t, l] * (np.dot(X[t, :], theta) - b[l ,0]) <= 0:
                    E = np.append(E, l)
            E = E.astype(int)
            if E.size:
                for i in range(E.size):
                    l = E[i]
                    theta += S[t, l] * np.array([X[t, :]]).T
                    b[l, 0] -= S[t, l]
    return {'theta':theta, 'b':b }
    