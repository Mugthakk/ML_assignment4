__author__ = 'Ivar A. Fauske'
# This Python file uses the following encoding: utf-8

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivated(x):
    return sigmoid(x)*(1-sigmoid(x))

def squared_loss(y_t, y):
    return np.power(y_t-y, 2)/2

def squared_loss_derivated(y_t, y):
    return y - y_t