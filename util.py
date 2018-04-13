# This Python file uses the following encoding: utf-8

import numpy as np
from matplotlib import pyplot

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivated(x):
    return sigmoid(x)*(1-sigmoid(x))

def squared_loss(y_t, y):
    return np.power(y_t-y, 2)/2

def squared_loss_derivated(y_t, y):
    return y - y_t

def error_plot(data):
    pyplot.plot(data)
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Sum error over all training samples in one epoch")
    pyplot.show()

def get_dropout_vector(size, dropout_rate):
    do_vec = (np.random.rand(size)>dropout_rate).astype(int)
    if np.max(do_vec) < 1:
        return get_dropout_vector(size,dropout_rate)
    do_vec = do_vec*size/np.sum(do_vec)
    return do_vec

def make_confusion_matrix(guesses, corrects, shape):
    cm = np.array([[0 for i in range(shape[1])] for j in range(shape[0])])
    for i in range(len(guesses)):
        cm[guesses[i],corrects[i]] += 1
    return cm

def plot_confusion_matrix(cm):
    pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pyplot.text(j, i, cm[i,j])
    pyplot.xlabel("Correct")
    pyplot.ylabel("Networks classification")
    pyplot.show()