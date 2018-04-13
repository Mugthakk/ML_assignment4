from sklearn.datasets import load_digits
from scipy import stats
import matplotlib.pyplot as plt
from numpy import reshape, array, int8, zeros, float16, argmax
import random


def get_digit_tuples(normalize=False):
    digits = load_digits()
    cases_tuples = []

    for i in range(len(digits.images)):
        if normalize:
            picture_to_vector = array(reshape(digits.images[i], len(digits.images[0]) * len(digits.images[0][0])),
                                      dtype=float16)
            picture_to_vector = stats.zscore(picture_to_vector)
        else:
            picture_to_vector = array(reshape(digits.images[i], len(digits.images[0])*len(digits.images[0][0])), dtype=int8)
        cases_tuples.append((picture_to_vector, one_hot_map[digits.target[i]]))
    return cases_tuples


def one_hot_encoding_of_sequence(classes):
    one_hot_map = {}
    num_classes = len(classes)
    for i in range(num_classes):
        vector = zeros(num_classes, dtype=int8)
        vector[i] = 1
        one_hot_map[classes[i]] = vector
    return one_hot_map


def get_train_test_split(labeled_list_of_tuples, train_percentage=0.8, shuffle=True):
    if shuffle:
        random.shuffle(labeled_list_of_tuples)
    return labeled_list_of_tuples[:int(len(labeled_list_of_tuples)*train_percentage)], labeled_list_of_tuples[int(len(labeled_list_of_tuples)*train_percentage):]


def plot_images(images_to_plot):
    plt.gray()
    for image in images_to_plot:
        plt.matshow(image)
    plt.show()

def encode_one_hot(num):
    return [int(x==num) for x in range(10)]

def decode_one_hot(vec):
    return argmax(vec)

one_hot_map = one_hot_encoding_of_sequence([i for i in range(10)])