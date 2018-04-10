from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from numpy import reshape, array, int8, zeros
import random


def make_case_tuples(digits):
    cases_tuples = []
    for i in range(len(digits.images)):
        picture_to_vector = array(reshape(digits.images[i], len(digits.images[0])*len(digits.images[0][0])), dtype=int8)
        cases_tuples.append((picture_to_vector, digits.target[i]))
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


if __name__ == "__main__":
    # Digits.images is the array of numpy np.arrays of 8x8 with innts in range [0, 16] for each pixel
    digits = load_digits()
    train, test = get_train_test_split(make_case_tuples(digits))
    one_hot_map = one_hot_encoding_of_sequence([i for i in range(10)])
    plot_images(digits.images[:5])
