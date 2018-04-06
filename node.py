import numpy as np


class Node:

    def __init__(self, ancestors, children, operator_function):
        self.ancestors = np.array(ancestors)
        self.children = np.array(children)
        self.weights = np.ones(len(children))
        self.operator = operator_function
        self.activation_threshhold = 0.5
        # TODO: Initialize more stuff that needs to be here?

    def __activation_function(self, weight_sum):
        # TODO: Base this one some threshhold that is to be updated later on.
        return weight_sum > self.activation_threshhold

    def is_activated(self, inputs):
        weight_sum = np.sum(self.operator(input_node, weight) for weight in self.weights for input_node in inputs if input_node)
        return bool(self.__activation_function(weight_sum))

    def update_weights(self, weights):
        if len(weights) is not len(self.weights):
            raise ValueError("Length of weights is not the same as this node was initialized to.")
        for weight in weights:
            if type(weight) is not float:
                raise TypeError("Not all weights of the given weights were floats.")
        self.weights = weights
