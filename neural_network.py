import numpy as np
from util import squared_loss, squared_loss_derivated


class NeuralNetwork:

    def __init__(self, features, layers, classes, alpha, node_operator):
        if type(features) is not list or type(classes) is not list or (layers and type(layers) is not list):
            raise TypeError("One of the inputs is not a list.")
        if type(alpha) is not float:
            raise TypeError("Alpha must be a float between 0.0 and 1.0")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be a float between 0.0 and 1.0")
        self.alpha = alpha
        self.features = np.array(features)
        self.layers = np.array(np.array(layer) for layer in layers)
        self.output_layer = self.layers[-1]
        self.classes = np.array(classes)
        self.node_operator = node_operator
        # TODO: Initialize the weights and make the connected graph that is the network in some way

    def train(self, cases, epochs):
        last_layer_index = len(self.layers) - 1
        outputs = [0 for _ in self.layers]
        error_props = [0 for _ in self.layers]

        for _ in range(epochs):
            for case in cases:

                # Forward stepping through the network
                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    output = layer.forward_step(np.array(case[0])) if i == 0 else layer.forward_step(outputs[i-1])
                    # TODO: Add this bias-append to forward_step
                    outputs[i] = np.append(output, [1])

                # Backpropagation to update weights
                for i in range(last_layer_index, -1, -1):
                    grad_loss = squared_loss_derivated(np.array(case[1]), outputs[i]) if i == last_layer_index else self.layers[i+1].get_weights()
                    input_vector = outputs[i-1] if i > 0 else np.array(case[0])
                    prev_error = error_props[i+1] if i < last_layer_index else np.array(layer.get_num_nodes())
                    error_prop = layer.update_weights(input_vector, grad_loss, self.alpha, prev_error)
                    error_props[i] = error_prop



