import numpy as np
from util import error_plot, squared_loss_derivated, squared_loss, make_confusion_matrix,plot_confusion_matrix
from layer import Layer
from recognize_digits import decode_one_hot


class NeuralNetwork:

    def __init__(self, num_of_features, layers, classes, learning_rate, dropout_rate=0):
        self.lr = learning_rate
        self.num_of_features = num_of_features
        self.layers = []
        inn = num_of_features
        for i in range(len(layers)):
            self.layers.append(Layer(inn+1, layers[i], i, i==len(layers)-1,dropout_rate))
            inn = layers[i]
        self.classes = np.array(classes)

    def train(self, cases, epochs, show_error_plot=False):
        plot_data = []
        last_layer_index = len(self.layers) - 1
        outputs = [0 for _ in self.layers]
        error_props = [0 for _ in self.layers]

        for _ in range(epochs):
            epoch_score = 0
            for case in cases:

                #no need to asign to variable, the results are stored in outputs
                self.forward_pass(case, outputs, True)

                # Backpropagation to update weights
                for i in range(last_layer_index, -1, -1):
                    grad_loss = squared_loss_derivated(np.array(case[1]), outputs[i]) if i == last_layer_index else self.layers[i+1].get_weights()[:-1]
                    input_vector = outputs[i-1] if i > 0 else np.array(case[0])
                    input_vector = np.append(input_vector, [1])
                    prev_error = error_props[i+1] if i < last_layer_index else np.ones(self.layers[i].get_num_nodes())
                    error_prop = self.layers[i].update_weights(input_vector, grad_loss, self.lr, prev_error)
                    error_props[i] = error_prop
                if show_error_plot:
                    epoch_score += np.sum(squared_loss(case[1], outputs[-1]))
            plot_data.append(epoch_score)
        if show_error_plot:
            error_plot(plot_data)

    #a bit weird as it manipulates the list outputs which it takes as a parameter
    def forward_pass(self, case, outputs, training=False):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            input_vector = np.array(case[0]) if i == 0 else outputs[i-1]
            #add bias
            input_vector = np.append(input_vector, [1])
            output = layer.forward_step(input_vector, training)
            outputs[i] = output

    def evaluate(self, samples):
        correct_samples = 0
        for i in range(len(samples)):
            outputs = [0]*len(self.layers)
            self.forward_pass(np.array(samples[i]), outputs)
            if np.argmax(outputs[-1]) == decode_one_hot(samples[i][1]):
                correct_samples += 1
        return correct_samples/len(samples)

    def confusion_matrix(self, samples):
        guesses = []
        for i in range(len(samples)):
            outputs = [0]*len(self.layers)
            self.forward_pass(np.array(samples[i]), outputs)
            guesses.append(np.argmax(outputs[-1]))
        cm = make_confusion_matrix(guesses, [decode_one_hot(sample[1]) for sample in samples],(10,10))
        plot_confusion_matrix(cm)


    def eval_xor(self,samples):
        correct = 0
        for i in range(len(samples)):
            outputs = [0]*len(self.layers)
            self.forward_pass(np.array(samples[i]), outputs)
            value = int(outputs[-1]>0.5)
            correct += value == samples[i][1]
        return correct/len(samples)



