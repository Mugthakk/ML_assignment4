import numpy as np
from util import error_plot, squared_loss_derivated,squared_loss
from layer import Layer


class NeuralNetwork:

    def __init__(self, num_of_features, layers, classes, learning_rate):
        self.lr = learning_rate
        self.num_of_features = num_of_features
        self.layers = []
        inn = num_of_features
        for i in range(len(layers)):
            self.layers.append(Layer(inn+1, layers[i], i, i==len(layers)-1))
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

                self.forward_pass(case,outputs)

                # Backpropagation to update weights
                for i in range(last_layer_index, -1, -1):
                    grad_loss = squared_loss_derivated(np.array(case[1]), outputs[i]) if i == last_layer_index else self.layers[i+1].get_weights()[:-1]
                    input_vector = outputs[i-1] if i > 0 else np.array(case[0])
                    input_vector = np.append(input_vector, [1])
                    prev_error = error_props[i+1] if i < last_layer_index else np.ones(self.layers[i].get_num_nodes())
                    error_prop = self.layers[i].update_weights(input_vector, grad_loss, self.lr, prev_error)
                    error_props[i] = error_prop
                if show_error_plot:
                    epoch_score += squared_loss(case[1],outputs[-1])
            plot_data.append(epoch_score)
        if show_error_plot:
            #print(plot_data[:3])
            error_plot(plot_data)

    def forward_pass(self, case, outputs):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            input_vector = np.array(case[0]) if i == 0 else outputs[i-1]
            #add bias
            input_vector = np.append(input_vector,[1])
            output = layer.forward_step(input_vector)
            outputs[i] = output

    def evaluate(self,samples):
        for i in range(len(samples)):
            outputs = [0]*len(self.layers)
            self.forward_pass(np.array(samples[i]),outputs)
            print("sample:", samples[i][0])
            result = 1 if outputs[-1] > 0.5 else 0
            print("Result:", outputs[-1], "is", "correct" if result == samples[i][1] else "wrong")
            print()

        #TODO: make function to test the network and calculate accuracy

sanderErNoob = NeuralNetwork(2,[2,1],[0,1], 0.25)
t = [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],0]]
sanderErNoob.train(t,5000,True)
sanderErNoob.evaluate(t)


