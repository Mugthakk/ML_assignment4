import numpy as np
from util import sigmoid,sigmoid_derivated,squared_loss, squared_loss_derivated


class Layer:

    def __init__(self, in_size, num_of_nodes, level, is_output_layer=False, activation_function=sigmoid):
        self.level = level
        self.is_output_layer = is_output_layer
        self.weights = 2*np.random.rand(in_size, num_of_nodes)-1
        self.activation_threshhold = 0.5
        #typecasting in case of datatype issues later
        self.activation_function = np.vectorize(lambda x: int(activation_function(x)>self.activation_threshhold))

    def forward_step(self, input_vector):
        return self.activation_function(np.dot(input_vector,self.weights))

    def update_weights(self, input_vctor, grad_loss, lr, error_prop=np.array([1])):
        z = np.dot(input_vctor,self.weights)
        grad_a = np.vectorize(sigmoid_derivated)(z)
        #uses is_output_layer to decide error function to be used
        error = grad_loss*grad_a if self.is_output_layer else np.dot(error_prop,grad_loss[np.newaxis].T)*grad_a
        #np.newaxis is hack to transpose 1-D arrays
        self.weights = self.weights - lr*np.dot(input_vctor[np.newaxis].T,error[np.newaxis])
        #returns so we can propagate error
        return error


#used as playground
def main():
    a = Layer(2,1,0)
    b = Layer(2,1,1)
    #print(np.array([1,1]).T.shape)
    o1 = a.forward_step(np.array([1,0]))
    grad_loss = squared_loss_derivated(np.array([1]),o1)
    print("out 1:",o1)
    print("grad loss:",grad_loss)
    a.update_weights(np.array([1,0]),grad_loss,0.1)
    #o2 = b.forward_step(o1)
    #print(o2)
    #l = squared_loss(np.array([1]),o2)
    #print(l)

#sample 1-layer network
def main2(train, epochs):
    #create layer with bias node
    a = Layer(len(train[0][0])+1,1,0,True)
    #add biases
    for sample in train:
        sample[0].append(1)
    #do training
    for i in range(epochs):
        for sample in train:
            out = a.forward_step(np.array(sample[0]))
            grad_loss = squared_loss_derivated(np.array(sample[1]),out)
            a.update_weights(np.array(sample[0]),grad_loss,0.1)
    #print performance after training
    for sample in train:
        print("sample:",sample[0][:-1])
        result = a.forward_step(np.array(sample[0]))
        print("Result:",result, "is","correct" if result==sample[1] else "wrong")
        print()

#training samples representing the AND function
t = [[[0,0],0],[[1,0],0],[[0,1],0],[[1,1],1]]
main2(t,100)