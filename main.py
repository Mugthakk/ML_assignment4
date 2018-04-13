from neural_network import NeuralNetwork
from recognize_digits import one_hot_map, get_digit_tuples, get_train_test_split

if __name__ == "__main__":


    digits = get_digit_tuples(normalize=True)

    for i in range(10):
        nn = NeuralNetwork(num_of_features=64, layers=[32, 10],
                           classes=one_hot_map.keys(), learning_rate=0.1,dropout_rate=0.5)
        train_tuples, test_tuples = get_train_test_split(digits)
        nn.train(cases=train_tuples, epochs=50,show_error_plot=False)
        print(nn.evaluate(test_tuples))
        nn.confusion_matrix(test_tuples)

'''
if __name__ == "__main__":

    for _ in range(10):
        nn = NeuralNetwork(2,[2,1],[0,1], 0.2)
        t = [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],0]]
        nn.train(t,5000,False)
        print(nn.eval_xor(t))
'''