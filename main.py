from neural_network import NeuralNetwork
from recognize_digits import one_hot_map, get_digit_tuples, get_train_test_split

if __name__ == "__main__":


    digits = get_digit_tuples(normalize=True)

    for i in range(10):
        nn = NeuralNetwork(num_of_features=64, layers=[64, 32, 16, 10],
                           classes=one_hot_map.keys(), learning_rate=0.1,dropout_rate=0.3)
        train_tuples, test_tuples = get_train_test_split(digits)
        nn.train(cases=train_tuples, epochs=50)
        print(nn.evaluate(test_tuples))
