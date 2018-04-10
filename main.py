from neural_network import NeuralNetwork
from recognize_digits import one_hot_map, get_digit_tuples, get_train_test_split
from datetime import datetime

if __name__ == "__main__":

    nn = NeuralNetwork(num_of_features=64, layers=[64, 32, 16, 10],
                       classes=one_hot_map.keys(), learning_rate=0.5)

    digits = get_digit_tuples(normalize=True)
    train_tuples, test_tuples = get_train_test_split(digits)

    print("starting training at", datetime.now())
    nn.train(cases=train_tuples, epochs=100, show_error_plot=True)
    print("finished training at", datetime.now())
    print(nn.evaluate(test_tuples))
