import configparser
from train import *

def main():
    neurons, layers, momentums, epochs, learning_rate = read_properties()

    (train_data, train_label), (test_data, test_label), m, n = data(shuffle=True)

    layers = None
    accuracies = []
    for momentum in momentums:
        layers = [Layer((neurons, 784), ReLU, ReLU_deriv, momentum), Layer((neurons, 10), softmax, ReLU_deriv, momentum)]
        print(f"starting momentum {momentum}")
        accuracy = gradient_descent(train_data, train_label,layers, learning_rate, epochs, m)
        accuracies.append((momentum, accuracy))
        print(f"end momentum {momentum} with max accuracy {accuracy.max()}")
    compare_results(accuracies, f"momentum-analysis-{epochs}-epochs")

def read_properties():
    config = configparser.ConfigParser()

    config.read("properties.ini")
    neurons = int(config.get("main","neurons"))
    layers =  int(config.get("main","layers"))
    momentums = [float(num) for num in config.get("main","momentum").split(',')]
    epochs = int(config.get("main","epochs"))
    learning_rate = float(config.get("main","learning_rate"))

    return neurons, layers, momentums, epochs, learning_rate

if __name__ == "__main__":
    main()