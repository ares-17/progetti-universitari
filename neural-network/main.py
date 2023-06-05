import configparser
from train import *

def main():
    config = configparser.ConfigParser()
    config.read("properties.ini")
    neurons = int(config.get("main","neurons"))
    layers =  int(config.get("main","layers"))
    momentums = [float(num) for num in config.get("main","momentum").split(',')]
    epochs = int(config.get("main","epochs"))
    learning_rate = float(config.get("main","learning_rate"))

    accuracies = []
    for momentum in momentums:
        print(f"starting momentum {momentum}")
        accuracy = gradient_descent(train_data, train_label, learning_rate, epochs, momentum)
        accuracies.append((momentum, accuracy))
        print(f"end momentum {momentum} with max accuracy {accuracy.max()}")
    compare_results(accuracies, f"momentum-analysis-{epochs}-epochs")

if __name__ == "__main__":
    main()