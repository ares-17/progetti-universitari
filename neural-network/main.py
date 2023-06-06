import configparser
from train import *
import pandas as pd

def main():
    neurons, layers, momentums, epochs, learning_rate = read_properties()

    (train_data, train_label), (test_data, test_label), rows_dataset = data(shuffle=True)

    layers = None
    accuracies = []
    for momentum in momentums:
        layers = [Layer((neurons, 784), ReLU, ReLU_deriv, momentum), Layer((neurons, 10), softmax, ReLU_deriv, momentum)]
        print(f"starting momentum {momentum}")
        accuracy = gradient_descent(train_data, train_label,layers, learning_rate, epochs, rows_dataset)
        accuracies.append((momentum, accuracy))
        test_accuracy = make_predictions(test_data, layers)
        print(f"end momentum {momentum} with accuracy on train: {accuracy.max()}, on test: {get_accuracy(test_accuracy, test_label)}")

    compare_results(accuracies, f"momentum-analysis-{epochs}-epochs")


def make_predictions(X, layers):
    """
    Execute forward propagation on network and gets prediction's class from result's array
    """
    forward_prop(X, layers)
    predictions = np.argmax(layers[-1].A, 0)
    return predictions

def test_prediction(index, layers, train_data, train_label):
    """
    Gets the index of the image in the train_data array, prints the expected class and the label class.
    Next, visualize the image with matplotlib
    """
    current_image = train_data[:, index, None]
    prediction = make_predictions(train_data[:, index, None], layers)
    label = train_label[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def read_properties():
    """
    Read some configs from properties file.
    Does not handle any exceptions
    """
    config = configparser.ConfigParser()

    config.read("properties.ini")
    configurarion = config.get("main","configuration")
    neurons = int(config.get(configurarion,"neurons"))
    layers =  int(config.get(configurarion,"layers"))
    momentums = [float(num) for num in config.get(configurarion,"momentum").split(',')]
    epochs = int(config.get(configurarion,"epochs"))
    learning_rate = float(config.get(configurarion,"learning_rate"))

    return neurons, layers, momentums, epochs, learning_rate

if __name__ == "__main__":
    main()