import configparser
from train import *
import pandas as pd
from Properties import *

"""
Caratteristiche:
1. dividere opportunatamente in training e test set con un rapporto approssimato a 1:4
2. fissare la discesa del gradiente con il momento
3. studiare l'andamento della rete per derivare:
    - numero epoche necessarie all'apprendimento
    - andamento errore su training e validation set, accuratezza su test set con un singolo strato al variare del learning rate
        e dal momento per almeno 5 diverse dimensioni dello strato interno
    - lasciare invariati parametri come funzioni di output
"""

def main():
    properties = read_properties()

    (train_data, train_label), (test_data, test_label), (valid_data, valid_label) = data(shuffle=True)

    layers = None
    accuracies = []

    for neurons in properties.neurons:
        for rate in properties.learning_rate:
            for momentum in properties.momentum:
                print(f"starting with: momentum {momentum}, learning_rate {rate}, neurons {neurons}")
                layers = [
                    Layer((neurons, train_data.shape[0]), ReLU, ReLU_deriv, momentum), 
                    Layer((10, neurons), softmax, ReLU_deriv, momentum)
                    ]
                accuracy, error_train, error_valid =  \
                    gradient_descent(train_data, train_label,layers, rate, \
                        properties.epochs, valid_data, valid_label)
                accuracies.append((momentum, accuracy))
                test_accuracy = make_predictions(test_data, layers)
                print(f"end momentum {momentum} with accuracy on train: {accuracy.max()}, on test: {get_accuracy(test_accuracy, test_label)}")
    
    compare_results(accuracies, f"momentum-analysis-{properties.epochs}-epochs")

    plt.plot(error_train, label="Training error")
    plt.plot(error_valid, label="Validation error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


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

def read_properties() -> Properties:
    """
    Read some configs from properties file.
    Does not handle any exceptions
    """
    config = configparser.ConfigParser()
    config.read("properties.ini")
    configurarion = config.get("main","configuration")

    return Properties(
        [int(num) for num in config.get(configurarion,"neurons").split(',')],
        [float(num) for num in config.get(configurarion,"momentum").split(',')],
        int(config.get(configurarion,"epochs")),
        [float(num) for num in config.get(configurarion,"learning_rate").split(',')],
    )

if __name__ == "__main__":
    main()