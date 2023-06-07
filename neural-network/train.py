import numpy as np
import cv2
from cache_functions import *
from matplotlib import pyplot as plt
from Layer import *
import os

def data(shuffle=False, with_cache=False):
    """
    Gets train's dataset, label and test's dataset, label from mnist.
    Only on train and test dataset performs certain operations that simplify its access 
    """
    (train_data, train_label), (test_data, test_label) = get_mnist(with_cache)
    if shuffle:
        train_data, train_label = permutation(train_data, train_label)
        test_data, test_label = permutation(test_data, test_label)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_data, train_label, valid_data, valid_label =  \
        train_validation_split(train_data, train_label)

    return (train_data, train_label), (test_data, test_label), (valid_data, valid_label)

def permutation(dataset, label):
    permutation = np.random.permutation(dataset.shape[0])
    dataset = dataset[permutation]
    label = label[permutation]
    return dataset, label

def prepare_data(data):
    """
    Accepts arrays of 3 dimensions, with the last two identical.
    Returns the same array by performing the operations:
    1. transponse
    2. resizing the dimensions to 2
    3. normalization with respect to the value 255
    """
    shape = (data.shape[0], data.shape[1] * data.shape[1])
    data = data.reshape(shape)
    data = data.T
    data = data / 255 
    return data

def train_validation_split(X, Y, validation_ratio=0.2):
    valid_size = int(X.shape[1] * validation_ratio)
    print(f"valid_size : {X.shape[1]}, {valid_size}")
    train_data = X[:,:-valid_size]
    train_label = Y[:-valid_size]
    valid_data = X[:,-valid_size:]
    valid_label = Y[-valid_size:]
    return train_data, train_label, valid_data, valid_label

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def cross_entropy(Y, A):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A + 1e-8)) / m
    return loss

def forward_prop(X, layers):
    """
    For each level, forward propagation is performed and the output for the next level is stored.
    """
    input_layer = X
    for layer in layers:
        layer.forward_prop(input_layer)
        input_layer = layer.A

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(X, one_hot_Y, layers):
    """
    For each layer store in array its output.
    Next, exeute back propagation with previous array and calculate derivative only 
    for n-1 layer
    """
    input_layers = [X]
    for index in range(len(layers) - 1):
        input_layers.append(layers[index].A)

    dZ = layers[-1].A - one_hot_Y
    for index in range(len(layers) - 1, -1, -1):
        current = layers[index]
        current.backward_prop(dZ, input_layers[index], X.shape[1])
        if index - 1 > - 1:
            dZ = current.W.T.dot(dZ) * current.derivative(layers[index - 1].Z)

def update_params(alpha, layers):
    for layer in layers:
        layer.update_params(alpha)

def gradient_descent(X, Y, layers, alpha, iterations):
    """
    For each layer execute forward and back propagation, update params e store accuracy
    """
    accuracy = np.empty(iterations)
    error = np.empty(iterations)
    for i in range(iterations):
        forward_prop(X, layers)
        one_hot_Y = one_hot(Y)
        backward_prop(X, one_hot_Y, layers)
        update_params(alpha, layers)
        accuracy[i] = current_accuracy(i, layers, Y)
        error[i] = cross_entropy(one_hot_Y, layers[-1].A)
    return accuracy, error

def current_accuracy(iteration, layers, Y):
    predictions = np.argmax(layers[-1].A, 0)
    return get_accuracy(predictions, Y)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def compare_results(accuracies, name):
    """
    Print on axis the accuracies calculated with gradient descent.
    """
    for accuracy in accuracies:
        plt.plot(accuracy[1], label=f"Accuracy with {accuracy[0]} momentum")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(get_result_path(name))
    plt.close()

def get_result_path(name):
    return os.path.join(os.getcwd(), "results", name + ".png")