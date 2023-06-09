import numpy as np
import cv2
from matplotlib import pyplot as plt
from model.Layer import *
import os
from model.Dataset import *

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def cross_entropy(Y, A):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A + 1e-8)) / m

def forward_prop(X, layers):
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

def gradient_descent(ds: Dataset, layers, alpha, iterations):
    """
    For each layer execute forward and back propagation, update params e store accuracy
    """
    accuracy, error_train, error_valid = np.empty(iterations), np.empty(iterations), np.empty(iterations)
    one_hot_Y = one_hot(ds.train_label)
    for i in range(iterations):
        forward_prop(ds.train_data, layers)
        backward_prop(ds.train_data, one_hot_Y, layers)
        update_params(alpha, layers)

        accuracy[i] = current_accuracy(i, layers, ds.train_label)
        error_train[i] = get_current_error(one_hot_Y, layers)
        error_valid[i] = get_current_error(ds.valid_label, layers, forward=True, X=ds.valid_data, apply_one_hot=True)
    return error_train, error_valid, accuracy.max()

def current_accuracy(iteration, layers, Y):
    predictions = np.argmax(layers[-1].A, 0)
    return get_accuracy(predictions, Y)

def get_current_error(Y, layers, forward=False, X=None, apply_one_hot=False):
    if forward:
        forward_prop(X, layers)
    label = Y
    if apply_one_hot:
        label = one_hot(Y)
    return cross_entropy(label, layers[-1].A)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def compare_results(accuracies, name):
    """
    Print on axis the accuracies calculated with gradient descent.
    """
    for accuracy in accuracies:
        plt.plot(accuracy, label=f"Accuracy momentum")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(get_result_path(name))
    plt.close()

def get_result_path(name):
    return os.path.join(os.getcwd(), "results", name + ".png")