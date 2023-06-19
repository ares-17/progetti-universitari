import numpy as np
import cv2
from matplotlib import pyplot as plt
from model.Layer import *
import os
from model.Dataset import *

def ReLU(Z):
    return np.maximum(Z, 0)

def identity(a):
	return a

def identity_deriv(a):
	return 1

def sigmoide(a):
    return 1 / (1 + np.exp(-a))

def sigmoide_deriv(a):
    z = sigmoide(a)
    return z * (1 - z)

def softmax(y):
    y_exp = np.exp(y - y.max(0))
    z = y_exp / sum(y_exp, 0)
    return z

def cross_entropy(predictions, targets):
    return -np.mean(targets * np.log(predictions + 1e-10))

def cross_entropy_softmax(y, t):
    z = softmax(y)
    return -(t * np.log(z)).sum()

def cross_entropy_softmax_deriv(y, t):
    z = softmax(y)
    return z - t

def cross_entropy_deriv(y, t):
	return -(t / y)

def ReLU_deriv(Z):
    return Z > 0

def sum_of_square(predizione_rete, label):
	y = np.array(predizione_rete)
	t = np.array(label)

	return 0.5 * np.sum(np.square(y - t))

def sum_of_square_deriv(y, t):
	return y - t

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def forward_prop(X, layers):
    input_layer = X
    for layer in layers:
        layer.forward_prop(input_layer)
        input_layer = layer.Z

def backward_prop(X, one_hot_Y, layers, error_deriv):
    """
    For each layer store in array its output.
    Next, exeute back propagation with previous array and calculate derivative only 
    for n-1 layer
    """
    input_layers = [X]
    for index in range(len(layers) - 1):
        input_layers.append(layers[index].Z)

    dZ = error_deriv(layers[-1].Z , one_hot_Y) * layers[-1].derivative(layers[-1].A)
    for index in range(len(layers) - 1, -1, -1):
        current = layers[index]
        current.backward_prop(dZ, input_layers[index], X.shape[1])
        if index - 1 > - 1:
            dZ = current.W.T.dot(dZ) * layers[index - 1].derivative(layers[index - 1].A)

def update_params(alpha, layers):
    for layer in layers:
        layer.update_params(alpha)

def gradient_descent(ds: Dataset, layers, alpha, iterations, error_function):
    """
    For each layer execute forward and back propagation, update params e store accuracy
    """
    accuracy, error_train, error_valid = np.empty(iterations), np.empty(iterations), np.empty(iterations)
    for i in range(iterations):
        forward_prop(ds.train_data, layers)
        backward_prop(ds.train_data, ds.train_label, layers, error_function["derivative"])
        update_params(alpha, layers)

        accuracy[i] = current_accuracy(i, layers, ds.test_data , ds.test_label)
        error_train[i] = get_error(ds.train_data, ds.train_label, layers, error_function["function"])
        error_valid[i] = get_error(ds.valid_data, ds.valid_label, layers, error_function["function"])

        progress_bar(i, iterations)

    return error_train, error_valid, accuracy.max()

def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

def current_accuracy(iteration, layers, test_data, Y):
    forward_prop(test_data, layers)
    predictions = np.argmax(softmax(layers[-1].Z), 0)
    return get_accuracy(predictions, Y)

def get_error(X, Y, layers, erro_foo):
    forward_prop(X, layers)
    return erro_foo(layers[-1].Z, Y)

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