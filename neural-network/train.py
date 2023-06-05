import numpy as np
import cv2
from cache_functions import *
from matplotlib import pyplot as plt
from Layer import *
import os

def data(shuffle=False, with_cache=False):
  (train_data, train_label), (test_data, test_label) = get_mnist(with_cache)
  if shuffle:
    permutation = np.random.permutation(train_data.shape[0])
    train_data = train_data[permutation]
    train_label = train_label[permutation]
    
    permutation = np.random.permutation(test_data.shape[0])
    test_data = test_data[permutation]
    test_label = test_label[permutation]
  train_data, m , n = prepare_data(train_data)
  return (train_data, train_label), (test_data, test_label), m , n

def prepare_data(data):
  shape = (data.shape[0], data.shape[1] * data.shape[1])
  data = data.reshape(shape)
  m, n = data.shape
  data = data.T
  data = data / 255 
  return data, m, n

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
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

def backward_prop(X, Y, layers, m):
    input_layers = [X]
    for index in range(len(layers) - 1):
        input_layers.append(layers[index].A)

    dZ = layers[-1].A - one_hot(Y)
    for index in range(len(layers) - 1, -1, -1):
        current = layers[index]
        current.backward_prop(dZ, input_layers[index], m)
        if index - 1 > - 1:
            dZ = current.W.T.dot(dZ) * current.derivative(layers[index - 1].Z)

def update_params(alpha, layers):
    for layer in layers:
        layer.update_params(alpha)

def gradient_descent(X, Y, layers, alpha, iterations, m):
    accuracy = np.empty(iterations)
    for i in range(iterations):
        forward_prop(X, layers)
        backward_prop(X, Y, layers, m)
        update_params(alpha, layers)
        accuracy[i] = current_accuracy(i, layers, Y)
    return accuracy

def current_accuracy(iteration, layers, Y):
    predictions = np.argmax(layers[-1].A, 0)
    return get_accuracy(predictions, Y)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def compare_results(accuracies, name):
    for accuracy in accuracies:
        plt.plot(accuracy[1], label=f"Accuracy with {accuracy[0]} momentum")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(get_result_path(name))
    plt.close()

def get_result_path(name):
    return os.path.join(os.getcwd(), "results", name + ".png")