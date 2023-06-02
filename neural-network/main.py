import numpy as np
import cv2
from keras.datasets import mnist
from matplotlib import pyplot as plt

def data(shuffle=False):
  (train_data, train_label), (test_data, test_label) = mnist.load_data()
  if shuffle:
    permutation = np.random.permutation(train_data.shape[0])
    train_data = train_data[permutation]
    train_label = train_label[permutation]
    
    permutation = np.random.permutation(test_data.shape[0])
    test_data = test_data[permutation]
    test_label = test_label[permutation]
  train_data = train_data.reshape(60000, 784)
  print(test_data.shape)
  m, n = train_data.shape
  train_data = train_data.T
  train_data = train_data / 255
  return (train_data, train_label), (test_data, test_label), m , n

def prepare_data(data):
  shape = (data.shape[0], data.shape[1] * data.shape[1])
  data = data.reshape(shape)
  m, n = data.shape
  data = data.T
  data = data / 255 
  return data, m, n

(train_data, train_label), (test_data, test_label), m, n = data(shuffle=True)

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    accuracy = np.empty(iterations)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
        predictions = get_predictions(A2)
        accuracy[i] = get_accuracy(predictions, Y)
    return accuracy

def show(accuracy):
    plt.plot(accuracy, label="Training set")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

accuracy = gradient_descent(train_data, train_label, 0.10, 300)
show(accuracy)
print(accuracy[-1])