import numpy as np


class Layer:
    def __init__(self, shape, activation, derivative, momentum=0):
        self.W = np.random.normal(0, 0.1, (shape[0], shape[1]))
        self.B = np.random.normal(0, 0.1, (shape[0], 1))
        self.activation = activation
        self.derivative = derivative
        self.momentum = momentum
        self.dW_prev = np.zeros_like(self.W)
        self.db_prev = np.zeros_like(self.B)
        self.A, self.Z, self.dZ, self.db, self.dW = None, None, None, None, None

    def forward_prop(self, input):
        self.A = self.W.dot(input) + self.B
        self.Z = self.activation(self.A)

    def backward_prop(self, dZ, input, m):
        self.dZ = dZ
        self.dW = self.dZ.dot(input.T)
        self.db = np.sum(self.dZ)

    def update_params(self, alpha):
        self.dW = self.momentum * self.dW_prev - alpha * self.dW
        self.db = self.momentum * self.db_prev - alpha * self.db
        self.W += self.dW
        self.B += self.db

        self.dW_prev = self.dW
        self.db_prev = self.db
