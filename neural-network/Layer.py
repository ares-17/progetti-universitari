import numpy as np

class Layer:
    def __init__(self, shape, activation, derivative, momentum = 0):
        self.W = np.random.rand(shape[0], shape[1]) - 0.5
        self.B = np.random.rand(shape[0], 1) - 0.5
        self.activation = activation
        self.derivative = derivative
        self.momentum = momentum
        self.dW_prev = np.zeros_like(self.W)
        self.db_prev = np.zeros_like(self.B)
        self.A, self.Z, self.dZ, self.db, self.dW = None, None, None, None, None

    def forward_prop(self, input):
        self.Z = self.W.dot(input) + self.B
        self.A = self.activation(self.Z)

    def backward_prop(self, dZ, input, m):
        self.dZ = dZ
        self.dW = 1 / m * self.dZ.dot(input.T)
        self.db = 1 / m * np.sum(self.dZ)

    def update_params(self, alpha):
        self.dW = self.momentum * self.dW_prev - alpha * self.dW
        self.db = self.momentum * self.db_prev - alpha * self.db
        self.W += self.dW
        self.B += self.db
        
        self.dW_prev = self.dW
        self.db_prev = self.db