import numpy as np
from cache_functions import *

class Dataset:
    def __init__(self, shuffle=False, validation_ratio=0.2):
        self.train_data, self.train_label, self.test_data,  \
            self.test_label, self.valid_data, self.valid_label = \
                None, None, None, None, None, None

        self.shuffle = shuffle
        self.validation_ratio = validation_ratio
        self.data()

    def data(self):
        """
        Gets train's dataset, label and test's dataset, label from mnist.
        Only on train and test dataset performs certain operations that simplify its access 
        """
        (self.train_data, self.train_label), (self.test_data, self.test_label) = get_mnist()
        if self.shuffle:
            self.train_data, self.train_label = self.permutation(self.train_data, self.train_label)
            self.test_data, self.test_label = self.permutation(self.test_data, self.test_label)

        self.train_data = self.prepare_data(self.train_data)
        self.test_data = self.prepare_data(self.test_data)

        self.train_data, self.train_label, self.valid_data, self.valid_label =  \
            self.train_validation_split(self.train_data, self.train_label)

    def permutation(self, dataset, label):
        permutation = np.random.permutation(dataset.shape[0])
        dataset = dataset[permutation]
        label = label[permutation]
        return dataset, label

    def prepare_data(self, data):
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

    def train_validation_split(self, X, Y):
        valid_size = int(X.shape[1] * self.validation_ratio)
        train_data = X[:,:-valid_size]
        train_label = Y[:-valid_size]
        valid_data = X[:,-valid_size:]
        valid_label = Y[-valid_size:]
        return train_data, train_label, valid_data, valid_label
