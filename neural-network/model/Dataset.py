import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.datasets import mnist

class Dataset:
    def __init__(self, shuffle=False, validation_ratio=0.2, training_size=10000, test_size=2500):
        self.train_data, self.train_label, self.test_data,  \
            self.test_label, self.valid_data, self.valid_label = \
                None, None, None, None, None, None

        self.shuffle = shuffle
        self.validation_ratio = validation_ratio
        self.training_size = training_size
        self.test_size = test_size
        self.data()

    def data(self):
        """
        Gets train's dataset, label and test's dataset, label from mnist.
        Only on train and test dataset performs certain operations that simplify its access 
        """
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()

        #self.train_data = self.train_data[:self.training_size]
        #self.train_label = self.train_label[:self.training_size]

        #self.test_data = self.test_data[:self.test_size]
        #self.test_label = self.test_label[:self.test_size]

        if self.shuffle:
            self.train_data, self.train_label = self.permutation(self.train_data, self.train_label)
            self.test_data, self.test_label = self.permutation(self.test_data, self.test_label)

        self.train_data = self.prepare_data(self.train_data)
        self.test_data = self.prepare_data(self.test_data)

        self.train_data, self.train_label, self.valid_data, self.valid_label =  \
            self.train_validation_split(self.train_data, self.train_label)

        self.train_label = self.one_hot(self.train_label)
        self.valid_label = self.one_hot(self.valid_label)
        self.test_label = self.one_hot(self.test_label)

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
        #data = self.resize_images(data)
        shape = (data.shape[0], data.shape[1] * data.shape[1])
        data = data.reshape(shape)
        data = data.T
        data = data / 255

        return data

    def resize_images(self, data, dim_img = 28):
        resized_matrix = np.zeros((data.shape[0], dim_img, dim_img))
        for i in range(data.shape[0]):
            original_image = data[i]
            resized_image = cv2.resize(original_image,( dim_img , dim_img))
            resized_matrix[i] = resized_image
        return resized_matrix

    def train_validation_split(self, X, Y):
        valid_size = int(X.shape[1] * self.validation_ratio)
        train_data = X[:,:-valid_size]
        train_label = Y[:-valid_size]
        valid_data = X[:,-valid_size:]
        valid_label = Y[-valid_size:]
        return train_data, train_label, valid_data, valid_label

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
