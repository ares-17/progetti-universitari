from keras.datasets import mnist
import cv2
import numpy as np

class Dataset:
    def __init__(self, dataset_configs: dict):
        self.load_mnist(dataset_configs)

    def load_mnist(self, dataset_configs: dict) -> None:
        (train_data, train_label), (test_data, test_label) = mnist.load_data()

        self._training_set(dataset_configs, train_data, train_label)
        self._validation_set(dataset_configs, train_data, train_label)
        self._test_set(dataset_configs, test_data, test_label)

    def _training_set(self, dataset_configs: dict, mnist_data, mnist_label) -> None:
        data = mnist_data[:dataset_configs['dim_train']].copy() / 255

        values = self._normalize_images(data, dataset_configs['dim_train'])
        targets = self._normalize_targets(mnist_label, dataset_configs['dim_train'])
        self.train_set = (values, targets)

    def _validation_set(self, dataset_configs: dict, mnist_data, mnist_label) -> None:
        data = mnist_data[dataset_configs['dim_train']:dataset_configs['dim_dataset']].copy() / 255

        values = self._normalize_images(data, dataset_configs['dim_valid'])
        targets = self._normalize_targets(mnist_label, dataset_configs['dim_valid'], dataset_configs['dim_train'])

        self.validation_set = (values, targets)

    def _normalize_images(self, inputs, size):
        values = []
        for i in range(size):
            res_img = cv2.resize(inputs[i], (14, 14))  # ogni img ha dimensione 14x14
            values.append(np.array(res_img.flatten(), ndmin=2).transpose())  # ogni img ha dimensione 196x1

        return np.array(values, ndmin=3)

    def _normalize_targets(self, mnist_labels, size, offset=0):
        targets = []

        for t in range(size):
            tmp = np.zeros((10, 1))  # restituisce un nuovo array con 10 elementi, inizializzato con zeri
            tmp[mnist_labels[offset + t]] = 1  # codifica one-hot settando a 1 solo la label di interesse
            targets.append(tmp)
        return np.array(targets, ndmin=3)

    def _test_set(self, dataset_configs: dict, test_data, test_label):
        test_x = test_data.copy() / 255
        tmp = []

        for i in range(dataset_configs['dim_test']):
            res = cv2.resize(test_x[i], (14, 14))
            tmp.append(np.array(res.flatten(), ndmin=2).transpose())

        test_x = tmp
        test_x = np.array(test_x, ndmin=3)
        test_y = []

        for t in range(dataset_configs['dim_test']):
            tmp = np.zeros((10, 1))
            tmp[test_label[t]] = 1
            test_y.append(tmp)
        test_y = np.array(test_y, ndmin=3)

        self.test_set = (test_x, test_y)
