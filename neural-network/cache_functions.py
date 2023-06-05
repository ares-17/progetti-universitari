import os
import sys
import numpy as np
from keras.datasets import mnist

def get_mnist(with_cache=False):
    (train_data, train_label), (test_data, test_label) = (None, None), (None, None)
    if with_cache:
        try:
            with open('/app/cache/train_data', 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                    train_data = np.array(data)
            with open('/app/cache/train_label', 'rb') as f:
                    train_label = np.load(f, train_label, allow_pickle=True)
            with open('/app/cache/test_data', 'rb') as f:
                    test_data = np.load(f, test_data, allow_pickle=True)
            with open('/app/cache/test_label', 'rb') as f:
                    test_label = np.load(f, test_label, allow_pickle=True)
        except FileNotFoundError:
            print("file non trovato")
            (train_data, train_label), (test_data, test_label) = mnist.load_data()
            save_mnist(train_data, train_label, test_data, test_label)
    else:
        (train_data, train_label), (test_data, test_label) = mnist.load_data()
        #save_mnist(train_data, train_label, test_data, test_label)
    return (train_data, train_label), (test_data, test_label)

def save_mnist(train_data, train_label, test_data, test_label):
    cache_folder = "/app/cache"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    arrays = [('train_data', train_data),  
        ('train_label', train_label), 
        ('test_data',test_data), 
        ('test_label',test_label)]
    for array in arrays:
        np.save(os.path.join(cache_folder, array[0]), array[1])