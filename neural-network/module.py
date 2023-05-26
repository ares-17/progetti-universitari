import cv2
import time
import numpy as np
import pylab as pl
from keras.datasets import mnist
from matplotlib import pyplot
import configparser

from model.Dataset import Dataset
from model.ReteNeurale import *
from model.ReteNeurale import ReteNeurale
from model.Test import Test


def network_error(neural_network: ReteNeurale, inputs, target):
    output = neural_network.risposta_rete(inputs)
    return neural_network.fun_errore(output, target)


def accuracy(neural_network, generic_set_data, label, softmax):
    label = np.array(label)

    output = neural_network.risposta_rete(generic_set_data, softmax)
    return compare_argmax(output, label) / len(generic_set_data)


def compare_argmax(first, second, axis=1):
    """
    For each row find the index of the maximum value and
    check that the indexes are identical for the two arrays.
    Returns the number of identical elements
    """
    first_argmax = np.argmax(first, axis=axis)
    second_argmax = np.argmax(second, axis=axis)

    return np.count_nonzero(np.equal(first_argmax, second_argmax), axis=None)


def carica_mnist_ref(dataset_configs: dict) -> Dataset:
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    img_size = 14
    dim_valid = dataset_configs['dim_dataset'] - dataset_configs['dim_train']

    # e necessaria la copia ?
    # copia i primi dim_train elementi
    train_x = train_data[:dataset_configs['dim_train']].copy() / 255  # ogni img ha dimensione 28x28
    tmp = []

    for i in range(dataset_configs['dim_train']):
        res_img = cv2.resize(train_x[i], (img_size, img_size))  # ogni img ha dimensione 14x14
        tmp.append(np.array(res_img.flatten(), ndmin=2).transpose())  # ogni img ha dimensione 196x1

    train_x = tmp
    train_x = np.array(train_x, ndmin=3)
    tmp = []
    train_y = []

    for t in range(dataset_configs['dim_train']):
        tmp = np.zeros((10, 1))  # restituisce un nuovo array con 10 elementi, inizializzato con zeri
        tmp[train_label[t]] = 1  # codifica one-hot settando a 1 solo la label di interesse
        train_y.append(tmp)
    train_y = np.array(train_y, ndmin=3)

    # validation set
    valid_x = train_data[dataset_configs['dim_train']:dataset_configs['dim_dataset']].copy() / 255
    tmp = []

    for i in range(dim_valid):
        res = cv2.resize(valid_x[i], (img_size, img_size))
        tmp.append(np.array(res.flatten(), ndmin=2).transpose())

    valid_x = tmp
    valid_x = np.array(valid_x, ndmin=3)
    valid_y = []

    for t in range(dim_valid):
        tmp = np.zeros((10, 1))
        tmp[train_label[dataset_configs['dim_train'] + t]] = 1
        valid_y.append(tmp)
    valid_y = np.array(valid_y, ndmin=3)

    # test set
    test_x = test_data.copy() / 255
    tmp = []

    for i in range(dataset_configs['dim_test']):
        res = cv2.resize(test_x[i], (img_size, img_size))
        tmp.append(np.array(res.flatten(), ndmin=2).transpose())

    test_x = tmp
    test_x = np.array(test_x, ndmin=3)
    test_y = []

    for t in range(dataset_configs['dim_test']):
        tmp = np.zeros((10, 1))
        tmp[test_label[t]] = 1
        test_y.append(tmp)
    test_y = np.array(test_y, ndmin=3)

    return Dataset(train_x, train_y, valid_x, valid_y, test_x, test_y)


def carica_mnist(dim_dataset, dim_train, dim_test):
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    dim_img = 14
    dim_valid = dim_dataset - dim_train

    # training set
    train_x = train_data[:dim_train].copy() / 255  # ogni img ha dimensione 28x28
    tmp = []

    for i in range(dim_train):
        res_img = cv2.resize(train_x[i], (dim_img, dim_img))  # ogni img ha dimensione 14x14
        tmp.append(np.array(res_img.flatten(), ndmin=2).transpose())  # ogni img ha dimensione 196x1

    train_x = tmp
    train_x = np.array(train_x, ndmin=3)
    tmp = []
    train_y = []

    for t in range(dim_train):
        tmp = np.zeros((10, 1))  # restituisce un nuovo array con 10 elementi, inizializzato con zeri
        tmp[train_label[t]] = 1  # codifica one-hot settando a 1 solo la label di interesse
        train_y.append(tmp)
    train_y = np.array(train_y, ndmin=3)

    # validation set
    valid_x = train_data[dim_train:dim_dataset].copy() / 255
    tmp = []

    for i in range(dim_valid):
        res = cv2.resize(valid_x[i], (dim_img, dim_img))
        tmp.append(np.array(res.flatten(), ndmin=2).transpose())

    valid_x = tmp
    valid_x = np.array(valid_x, ndmin=3)
    valid_y = []

    for t in range(dim_valid):
        tmp = np.zeros((10, 1))
        tmp[train_label[dim_train + t]] = 1
        valid_y.append(tmp)
    valid_y = np.array(valid_y, ndmin=3)

    # test set
    test_x = test_data.copy() / 255
    tmp = []

    for i in range(dim_test):
        res = cv2.resize(test_x[i], (dim_img, dim_img))
        tmp.append(np.array(res.flatten(), ndmin=2).transpose())

    test_x = tmp
    test_x = np.array(test_x, ndmin=3)
    test_y = []

    for t in range(dim_test):
        tmp = np.zeros((10, 1))
        tmp[test_label[t]] = 1
        test_y.append(tmp)
    test_y = np.array(test_y, ndmin=3)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def discesa_gradiente(rete_neurale, dataset: Dataset, num_epoche=1000, learning_rate=0.001, alfa_momento=0.0):
    somma_derivate_pesi = []
    somma_derivate_bias = []

    punti_errore_train = []
    punti_errore_valid = []
    num_epoche_rete_non_migliora = 0
    punto_min = 0

    rete_migliore = rete_neurale
    rete_attuale = rete_neurale.copia()

    for epoca in range(num_epoche):

        for matrice in rete_attuale.matrici_pesi_strati:
            somma_derivate_pesi.append(np.zeros(np.shape(matrice)))
            # concatena un array vuoto di dimensione matrice

        for bias in rete_attuale.bias_strati:
            somma_derivate_bias.append(np.zeros(np.shape(bias)))

        derivate = rete_attuale.backward_propagation(dataset.train_set[0], dataset.train_set[1])
        derivate_pesi = derivate[0]
        derivate_bias = derivate[1]

        for strato in range(len(derivate_pesi)):
            somma_derivate_pesi[strato] = np.sum(derivate_pesi[strato], axis=0)

        for strato in range(len(derivate_bias)):
            somma_derivate_bias[strato] = np.sum(derivate_bias[strato], axis=0)

        # aggiorno i pesi con la regola di aggiornamento del gradiente
        for strato in range(len(rete_attuale.num_nodi_per_strato)):
            variazione_epoca_precedente = rete_attuale.matrici_pesi_strati[strato]
            rete_attuale.matrici_pesi_strati[strato] = rete_attuale.matrici_pesi_strati[strato] - (
                    learning_rate * somma_derivate_pesi[strato])
            rete_attuale.matrici_pesi_strati[strato] = rete_attuale.matrici_pesi_strati[strato] + alfa_momento * (
                    rete_attuale.matrici_pesi_strati[strato] - variazione_epoca_precedente)

        # aggiorno i bias con la regola di aggiornamento del gradiente
        for strato in range(len(rete_attuale.num_nodi_per_strato)):
            variazione_epoca_precedente = rete_attuale.bias_strati[strato]
            rete_attuale.bias_strati[strato] = rete_attuale.bias_strati[strato] - (
                    learning_rate * somma_derivate_bias[strato])
            rete_attuale.bias_strati[strato] = rete_attuale.bias_strati[strato] + alfa_momento * (
                    rete_attuale.bias_strati[strato] - variazione_epoca_precedente)

        errore_train_rete_attuale = network_error(rete_attuale, dataset.train_set[0], dataset.train_set[1])
        errore_valid_rete_attuale = network_error(rete_attuale, dataset.validation_set[0], dataset.validation_set[1])
        errore_valid_rete_migliore = network_error(rete_migliore, dataset.validation_set[0], dataset.validation_set[1])

        punti_errore_train.append(errore_train_rete_attuale)
        punti_errore_valid.append(errore_valid_rete_attuale)

        if errore_valid_rete_attuale < errore_valid_rete_migliore:
            rete_migliore = rete_attuale.copia()
            punto_min = epoca
            num_epoche_rete_non_migliora = 0
        else:
            num_epoche_rete_non_migliora = num_epoche_rete_non_migliora + 1
            if num_epoche_rete_non_migliora > 30:
                return rete_migliore, punti_errore_train, punti_errore_valid, punto_min

        print(f'\nEpoca: {epoca} (di {num_epoche}), Numero epoche senza miglioramenti: {num_epoche_rete_non_migliora}, \
        Errore sul training set: {errore_train_rete_attuale}, Errore validation set: {errore_valid_rete_attuale}')

    return rete_migliore, punti_errore_train, punti_errore_valid, punto_min


def analisi_risultati_apprendimento(rete_addestrata, test: Test, tempo_inizio_esecuzione, dataset: Dataset,
                                    usa_softmax_in_accuratezza, punti_errore_train,
                                    punti_errore_valid, punto_min):
    print(f'\n****** {test.num_test}° TEST ******')
    print('Tempo inizio esecuzione: ', tempo_inizio_esecuzione)
    print('Tempo fine esecuzione: ', time.asctime(time.localtime(time.time())))
    print(f'\nNumero nodi interni = {rete_addestrata.num_nodi_per_strato[0]}')
    print(f'Learning rate = {test.learning_rate}')
    print(f'Coefficiente del momento = {test.alfa_momento}')

    print("\nAccuratezza sul test set: ",
          accuracy(rete_addestrata, dataset.test_set[0], dataset.test_set[1], usa_softmax_in_accuratezza))

    pyplot.plot(punti_errore_train, label="Training set")
    pyplot.plot(punti_errore_valid, label="Validation set")
    ax = pl.gca()
    ylim = ax.get_ylim()
    pyplot.vlines(punto_min, ylim[0], ylim[1], label="Minimo", color="green")
    pyplot.xlabel("Epoche")
    pyplot.ylabel("Errore")
    pyplot.legend()
    pyplot.show()

    return


def esegui_test(test: Test, test_configs: dict, dataset: Dataset) -> None:
    neural_network = ReteNeurale(test_configs['num_var_input'], test.num_nodi_per_strato)
    print(neural_network.to_string())

    tempo_inizio_esecuzione = time.asctime(time.localtime(time.time()))
    rete_addestrata, punti_errore_train, \
        punti_errore_valid, punto_min = discesa_gradiente(neural_network, dataset, test_configs['num_epoche'],
                                                          test.learning_rate, test.alfa_momento)

    analisi_risultati_apprendimento(rete_addestrata, test, tempo_inizio_esecuzione, dataset,
                                    test_configs['usa_softmax_in_accuratezza'], punti_errore_train,
                                    punti_errore_valid, punto_min)
    return


def configs_as_dictionary():
    config = configparser.ConfigParser()
    config.read('properties.ini')

    dataset_configs = {
        'dim_dataset': int(config.get('dataset', 'dim_dataset')),
        'dim_train': int(config.get('dataset', 'dim_train')),
        'dim_test': int(config.get('dataset', 'dim_test')),
    }

    num_var_input = int(config.get('test', 'num_var_input')) * int(config.get('test', 'num_var_input'))
    test_configs = {
        'num_var_input': num_var_input,
        'num_epoche': int(config.get('test', 'num_epoche')),
        'usa_softmax_in_accuratezza': config.get('test', 'usa_softmax_in_accuratezza')
    }
    return dataset_configs, test_configs
