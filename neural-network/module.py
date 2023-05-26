import time
import numpy as np
import pylab as pl
from matplotlib import pyplot
import configparser

from model.Dataset import Dataset
from model.NeuralNet import *
from model.NeuralNet import NeuralNet
from model.Test import Test


def network_error(neural_network: NeuralNet, inputs, target):
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


def print_result(rete_addestrata, test: Test, tempo_inizio_esecuzione, dataset: Dataset,
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
    neural_network = NeuralNet(test_configs['num_var_input'], test.num_nodi_per_strato)
    print(neural_network.to_string())

    tempo_inizio_esecuzione = time.asctime(time.localtime(time.time()))
    rete_addestrata, punti_errore_train, \
        punti_errore_valid, punto_min = discesa_gradiente(neural_network, dataset, test_configs['num_epoche'],
                                                          test.learning_rate, test.alfa_momento)

    print_result(rete_addestrata, test, tempo_inizio_esecuzione, dataset,
                 test_configs['usa_softmax_in_accuratezza'], punti_errore_train,
                 punti_errore_valid, punto_min)


def configs_as_dictionary():
    config = configparser.ConfigParser()
    config.read('properties.ini')

    dataset_configs = {
        'dim_dataset': int(config.get('dataset', 'dim_dataset')),
        'dim_train': int(config.get('dataset', 'dim_train')),
        'dim_test': int(config.get('dataset', 'dim_test')),
        'dim_valid': int(config.get('dataset', 'dim_dataset')) - int(config.get('dataset', 'dim_train'))
    }

    num_var_input = int(config.get('test', 'num_var_input')) * int(config.get('test', 'num_var_input'))
    test_configs = {
        'num_var_input': num_var_input,
        'num_epoche': int(config.get('test', 'num_epoche')),
        'usa_softmax_in_accuratezza': config.get('test', 'usa_softmax_in_accuratezza')
    }
    return dataset_configs, test_configs
