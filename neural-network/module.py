import cv2
import time
import numpy as np
import pylab as pl
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot
import configparser

from ReteNeurale import *


def softmax(y):
    output_rete = np.array(y, ndmin=3)
    e_output_rete = np.exp(y)

    sum_e_output_rete_axis1 = np.sum(e_output_rete, axis=1)
    e_output_rete_transpose = np.array(e_output_rete, ndmin=3).transpose(0, 2, 1)

    forma = np.shape(output_rete)
    dim = forma[0]
    neuroni = forma[1]
    feature = forma[2]

    e_output_rete_transpose_reshape = np.reshape(e_output_rete_transpose, (dim, neuroni), order='C')
    output_rete_normalizzato = e_output_rete_transpose_reshape / sum_e_output_rete_axis1
    risposta = np.reshape(output_rete_normalizzato, (dim, neuroni, feature))

    return risposta


def identity(a):
    return a


def derivata_identity(a):
    return 1


def sigmoide(a):
    return 1 / (1 + np.exp(-a))


def derivata_sigmoide(a):
    z = sigmoide(a)
    return z * (1 - z)


def cross_entropy(predizione_rete, label):
    y = np.array(predizione_rete)
    t = np.array(label)

    epsilon = 1e-15
    y = np.clip(y, epsilon, 1. - epsilon)

    return -np.sum(t * np.log(y))


def derivata_cross_entropy(y, t):
    return -(t / y)


def cross_entropy_softmax(predizione_rete, label):
    y = np.array(predizione_rete)
    t = np.array(label)

    return cross_entropy(softmax(y), t)


def derivata_cross_entropy_softmax(y, t):
    return (softmax(y) - t)


def sum_of_square(predizione_rete, label):
    y = np.array(predizione_rete)
    t = np.array(label)

    return 0.5 * np.sum(np.square(y - t))


def derivata_sum_of_square(y, t):
    return y - t


def errore_rete(rete_neurale, generic_set_data, label):
    output_rete_neurale = rete_neurale.risposta_rete(generic_set_data)
    return rete_neurale.fun_errore(output_rete_neurale, label)


def accuratezza(rete_neurale, generic_set_data, label, use_softmax):
    label = np.array(label)

    if use_softmax == True:
        output_rete_neurale = rete_neurale.risposta_rete_con_softmax(generic_set_data)
    else:
        output_rete_neurale = rete_neurale.risposta_rete(generic_set_data)

    output_rete_neurale_max = np.argmax(output_rete_neurale, axis=1)
    label_max = np.argmax(label, axis=1)

    confronto_uguaglianza = np.equal(output_rete_neurale_max, label_max)
    conteggio_corrette = np.count_nonzero(confronto_uguaglianza, axis=None)

    return conteggio_corrette / len(generic_set_data)


def carica_mnist(dim_dataset, dim_train, dim_test):
    (train_data, train_label), (test_data, test_label) = mnist.load_data(path='mnist.dataset')

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


def discesa_gradiente(rete_neurale, train_data, train_label, valid_data, valid_label, num_epoche=1000,
                      learning_rate=0.001, alfa_momento=0.0):
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

        derivate = rete_attuale.backward_propagation(train_data, train_label)
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

        errore_train_rete_attuale = errore_rete(rete_attuale, train_data, train_label)
        errore_valid_rete_attuale = errore_rete(rete_attuale, valid_data, valid_label)
        errore_valid_rete_migliore = errore_rete(rete_migliore, valid_data, valid_label)

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


def analisi_risultati_apprendimento(rete_addestrata, num_test, tempo_inizio_esecuzione, learning_rate, alfa_momento,
                                    test_data, test_label, usa_softmax_in_accuratezza, punti_errore_train,
                                    punti_errore_valid, punto_min):
    print(f'\n****** {num_test}° TEST ******')
    print('Tempo inizio esecuzione: ', tempo_inizio_esecuzione)
    print('Tempo fine esecuzione: ', time.asctime(time.localtime(time.time())))
    print(f'\nNumero nodi interni = {rete_addestrata.num_nodi_per_strato[0]}')
    print(f'Learning rate = {learning_rate}')
    print(f'Coefficiente del momento = {alfa_momento}')

    print("\nAccuratezza sul test set: ",
          accuratezza(rete_addestrata, test_data, test_label, usa_softmax_in_accuratezza))

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


def esegui_test(num_test, num_nodi_per_strato, learning_rate, alfa_momento, num_var_input, fun_errore, fun_attivazione,
                derivata_fun_attivazione, fun_output, train_data, train_label, valid_data, valid_label, num_epoche,
                test_data, test_label, usa_softmax_in_accuratezza):
    rete_neurale = ReteNeurale(num_var_input, num_nodi_per_strato, fun_errore, fun_attivazione,
                               derivata_fun_attivazione, fun_output)
    print(rete_neurale.to_string())

    tempo_inizio_esecuzione = time.asctime(time.localtime(time.time()))
    rete_addestrata, punti_errore_train, \
        punti_errore_valid, punto_min = discesa_gradiente(rete_neurale, train_data,
                                                          train_label, valid_data,
                                                          valid_label, num_epoche,
                                                          learning_rate, alfa_momento)

    analisi_risultati_apprendimento(rete_addestrata, num_test, tempo_inizio_esecuzione, learning_rate, alfa_momento,
                                    test_data, test_label, usa_softmax_in_accuratezza, punti_errore_train,
                                    punti_errore_valid, punto_min)
    return

'''
    refactor del metodo
    le variabili : num_var_input, num_epoche, dim_dataset, dim_train, dim_test, usa_softmax_in_accuratezza 
        sono include nell'oggetto configurations
'''
def esegui_test_ref(num_test, num_nodi_per_strato, learning_rate, alfa_momento, fun_errore, fun_attivazione,
                derivata_fun_attivazione, fun_output, train_data, train_label, valid_data, valid_label,
                test_data, test_label, configurations):

    rete_neurale = ReteNeurale(configurations['num_var_input'], num_nodi_per_strato, fun_errore, fun_attivazione,
                               derivata_fun_attivazione, fun_output)
    print(rete_neurale.to_string())

    tempo_inizio_esecuzione = time.asctime(time.localtime(time.time()))
    rete_addestrata, punti_errore_train, \
        punti_errore_valid, punto_min = discesa_gradiente(rete_neurale, train_data,
                                                          train_label, valid_data,
                                                          valid_label, configurations['num_epoche'],
                                                          learning_rate, alfa_momento)

    analisi_risultati_apprendimento(rete_addestrata, num_test, tempo_inizio_esecuzione, learning_rate, alfa_momento,
                                    test_data, test_label, configurations['usa_softmax_in_accuratezza'],
                                    punti_errore_train, punti_errore_valid, punto_min)
    return


def read_configurations():
    config = configparser.ConfigParser()
    config.read('properties.ini')

    num_var_input = int(config.get('Generics', 'num_var_input')) * int(config.get('Generics', 'num_var_input'))
    num_epoche = int(config.get('Generics', 'num_epoche'))
    dim_dataset = int(config.get('Generics', 'dim_dataset'))
    dim_train = int(config.get('Generics', 'dim_train'))
    dim_test = int(config.get('Generics', 'dim_test'))
    usa_softmax_in_accuratezza = config.get('Generics', 'usa_softmax_in_accuratezza')
    return num_var_input, num_epoche, dim_dataset, dim_train, dim_test, usa_softmax_in_accuratezza


def configs_as_dictionary():
    config = configparser.ConfigParser()
    config.read('properties.ini')

    return {
        'num_var_input': int(config.get('Generics', 'num_var_input')) * int(config.get('Generics', 'num_var_input')),
        'num_epoche': int(config.get('Generics', 'num_epoche')),
        'dim_dataset': int(config.get('Generics', 'dim_dataset')),
        'dim_train': int(config.get('Generics', 'dim_train')),
        'dim_test': int(config.get('Generics', 'dim_test')),
        'usa_softmax_in_accuratezza': config.get('Generics', 'usa_softmax_in_accuratezza')
    }
