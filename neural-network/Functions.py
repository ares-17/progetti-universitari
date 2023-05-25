import numpy as np


class Functions:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
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

    @staticmethod
    def identity(a):
        return a

    @staticmethod
    def derivata_identity(a):
        return 1

    @staticmethod
    def sigmoide(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def derivata_sigmoide(a):
        z = Functions.sigmoide(a)
        return z * (1 - z)

    @staticmethod
    def cross_entropy(predizione_rete, label):
        y = np.array(predizione_rete)
        t = np.array(label)

        epsilon = 1e-15
        y = np.clip(y, epsilon, 1. - epsilon)

        return -np.sum(t * np.log(y))

    @staticmethod
    def derivata_cross_entropy(y, t):
        return -(t / y)

    @staticmethod
    def cross_entropy_softmax(predizione_rete, label):
        y = np.array(predizione_rete)
        t = np.array(label)

        return Functions.cross_entropy(Functions.softmax(y), t)

    @staticmethod
    def derivata_cross_entropy_softmax(y, t):
        return Functions.softmax(y) - t

    @staticmethod
    def sum_of_square(predizione_rete, label):
        y = np.array(predizione_rete)
        t = np.array(label)

        return 0.5 * np.sum(np.square(y - t))

    @staticmethod
    def derivata_sum_of_square(y, t):
        return y - t

    @staticmethod
    def activation_functions():
        return Functions.sigmoide, Functions.identity

    @staticmethod
    def derivate_functions():
        return Functions.derivata_sigmoide, Functions.derivata_identity

    @staticmethod
    def error_functions():
        return Functions.cross_entropy_softmax, Functions.derivata_cross_entropy_softmax

    @staticmethod
    def output_functions():
        return Functions.identity, Functions.derivata_identity


