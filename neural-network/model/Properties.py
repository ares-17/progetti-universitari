import configparser
from train import *

class Properties:

    ACT_FUNCTIONS_MAP = {
        'relu' : {
            'function' : ReLU,
            'derivative': ReLU_deriv
        },
        'sigmoide' : {
            'function' : sigmoide,
            'derivative' : sigmoide_deriv
        }
    }

    ERROR_FUNCTIONS_MAP = {
        'cross-entropy' : {
            'function' : cross_entropy,
            'derivative': cross_entropy_deriv
        },
        'sum-of-square' : {
            'function' : sum_of_square,
            'derivative' : sum_of_square_deriv
        }
    }

    def __init__(self, filename):
        self.neurons = None
        self.momentum =  None
        self.epochs = None
        self.learning_rate = None
        self.act_function = None
        self.error_function = None
        self.read_file(filename)

    def read_file(self, filename):
        """
        Read some configs from properties file.
        Does not handle any exceptions
        """
        config = configparser.ConfigParser()
        config.read(filename)
        configurarion = config.get("main","configuration")

        self.neurons  = [int(num) for num in config.get(configurarion,"neurons").split(',')]
        self.momentum = [float(num) for num in config.get(configurarion,"momentum").split(',')]
        self.epochs = int(config.get(configurarion,"epochs"))
        self.learning_rate = [float(num) for num in config.get(configurarion,"learning_rate").split(',')]
        self.act_function = Properties.ACT_FUNCTIONS_MAP[config.get(configurarion,"act_function").lower()]
        self.error_function = Properties.ERROR_FUNCTIONS_MAP[config.get(configurarion,"error_function").lower()]