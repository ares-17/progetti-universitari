import configparser
from train import *
import pandas as pd
from Properties import *
from Dataset import *
from Analysis import *
"""
Caratteristiche:
1. dividere opportunatamente in training e test set con un rapporto approssimato a 1:4
2. fissare la discesa del gradiente con il momento
3. studiare l'andamento della rete per derivare:
    - numero epoche necessarie all'apprendimento
    - andamento errore su training e validation set, accuratezza su test set con un singolo strato al variare del learning rate
        e dal momento per almeno 5 diverse dimensioni dello strato interno
    - lasciare invariati parametri come funzioni di output
"""

def main():
    properties = read_properties()
    ds = Dataset(shuffle=True)
    analysis = Analysis()

    for neurons in properties.neurons:
        for rate in properties.learning_rate:
            for momentum in properties.momentum:
                print(f"starting with: momentum {momentum}, learning_rate {rate}, neurons {neurons}")
                layers = get_layers(neurons, momentum, ds.train_data.shape[0])
                analysis.partial(*gradient_descent(ds, layers, rate,properties.epochs), ds.test_data, layers)
                print(f"end momentum {momentum} with accuracy on train: {analysis.accuracies[-1].max()}")
    
    compare_results(analysis.accuracies, f"momentum-analysis-{properties.epochs}-epochs")

def get_layers(neurons, momentum, columns):
    return [Layer((neurons, columns), ReLU, ReLU_deriv, momentum), 
            Layer((10, neurons), softmax, ReLU_deriv, momentum)
    ]

def read_properties() -> Properties:
    """
    Read some configs from properties file.
    Does not handle any exceptions
    """
    config = configparser.ConfigParser()
    config.read("properties.ini")
    configurarion = config.get("main","configuration")

    return Properties(
        [int(num) for num in config.get(configurarion,"neurons").split(',')],
        [float(num) for num in config.get(configurarion,"momentum").split(',')],
        int(config.get(configurarion,"epochs")),
        [float(num) for num in config.get(configurarion,"learning_rate").split(',')],
    )

if __name__ == "__main__":
    main()