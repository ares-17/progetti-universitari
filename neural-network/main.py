from train import *
import pandas as pd
from model.Properties import *
from model.Dataset import *
from model.Analysis import *

def main():
    properties = Properties("properties.ini")
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

if __name__ == "__main__":
    main()