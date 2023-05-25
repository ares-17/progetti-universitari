from Functions import Functions
from module import *


def main():

    # configurations = configs_as_dictionary()

    num_var_input, num_epoche,\
        dim_dataset, dim_train, \
        dim_test, usa_softmax_in_accuratezza = read_configurations()

    # creare un'unica assegnazione
    fun_attivazione = Functions.activation_functions()
    derivata_fun_attivazione = Functions.derivate_functions()
    fun_output = Functions.output_functions()
    fun_errore = Functions.error_functions()

    # creare un unico oggetto
    train_data, train_label, \
        valid_data, valid_label, \
        test_data, test_label = carica_mnist(dim_dataset, dim_train, dim_test)

    #
    # 1° TEST:	Numero nodi interni: 10		Learning rate: 0.0001	Momento: 0.0
    #
    num_test = 1
    num_nodi_per_strato = [10, 10]
    learning_rate = 0.0001
    alfa_momento = 0.0

    esegui_test(num_test, num_nodi_per_strato, learning_rate,
                alfa_momento, num_var_input, fun_errore,
                fun_attivazione, derivata_fun_attivazione, fun_output,
                train_data, train_label, valid_data, valid_label, num_epoche,
                test_data, test_label, usa_softmax_in_accuratezza)


if __name__ == "__main__":
    main()
