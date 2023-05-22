from module import *


def main():
    num_var_input, num_epoche,\
        dim_dataset, dim_train, \
        dim_test, usa_softmax_in_accuratezza = read_configurations()

    fun_attivazione = (sigmoide, identity)
    derivata_fun_attivazione = (derivata_sigmoide, derivata_identity)
    fun_output = (identity, derivata_identity)

    fun_errore = (cross_entropy_softmax, derivata_cross_entropy_softmax)

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
