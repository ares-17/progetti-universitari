from module import *


def main():
    dataset_configs, test_configs = configs_as_dictionary()
    dataset: Dataset = carica_mnist_ref(dataset_configs)
    test: Test = Test([10, 10], 0.0001, 0.0)

    esegui_test(test, test_configs, dataset)


if __name__ == "__main__":
    main()
