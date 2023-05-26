class Dataset:
    def __init__(self, train_data, train_target, valid_data, valid_target, test_data, test_target):
        self.train_set = (train_data, train_target)
        self.validation_set = (valid_data, valid_target)
        self.test_set = (test_data, test_target)
