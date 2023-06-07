class Dataset:
    def __init__(self, train_data, train_label, test_data, test_label, valid_data, valid_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data 
        self.test_label = test_label 
        self.valid_data = valid_data 
        self.valid_label = valid_label 