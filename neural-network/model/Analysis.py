from train import *

class Analysis:
    def __init__(self):
        self.accuracies = []
        self.errors_train = []
        self.errors_valid = []
        self.test_accuracy = []

    def partial(self, accuracy, error_train, error_valid, test_data, layers):
        self.accuracies.append(accuracy)
        self.errors_train.append(error_train)
        self.errors_valid.append(error_valid)
        self.test_accuracy = self.make_predictions(test_data, layers)

    def make_predictions(self, X, layers):
        """
        Execute forward propagation on network and gets prediction's class from result's array
        """
        forward_prop(X, layers)
        predictions = np.argmax(layers[-1].A, 0)
        return predictions

    def test_prediction(self, index, layers, X, Y):
        """
        Gets the index of the image in the X array, prints the expected class and the label class.
        Next, visualize the image with matplotlib
        """
        current_image = X[:, index, None]
        prediction = self.make_predictions(X[:, index, None], layers)
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()