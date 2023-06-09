import configparser

class Properties:
    def __init__(self, filename):
        self.neurons = None
        self.momentum =  None
        self.epochs = None
        self.learning_rate = None
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