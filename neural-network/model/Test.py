class Test:
    _PROGRESSIVE_ID = 0

    def __init__(self, nodes, learning_rate, alfa_momento):
        self.num_test = Test._get_id()
        self.num_nodi_per_strato = nodes
        self.learning_rate = learning_rate
        self.alfa_momento = alfa_momento

    @staticmethod
    def _get_id():
        Test._PROGRESSIVE_ID += 1
        return Test._PROGRESSIVE_ID
