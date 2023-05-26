from model.Functions import Functions
from module import *


class NeuralNet:
    def __init__(self, num_var_input, num_nodi_per_strato):
        self.num_var_input = num_var_input  # numero di variabili in ingresso alla rete
        self.num_nodi_per_strato = num_nodi_per_strato  # lista contenente il numero di nodi interni per ciascun livello previsto
        self.matrici_pesi_strati = []
        self.bias_strati = []
        self.fun_errore = Functions.error_functions()[0]
        self.derivata_fun_errore = Functions.error_functions()[1]
        self.fun_attivazione = Functions.activation_functions()
        self.derivata_fun_attivazione = Functions.activation_functions()  # n-upla contenente nella posizione i-esima la funzione di attivazione per lo strato i-esimo
        self.fun_output = Functions.activation_functions()[0]
        self.derivata_fun_output = np.vectorize(Functions.output_functions()[1])

        self.init_weigths(num_var_input, num_nodi_per_strato)
        self.init_bias(num_nodi_per_strato)

    def init_weigths(self, num_var_input, num_nodi_per_strato):
        num_input_strato_preced = num_var_input
        for n in num_nodi_per_strato:
            self.matrici_pesi_strati.append(np.random.normal(0, 0.1, (n, num_input_strato_preced)))
            num_input_strato_preced = n

    def init_bias(self, num_nodi_per_strato):
        for n in num_nodi_per_strato:
            self.bias_strati.append(np.random.normal(0, 0.1, (n, 1)))

    def to_string(self):
        return f"Rete Neurale\n \
		Numero variabili in input: {self.num_var_input}\n \
		Numero di nodi per ogni strato: {self.num_nodi_per_strato}\n \
		Funzione di errore utilizzata: {self.fun_errore}\n \
		Funzione attivazione per ogni strato: {self.fun_attivazione}\n \
		Funzione per il livello di output: {self.fun_output}\n \
		Matrici dei pesi: \n{self.matrici_pesi_strati}\n \
		Bias: \n{self.bias_strati}\n"

    def copia(self):
        rete_copia = NeuralNet(self.num_var_input, self.num_nodi_per_strato.copy())

        rete_copia.matrici_pesi_strati = []
        rete_copia.bias_strati = []

        for strato in range(len(self.num_nodi_per_strato)):
            rete_copia.matrici_pesi_strati.append(self.matrici_pesi_strati[strato].copy())
            rete_copia.bias_strati.append(self.bias_strati[strato].copy())

        return rete_copia

    def risposta_rete(self, generic_set_data, softmax=False):
        result = self.forward_propagation(generic_set_data)[1][len(self.num_nodi_per_strato) - 1]
        return Functions.softmax(result) if softmax else result

    def forward_propagation(self, train_data):
        if train_data.shape[1] != self.num_var_input:
            print("Errore. La dimensione dell'input non coincide con la dimensione dell'input della rete creata\n")
            return None

        g = np.vectorize(self.fun_output)

        input_nod_strato = []
        output_nod_strato = []

        z = np.array(train_data, ndmin=3)

        for strato in range(len(self.num_nodi_per_strato) - 1):
            f = np.vectorize(self.fun_attivazione[strato])
            a = np.matmul(self.matrici_pesi_strati[strato], z) + self.bias_strati[strato]
            z = f(a)

            input_nod_strato.append(a)
            output_nod_strato.append(z)

        a = np.matmul(self.matrici_pesi_strati[len(self.num_nodi_per_strato) - 1], z) + self.bias_strati[
            len(self.num_nodi_per_strato) - 1]
        z = g(a)

        input_nod_strato.append(a)
        output_nod_strato.append(z)

        return (input_nod_strato, output_nod_strato)

    def backward_propagation(self, train_data, train_label):
        input_output_nod_strato = self.forward_propagation(train_data)
        delta_nod_strato = []
        derivate_pesi_strato = []

        # calcolo dei delta dei nodi dello strato di output
        tmp = self.derivata_fun_output(input_output_nod_strato[0][len(self.num_nodi_per_strato) - 1])
        delta_out = tmp * self.derivata_fun_errore(input_output_nod_strato[1][len(self.num_nodi_per_strato) - 1],
                                                   train_label)
        delta_nod_strato.insert(0, delta_out)

        # calcolo dei delta dei nodi interni partendo dall'ultimo strato interno fino al primo
        for strato in range(len(self.num_nodi_per_strato) - 2, -1, -1):
            tmp = np.matmul(self.matrici_pesi_strati[strato + 1].transpose(), delta_nod_strato[0])
            delta_hidden = tmp * np.vectorize(self.derivata_fun_attivazione[strato])(input_output_nod_strato[0][strato])
            delta_nod_strato.insert(0, delta_hidden)

        # calcolo le derivate dei pesi delle connessioni dello strato di input
        train_data = np.array(train_data, ndmin=3)
        train_data_transpose = np.array(train_data, ndmin=3).transpose(0, 2, 1)
        tmp = np.matmul(delta_nod_strato[0], train_data_transpose)
        derivate_pesi_strato.append(tmp)

        # calcolo le derivate per tutti gli strati successivi
        for strato in range(1, len(self.num_nodi_per_strato)):
            out_transpose = input_output_nod_strato[1][strato - 1].transpose(0, 2, 1)
            tmp = np.matmul(delta_nod_strato[strato], out_transpose)
            derivate_pesi_strato.append(tmp)

        # le derivate dei bias sono i delta stessi
        derivate_bias_strato = delta_nod_strato

        return derivate_pesi_strato, derivate_bias_strato
