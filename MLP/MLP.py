"""
Questo file contiene la classe MLP preposta ad implementare la rete neurale;
- Ogni elemento e Vettoriazzato
- Non necessita di classi come Neuron o Layers
- Usa le classi/file: Utility & ActivationFunction
- MLP avra un bool per effettuare operazioni di classificazione oppure di regressione: classification

"""
import sys
sys.path.append("../")

from Trainers.TrainBackprop import *

class MLP:
    "Costruttore classe con stati; NOTA: Inseriti Pesi con bias"
    """
    :param n_feature : Numero di features
    :param n_hidden : Numero neuroni nell'hidden layer
    :param n_output : Numero neuroni nell'output layer
    :param activation_h : Funzione di attivazione dell'hidden layer
    :param activation_o : Funzione di attivazione dell'output layer
    :param eta : Learning rate
    :param lambd : Penalty term
    :param alfa : Momentum
    :param fan_in_h : Se true, usa il fan in nell'hidden layer per inizializzare i pesi
    :param range_start_h, range_end_h : I pesi nell'hidden layer sono inizializzati con valori in [range_start_h, range_end_h]
    :param fan_in_o : Se true, usa il fan in nell'output layer per inizializzare i pesi
    :param range_start_h, range_end_h : I pesi nell'output layer sono inizializzati con valori in 
                                        [range_start_o,range_end_o]
    
    :param classification: Se true, mlp deve risolvere un problema di classificazione(usato per sapere se
                            riempire lista di accuracy o di MEE)
    :param trainer: Indica quale algoritmo di ottimizzazione usare per la fase di training
    
    """
    def __init__(self, n_feature, n_hidden, n_output, activation_h, activation_o, eta=0.1, lambd=0, alfa=0.75,
                 fan_in_h=True, range_start_h=-0.7, range_end_h=0.7, fan_in_o=True, range_start_o=-0.7, range_end_o=0.7,
                 classification=True,trainer = TrainBackprop()):
        # Valori scalari
        # self.n_input = n_input  # righe di X
        self.n_feature = n_feature  # colonne di X, oppure neuroni input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Vettorizzato: Matrici
        ## NOTA: Indico gli indici delle dimensioni delle matrici/vettori
        self.W_h = init_Weights(n_hidden, n_feature, fan_in_h, range_start_h,
                                range_end_h)  # (n_neuroni_h x n_feature +1)
        self.W_o = init_Weights(n_output, n_hidden, fan_in_o, range_start_o,
                                range_end_o)  # (n_neuroni_o x n_neuroni_h +1)
        self.Out_h = None  # (n_esempi x n_neuroni_h)
        self.Out_o = None  # (n_esempi x n_neuroni_o) //Per Monk  un vettore
        self.Net_h = None  # (n_esempi x n_neuroni_h)
        self.Net_o = None  # (n_esempi x n_neuroni_o) //Per Monk un vettore

        # Si specifica il tipo di f. attivazione dei neuroni
        self.activation_h = activation_h
        self.activation_o = activation_o

        # Hyperparameter!
        self.eta = eta  # learning rate
        self.lambd = lambd  # regolarizzazione-penalityTerm
        self.alfa = alfa  # momentum

        # Lista per avere il plot LC, Accuracy(class->(N-num_err)/N), regress->MEE
        self.errors_tr = []  # MSE/num_epoche sul TR
        self.accuracies_tr = []  # Accuracy/num_epoche sul TR
        self.errors_vl = []  # MSE/num_epoche sul VL
        self.accuracies_vl = []  # Accuracy/num_epoche sul VL
        self.errors_mee_tr = []  # MEE sul TR
        self.errors_mee_vl = []  # MEE sulla VL

        self.gradients = [] #Lista dei gradienti

        # Servono nella fase di train->backperopagation; delta vecchio dei pesi hidden e output
        self.dW_o_old = np.zeros(self.W_o.shape)
        self.dW_h_old = np.zeros(self.W_h.shape)

        # Bool per Classificazione/Regressione
        self.classification = classification

        self.trainer = trainer

    "FeedFoward: X con bias"

    def feedforward(self, X):
        # Calcolo hidden layer
        self.Net_h = np.dot(X, self.W_h.T)
        self.Out_h = self.activation_h.compute_function(self.Net_h)  # Output_h=f(Net_h)

        # Calcolo output layer
        Out_h_bias = addBias(self.Out_h)
        self.Net_o = np.dot(Out_h_bias, self.W_o.T)
        self.Out_o = self.activation_o.compute_function(
            self.Net_o)  # Output_o=f(Net_o)=>Classificazione rete; Per Monk vettore

    "Backpropagation: X con bias"

    def backpropagation(self, X, T):
        assert T.shape == self.Out_o.shape

        # Calcolo della f'(Net_o), calcolo delta_neuroneOutput, calcolo delta peso
        ## NOTA: vedere slide Backprop.
        grad_f_o = self.activation_o.compute_function_gradient(self.Out_o)
        diff = (T - self.Out_o)
        delta_o = np.multiply(diff, grad_f_o)  # elemento-per-elemento
        Out_h_bias = addBias(self.Out_h)
        delta_W_o = np.dot(delta_o.T, Out_h_bias)

        # Calcolo della f'(Net_h), calcolo delta_o*pesi_interessati, calcolo delta hidden layer
        ## NOTA: vedere slide Backprop.
        grad_f_h = self.activation_h.compute_function_gradient(self.Out_h)
        W_o_nobias = removeBias(self.W_o)
        sp_h = np.dot(delta_o, W_o_nobias)
        delta_h = np.multiply(sp_h, grad_f_h)  # elemento-per-elemento
        delta_W_h = np.dot(delta_h.T, X)

        return delta_W_o / X.shape[0], delta_W_h / X.shape[0]


    "Classificazione: predizione"

    def predict_class(self, X, treshold=0.5):
        self.feedforward(X)
        predictions = np.zeros(self.Out_o.shape)
        predictions[self.Out_o >= treshold] = 1
        return predictions

    "Regressione: predizione"

    def predict_value(self, X):
        self.feedforward(X)
        return self.Out_o
