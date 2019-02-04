
import sys
sys.path.append("../")

"""
Questo file contiene le varie funzioni di attivazione disponibili per la rete neurale.
Le funzioni di attivazioni sono vettorizzate
"""

import numpy as np


class ActivationFunction:
    """
    Calcola la funzione.
    :param X : matrice su cui calcolare la funzione
    :return Y : matrice contenente il risultato di f(X)
    """

    def compute_function(self, X):
        raise NotImplementedError("This method has not been implemented")

    """
    Calcola il gradiente della funzione.
    :param X : matrice su cui calcolare il gradiente
    :return Y : matrice contenente il risultato di grad(f(X)) 
    """

    def compute_function_gradient(self, X):
        raise NotImplementedError("This method has not been implemented")


class SigmoidActivation(ActivationFunction):
    def compute_function(self, X):
        return 1. / (1 + np.exp(-X))

    """
    NOTA: LA DERIVATA DI SIGMOIDE E'
          SIGM(X) * (1 -SIGM(X))
         ==> LA FUNZIONE DEL GRADIENTE VA CHIAMATA SU X' = SIGMOID(X)
         ==> IN QUESTO MODO NON RICALCOLO DUE VOLTE SIGMOID(X) CHE E' COSTOSO VISTO CHE C'E' DI MEZZO IL
                CALCOLO DI UN'ESPONENZIALE!!!!!
    """

    def compute_function_gradient(self, X):
        return np.multiply(X, (1 - X))


class TanhActivation(ActivationFunction):
    def compute_function(self, X):
        return np.tanh(X)

    """
       NOTA: LA DERIVATA DI TANH E'
             1 -(TANH(X) **2)
            ==> LA FUNZIONE DEL GRADIENTE VA CHIAMATA SU X' = TANH(X) 
            ==> IN QUESTO MODO NON RICALCOLO DUE VOLTE TANH(X) CHE E' COSTOSO VISTO CHE C'E' DI MEZZO IL
                CALCOLO DI UN'ESPONENZIALE!!!!!
       """

    def compute_function_gradient(self, X):
        return 1 - (X ** 2)


class LinearActivation(ActivationFunction):
    def compute_function(self, X):
        return X

    def compute_function_gradient(self, X):
        return np.ones((X.shape[0], X.shape[1]))


"----------------------------------------------------------------------------------------------------------------------"
if __name__ == "__main__":
    X = np.arange(0, 9).reshape(3, 3)
    print("X = ", X)

    activation = SigmoidActivation()

    print("Sigmoid(X) =\n", activation.compute_function(X))
    print("Gradient of Sigmoid(X)\n", activation.compute_function_gradient(activation.compute_function(X)))

    activation = TanhActivation()

    print("Tanh(X) =\n", activation.compute_function(X))
    print("Gradient of Tanh(X)\n", activation.compute_function_gradient(activation.compute_function(X)))

    activation = LinearActivation()

    print("Linear(X) =\n", activation.compute_function(X))
    print("Gradient of Linear(X)\n", activation.compute_function_gradient(X))
