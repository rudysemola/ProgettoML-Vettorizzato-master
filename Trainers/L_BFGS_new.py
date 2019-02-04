"""
Questo file contiene l'implementazione del metodo del Quasi Newton a memoria limitata.
L-BFGS
"""
"""
TODO: FINIRE E CONTROLLARE!!!!!!!!!!
"""

import sys

sys.path.append("../")

from Trainers.Training import *
import numpy as np
from Utilities.UtilityCM2 import *
from Utilities.Utility import *
from Trainers.LineSearch_new import *


class LBFGS(Training):
    """
    Inizializza gli iperparametri algoritmici
    :param eta_start: Eta iniziale da provare durante la AWLS
    :param eta_max: Eta massimo accettabile
    :param max_iter_AWLS_train : Numero massimo di iterazioni che (in media) AWLS può compiere
    :param m1: Parametro della condizione di Armijo
    :param m2: Parametro della condizione di Wolfe
    :param tau : Parametro che indica di quanto devo incrementare eta nella prima fase di AWLS
    :param sfgrd: Valore della safeguard. Serve ad assicurare che durante la fase di interpolazione,
            l'ampiezza dell'intervallo diminuisca almeno di un fattore pari a sfgrd
    :param mina: Indica quanto deve essere l'ampiezza minima dell'intervallo in cui fare interpolazione
    :param m : Indica la quantità di "memoria" usata dall'algoritmo L-BFGS. Tiene cioè in memoria informazioni
                sulle ultime m iterazioni.

    :param delta : Indica il fattore usato per l'inizializzazione di H0. H0 = delta*I
    :param use_delta: Se true, indica che H0 deve essere inizializzato usando il fattore delta.
                        Altrimenti usa formula descritta in Nocedal.
    """

    def __init__(self, eta_start=1, eta_max=2, max_iter_AWLS_train=100, m1=0.0001, m2=0.9, tau=0.9,
                 sfgrd=0.001, mina=1e-16, m=3, delta=0.8, use_delta=True):

        # PARAMETRI LS
        self.eta_start = eta_start  # L-BFGS vuole AWLS=> passare alpha=1
        self.eta_max = eta_max
        self.max_iter_AWLS_train = max_iter_AWLS_train
        self.m1 = m1
        self.m2 = m2
        self.tau = tau
        self.sfgrd = sfgrd
        self.mina = mina

        # PARAMETRI BFGS
        self.m = m
        self.s_y_list = []
        self.H0 = None
        self.delta = delta
        self.use_delta = use_delta

        self.it_AWLS_list = []

        self.w_prec = None
        self.gradE_prec = None

        self.w_new = None
        self.gradE_new = None

        self.norm_gradE = 0
        self.norm_gradE_0 = 0
        self.epsilon_prime = None

        self.n_iters_AWLS_train = 0

        # Per sapere quante volte usa il delta il calcolo di H0!
        self.num_delta = 0
    """
    Esegue la prima iterazione del L-BFGS per inizializzare i vari elementi in modo corretto
    Nell'ordine, le azioni eseguite sono:
    
    1) Calcola funzione nel punto iniziale w_start
    2) Calcola gradiente nel punto iniziale
    3) Calcola norma del gradiente e setta quindi il valore di epsilon_prime = eps * norma_gradiente
    4) Inizializza H0 come H0 = delta*I
    5) Calcola direzione d = H0 * gradiente
    6) Calcola eta ottimo con AWLS
    7) Calcola errore di TR e VL (MSE + MEE)
    8) Aggiorna i pesi, ottenendo w_new
    9) Calcola funzione in w_new
    10) Calcola gradiente in w_new
    11) Calcola norma del gradiente in w_new
    12) Controlla se norma_gradE_new < epsilon_prime
    13) Calcola s = w_new - w_start
    14) Calcola y = gradE_new - gradE_start
    15) Aggiunge (s,y) alla lista
    
    :param mlp : MLP su cui effetuare training
    :param X : Matrice di training
    :param T : Matrice dei target di training
    :param X_vl : Matrice di validazione
    :param T_vl : Matrice dei target di validazione
    :param eps : Accuracy desiderata
    :param threshold: Indica threshold usato per classificare oggetti (solo per problemi di classificazione)
    
    :return converged : se true, indica che si è raggiunto un minimo
    """

    def do_first_iteration(self, mlp, X, T, X_vl, T_vl, eps, threshold):

        # Calcolo la funzione nel punto iniziale
        self.w_prec = get_current_point(mlp)

        """
        Calcolo funzione e gradiente nel punto iniziale
        """
        E, gradE = evaluate_function(mlp, X, T, self.w_prec)
        mlp.errors_tr.append(E)

        self.gradE_prec = gradE
        self.norm_gradE_0 = np.linalg.norm(self.gradE_prec)
        self.epsilon_prime = eps * self.norm_gradE_0

        """
        Inizializzo H0 come delta*I
        """
        self.H0 = self.delta * np.eye(self.w_prec.shape[0])

        """
        Calcolo la direzione iniziale d_0 = - H0*gradE_0
        """
        d = - np.dot(self.H0, self.gradE_prec)

        assert d.shape[1] == 1
        assert d.shape[0] == self.gradE_prec.shape[0]

        """
        Effettua la line search per trovare eta ottimo
        """
        mlp.eta, it_AWLS = AWLS(mlp, X, T, d, mlp.lambd, self.eta_start, self.eta_max,
                                self.max_iter_AWLS_train,
                                self.m1, self.m2,
                                self.tau, self.mina, self.sfgrd, l_bfgs=True, debug=True)
        print("eta= ", mlp.eta)
        print("it= ", it_AWLS)

        self.it_AWLS_list.append(it_AWLS)
        self.n_iters_AWLS_train += it_AWLS

        """
        Aggiorna i pesi
        """

        self.w_new = self.w_prec + mlp.eta * d

        update_weights(mlp, self.w_new)

        """
        Calcolo la funzione nel nuovo punto
        """

        E_new, gradE_new = evaluate_function(mlp, X, T, self.w_new)
        mlp.errors_tr.append(E_new)

        mee = compute_Regr_MEE(T, mlp.Out_o)
        mlp.errors_mee_tr.append(mee)
        self.gradE_new = gradE_new

        self.norm_gradE = np.linalg.norm(self.gradE_new)

        print("First Error ", E)
        print("New Error ", E_new)
        print("New gradient ", self.norm_gradE / self.norm_gradE_0)
        converged = self.norm_gradE < self.epsilon_prime

        """
        Calcolo s, y
        """
        s = self.w_new - self.w_prec
        y = self.gradE_new - self.gradE_prec

        self.s_y_list.append((s, y))
        return converged

    """
    Calcola H0 con la formula descritta da Nocedal
    """

    def compute_H0(self):

        # Prendo s,y più recenti
        s = self.s_y_list[-1][0]
        y = self.s_y_list[-1][1]

        den = np.dot(y.T, y)

        """"
        # Pongo un limite inferiore al valore del denominatore
        if den < 1e-16:
            print("H0 con delta!")
            #self.num_delta += 1
            self.H0 = self.delta * np.eye(self.H0.shape[0])
        """

        # print(float(np.dot(s.T,y)))
        num = np.dot(s.T, y)
        assert num > 0, "Curvature condition non è verificata!!!"
        gamma = float(num / den)

        self.H0 = gamma * np.eye(self.H0.shape[0])
        # print("gamma= ",gamma)
        return

    def find_direction(self):

        q = self.gradE_new
        alpha_bfgs_list = []
        rho_bfgs_list = []

        for sy in reversed(self.s_y_list):

            s_i = sy[0]
            y_i = sy[1]
            den = np.dot(y_i.T, s_i)
            assert den > 0, "Curvature condition non è verificata!!!"
            if den < 1e-16:
                print("=(")
                den = 1e-16

            rho_i = 1 / den
            rho_bfgs_list.append(rho_i)

            alpha_i = rho_i * np.dot(s_i.T, q)
            alpha_bfgs_list.append(alpha_i)

            q = q - alpha_i * y_i

        r = np.dot(self.H0, q)

        alpha_bfgs_list.reverse()
        rho_bfgs_list.reverse()

        for (i, sy) in enumerate(self.s_y_list):
            s_i = sy[0]
            y_i = sy[1]

            rho_i = rho_bfgs_list[i]
            alpha_i = alpha_bfgs_list[i]

            beta = rho_i * np.dot(y_i.T, r)
            r = r + s_i * (alpha_i - beta)

        return r

    def train(self, mlp, X, T, X_val, T_val, n_epochs=1000, eps=10 ^ (-3), threshold=0.5, suppress_print=False):

        assert n_epochs > 0
        assert eps > 0

        self.max_iter_AWLS_train *= n_epochs

        epoch = 0

        done_max_epochs = False  # Fatte numero massimo iterazioni
        found_optimum = False  # Gradiente minore o uguale a eps_prime
        numerical_problems = False  # rho oppure gamma problem
        done_max_AWLS_iters_train = False  # terminato il numero massimo di iterazioni complessive di AWLS

        while (not done_max_epochs) and (not done_max_AWLS_iters_train) and (not found_optimum) and (
        not numerical_problems):

            if epoch == 0:

                converged = self.do_first_iteration(mlp, X, T, X_val, T_val, eps, threshold)
                if converged:
                    found_optimum = True

                else:
                    epoch += 1
                    if epoch >= n_epochs:
                        done_max_epochs = True

                    if self.n_iters_AWLS_train >= self.max_iter_AWLS_train:
                        done_max_AWLS_iters_train = True

            else:

                self.compute_H0()

                d = - self.find_direction()

                mlp.eta, it_AWLS = AWLS(mlp, X, T, d, mlp.lambd, self.eta_start, self.eta_max,
                                        self.max_iter_AWLS_train,
                                        self.m1, self.m2,
                                        self.tau, self.mina, self.sfgrd, l_bfgs=True, debug=False,
                                        epsilon=self.epsilon_prime)

                self.it_AWLS_list.append(it_AWLS)
                self.n_iters_AWLS_train += it_AWLS

                """
                Aggiorno i pesi
                """

                w_new = self.w_new + mlp.eta * d

                update_weights(mlp, w_new)

                self.w_prec = self.w_new
                self.w_new = w_new

                """
                Calcolo funzione e gradiente nel nuovo punto
                """

                E, gradE = evaluate_function(mlp, X, T, self.w_new)
                mlp.errors_tr.append(E)
                mee = compute_Regr_MEE(T, mlp.Out_o)
                mlp.errors_mee_tr.append(mee)
                self.norm_gradE = np.linalg.norm(gradE)
                print("Iterazione %s) Eta = %s New Error %s New Gradient %s" % (
                    epoch, mlp.eta, E, self.norm_gradE / self.norm_gradE_0))
                self.gradE_prec = self.gradE_new
                self.gradE_new = gradE

                """
                Calcolo s, y
                """

                s = self.w_new - self.w_prec
                y = self.gradE_new - self.gradE_prec

                self.s_y_list.append((s, y))
                if len(self.s_y_list) > self.m:
                    self.s_y_list.pop(0)

                epoch += 1
                if epoch >= n_epochs:
                    done_max_epochs = True

                if self.n_iters_AWLS_train >= self.max_iter_AWLS_train:
                    done_max_AWLS_iters_train = True

                if self.norm_gradE < self.epsilon_prime:
                    found_optimum = True

        if found_optimum:
            print("Trovato ottimo")

        elif done_max_epochs:
            print("Fatto il numero massimo di epoche")


        elif done_max_AWLS_iters_train:
            print("Terminato per numero massimo di iterazioni totali di AWLS..")

        # print("num_delta= ", self.num_delta)

        return len(mlp.errors_tr)
