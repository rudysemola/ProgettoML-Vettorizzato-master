"""
Questo file contiene l'implementazione del metodo del Quasi Newton a memoria limitata.
L-BFGS
"""
import sys
sys.path.append("../")

from Trainers.Training import *
import numpy as np
from Utilities.UtilityCM import *
from Utilities.Utility import *
from Trainers.LineSearch_old import *


class L_BFGS(Training):

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
    def __init__(self,eta_start=1,eta_max=2,max_iter_AWLS_train=100,m1=0.0001,m2=0.9,tau=0.9,sfgrd = 0.001,mina=1e-16,m=3, delta =0.01,use_delta=True):

        #PARAMETRI LS
        self.eta_start = eta_start # L-BFGS vuole AWLS=> passare alpha=1
        self.eta_max = eta_max
        self.max_iter_AWLS_train = max_iter_AWLS_train
        self.m1 = m1
        self.m2 = m2
        self.tau = tau
        self.sfgrd = sfgrd
        self.mina = mina

        # PARAMETRI BFGS
        self.m = m
        self.s_y_list= []
        self.H0 = None
        self.delta = delta
        self.use_delta = use_delta

        self.it_AWLS_list = []
    """
    Calcolo H0 per calcolo direzione e per L-BFGS
    """
    def compute_H0(self):

        if not self.use_delta:
            sy = self.s_y_list[-1] #quello più recente
            s = sy[0]
            y = sy[1]
            gamma = (s.T * y ) /(y.T * y)
            self.H0 = gamma * np.eye(y.shape[0])
        else:
            self.H0 = self.delta * np.eye(self.H0.shape[0])


    """
    L-BFGS two-loop recursion (vedi libro riferimento)
    Calcola la direzione da usare
    """
    def compute_direction(self,gradE_vec):
        #Liste da mettere ed init
        alpha_bfgs_list = []
        rho_list = []
        q = gradE_vec
        #primo loop
        try:
            for sy in reversed(self.s_y_list):
                s = sy[0]
                y = sy[1]
                #print("s= %s,\ny=%s"%(s[0],y[0]))

                dot_p = np.dot(y.T, s)
                if abs(dot_p) < 1e-10:
                    if dot_p < -(1e-10):
                        dot_p = -(1e-10)
                    else:
                        dot_p = (1e-10)


                rho = float(1/dot_p)
                #print(rho)
                #print("Rho = ",rho)
                rho_list.append(rho)

                alpha_bfgs = float(rho*(np.dot(s.T,q)))
                #print("Alpha_BFGS = ", alpha_bfgs)
                alpha_bfgs_list.append(alpha_bfgs)

                q = q - (alpha_bfgs*y)

            # calcolo r (vedi libro riferimento) e reverso le liste
            r = np.dot(self.H0 ,q) # q ultima iterazione
            rho_list.reverse()
            alpha_bfgs_list.reverse()

                #secondo loop
            for (idx,sy) in enumerate(self.s_y_list):
                s = sy[0]
                y = sy[1]
                rho = rho_list[idx]
                beta = rho*(np.dot(y.T, r))
                alpha_bfgs = alpha_bfgs_list[idx]
                r = r + s*(alpha_bfgs - beta)


            return r

        except:
            print("NUMPROBLEM")
            return None


    def train(self,mlp,X, T, X_val, T_val, n_epochs = 1000, eps = 10 ^ (-3), threshold = 0.5, suppress_print = False):

        assert n_epochs > 0
        assert eps > 0

        self.max_iter_AWLS_train *= n_epochs

        epoch = 0
        n_iters_AWLS_train = 0
        norm_gradE_0 = 0.
        eps_prime = 0.
        norm_gradE = 0.
        E = 0.
        gradE_h = None
        gradE_o = None
        gradE_vec_old = None

        done_max_epochs = False  # Fatte numero massimo iterazioni
        found_optimum = False  # Gradiente minore o uguale a eps_prime
        numerical_problems = False # rho oppure gamma problem
        done_max_AWLS_iters_train = False #terminato il numero massimo di iterazioni complessive di AWLS

        while (not done_max_epochs) and (not done_max_AWLS_iters_train)and (not found_optimum) and (not numerical_problems):

            if epoch == 0:
                E = compute_obj_function(mlp, X, T, mlp.lambd)

                if mlp.classification:
                    accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
                    mlp.errors_tr.append(E)
                    mlp.accuracies_tr.append(accuracy)
                else:
                    error_MEE = compute_Regr_MEE(T, mlp.Out_o)
                    mlp.errors_tr.append(E)
                    mlp.errors_mee_tr.append(error_MEE)

                gradE_h, gradE_o = compute_gradient(mlp, X, T, mlp.lambd)

                norm_gradE_0 = np.linalg.norm(gradE_h) ** 2 + np.linalg.norm(gradE_o) ** 2

                eps_prime = eps * norm_gradE_0
                norm_gradE = norm_gradE_0

                mlp.gradients.append(norm_gradE / norm_gradE_0)

                #METTO PESI E GRADIENTI COME VETTORE
                w_vec = matrix2vec(mlp.W_h,mlp.W_o)
                #print("w_vec ",w_vec.shape)
                gradE_vec_new = matrix2vec(gradE_h,gradE_o)

                self.H0 = np.eye(gradE_vec_new.shape[0])

                #APPENDO ALLA LISTA S_Y
                #self.s_y_list.append((np.zeros(w_vec.shape),np.zeros(gradE_vec_new.shape)))

            else:
                # Calcolo funzione, gradiente, (s,y)
                E = compute_obj_function(mlp, X, T, mlp.lambd)

                gradE_vec_old = gradE_vec_new

                gradE_h, gradE_o = compute_gradient(mlp, X, T, mlp.lambd)
                norm_gradE = np.linalg.norm(gradE_h) ** 2 + np.linalg.norm(gradE_o) ** 2

                gradE_vec_new = matrix2vec(gradE_h, gradE_o)
                # Servono per calcolo s = w(i+1) - w(i) e per calcolo y = gradE_vec(i+1) - gradE_vec(i)
                mlp.dW_o_old = dW_o_new
                mlp.dW_h_old = dW_h_new

                dW_vec = matrix2vec(mlp.dW_h_old, mlp.dW_o_old)

                # Aggiorno s e y
                s = dW_vec
                y = gradE_vec_new - gradE_vec_old

                #Aggiungo (s,y) alla lista.
                #Se ho più di m elementi nella lista_s_y, tolgo il primo elemento....
                self.s_y_list.append((s,y))
                if len(self.s_y_list) > self.m:
                    self.s_y_list.pop(0)

                if mlp.classification:
                    accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
                    mlp.errors_tr.append(E)
                    mlp.accuracies_tr.append(accuracy)
                else:
                    error_MEE = compute_Regr_MEE(T, mlp.Out_o)
                    mlp.errors_tr.append(E)
                    mlp.errors_mee_tr.append(error_MEE)

                mlp.gradients.append(norm_gradE / norm_gradE_0)

            # CALCOLO IL VALIDATION ERROR
            error_MSE_val = compute_obj_function(mlp, X_val, T_val, mlp.lambd)

            if mlp.classification:
                accuracy_val = compute_Accuracy_Class(T_val, convert2binary_class(mlp.Out_o, threshold))
                mlp.errors_vl.append(error_MSE_val)
                mlp.accuracies_vl.append(accuracy_val)
            else:
                error_MEE_val = compute_Regr_MEE(T_val, mlp.Out_o)
                mlp.errors_vl.append(error_MSE_val)
                mlp.errors_mee_vl.append(error_MEE_val)

            # CONTROLLO GRADIENTE
            if norm_gradE < eps_prime:
                found_optimum = True

            if not found_optimum:

                # CALCOLO H0
                H0 = self.compute_H0()

                # Calcolo direzione di = - ([approx_H]^-1)* gradE_vec // notare il meno!
                di = self.compute_direction(gradE_vec_new)

                if di is None:
                    numerical_problems = True

                if not numerical_problems:
                    di = -di
                    # LINE_SEARCH
                    mlp.eta, it_AWLS = AWLS(mlp, X, T, E, gradE_h, gradE_o, mlp.lambd, self.eta_start, self.eta_max, self.max_iter_AWLS_train,
                                   self.m1, self.m2,
                                   self.tau, self.mina, self.sfgrd,l_bfgs=True)

                    self.it_AWLS_list.append(it_AWLS)
                    n_iters_AWLS_train += it_AWLS

                    # print("Epoca %s) Eta = %s"%(epoch+1,mlp.eta))

                    # AGGIORNAMENTO: Warning!!!
                    ##NOTA: direzione gia col meno!
                    di_h, di_o = vec2matrix(di, mlp.W_h.shape, mlp.W_o.shape)
                    dW_o_new = mlp.eta * di_o
                    mlp.W_o = mlp.W_o + dW_o_new

                    dW_h_new = mlp.eta * di_h
                    mlp.W_h = mlp.W_h + dW_h_new



                    # per stampa per ogni epoca
                    if not suppress_print:
                        if mlp.classification:
                            print(
                                "Epoch %s/%s) Eta = %s ||gradE||/ ||gradE_0|| = %s,TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                                    epoch + 1, n_epochs, mlp.eta, norm_gradE / norm_gradE_0, E, error_MSE_val, accuracy,
                                    accuracy_val))
                        else:
                            print(
                                "Epoch %s/%s) Eta = %s ||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL ((MEE) : %s" % (
                                    epoch + 1, n_epochs, mlp.eta, norm_gradE / norm_gradE_0, E, error_MSE_val, error_MEE,
                                    error_MEE_val))

                    epoch += 1

                    # CONTROLLO EPOCHE
                    if epoch >= n_epochs:
                        done_max_epochs = True

                    if n_iters_AWLS_train >= self.max_iter_AWLS_train:

                        done_max_AWLS_iters_train = True


        # CALCOLO ERRORE DOPO L'ULTIMO AGGIORNAMENTO

        E = compute_obj_function(mlp, X, T, mlp.lambd)

        if mlp.classification:
            accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
            mlp.errors_tr.append(E)
            mlp.accuracies_tr.append(accuracy)
        else:
            error_MEE = compute_Regr_MEE(T, mlp.Out_o)
            mlp.errors_tr.append(E)
            mlp.errors_mee_tr.append(error_MEE)

        error_MSE_val = compute_obj_function(mlp, X_val, T_val, mlp.lambd)

        if mlp.classification:
            accuracy_val = compute_Accuracy_Class(T_val, convert2binary_class(mlp.Out_o, threshold))
            mlp.errors_vl.append(error_MSE_val)
            mlp.accuracies_vl.append(accuracy_val)

        else:
            error_MEE_val = compute_Regr_MEE(T_val, mlp.Out_o)
            mlp.errors_vl.append(error_MSE_val)
            mlp.errors_mee_vl.append(error_MEE_val)

        # per stampa di risultato finale
        if suppress_print:
            if mlp.classification:
                print(
                    "Final Results: ||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                        norm_gradE / norm_gradE_0, mlp.errors_tr[-1], mlp.errors_vl[-1], mlp.accuracies_tr[-1],
                        mlp.accuracies_vl[-1]))
            else:
                print(
                    "Final Results:||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL (MEE) : %s Epochs: %s" % (
                        norm_gradE / norm_gradE_0, mlp.errors_tr[-1], mlp.errors_vl[-1], mlp.errors_mee_tr[-1],
                        mlp.errors_mee_vl[-1],len(mlp.errors_tr)))
        """
        if found_optimum:
            vettore_hidden = np.reshape(mlp.W_h, (-1, 1))
            vettore_out = np.reshape(mlp.W_o, (-1, 1))
            vettore_finale = np.concatenate((vettore_hidden, vettore_out), axis=0)
            print("TROVATO OTTIMO:\nE = %3f\nnorma gradE/gradE_0=%s, W_star=\n%s" % (
            E, norm_gradE / norm_gradE_0, vettore_finale.T))

        elif done_max_epochs:

            print("Terminato per numero massimo di iterazioni")
            vettore_hidden = np.reshape(mlp.W_h, (-1, 1))
            vettore_out = np.reshape(mlp.W_o, (-1, 1))
            vettore_finale = np.concatenate((vettore_hidden, vettore_out), axis=0)
            print("VALORI FINALI(NON OTTIMI):\nE = %3f\nnorma gradE/gradE_0 =%s\nW_star=\n%s" % (
                E, norm_gradE / norm_gradE_0, vettore_finale.T))

        elif done_max_AWLS_iters_train:
            print("Terminato per numero massimo di iterazioni totali di AWLS..")"""
        if numerical_problems:
            print("Ci sono stati problemi...")

        return len(mlp.errors_tr)