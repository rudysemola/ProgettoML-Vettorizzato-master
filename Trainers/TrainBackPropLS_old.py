import sys
sys.path.append("../")
from Utilities.UtilityCM2 import *
from Trainers.Training import *
from Trainers.LineSearch_new import *


import numpy as np

"""
Effettua training con SGD con momentum e Line Search
"""
class TrainBackPropLS(Training):
    """
       Inizializza gli iperparametri algoritmici
       :param eta_start: Eta iniziale da provare durante la AWLS
       :param eta_max: Eta massimo accettabile
       :param max_iter: Numero massimo di iterazioni che AWLS può compiere
       :param m1: Parametro della condizione di Armijo
       :param m2: Parametro della condizione di Wolfe
       :param tau : Parametro che indica di quanto devo incrementare eta nella prima fase di AWLS
       :param sfgrd: Valore della safeguard. Serve ad assicurare che durante la fase di interpolazione,
               l'ampiezza dell'intervallo diminuisca almeno di un fattore pari a sfgrd
       :param mina: Indica quanto deve essere l'ampiezza minima dell'intervallo in cui fare interpolazione
       """
    def __init__(self,eta_start=0.1,eta_max=2,max_iter=100,m1=0.0001,m2=0.9,tau=0.9,sfgrd = 0.001,mina=1e-16):
        self.eta_start = eta_start
        self.eta_max = eta_max
        self.max_iter = max_iter
        self.m1 = m1
        self.m2 = m2
        self.tau = tau
        self.sfgrd = sfgrd
        self.mina = mina
        self.it_AWLS_list = []

    def train(self,mlp,X, T, X_val, T_val, n_epochs = 1000, eps = 1e-12, threshold = 0.5, suppress_print = False):

        assert n_epochs > 0
        assert eps > 0

        epoch = 0
        norm_gradE_0 = 0.
        eps_prime = 0.
        norm_gradE = 0.
        E = 0.
        gradE_h = None
        gradE_o = None
        done_max_epochs = False #Fatte numero massimo iterazioni
        found_optimum = False #Gradiente minore o uguale a eps_prime

        while (not done_max_epochs) and (not found_optimum):

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

            else:
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
                norm_gradE = np.linalg.norm(gradE_h) ** 2 + np.linalg.norm(gradE_o) ** 2

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

            #CONTROLLO GRADIENTE
            if norm_gradE < eps_prime:
                found_optimum = True

            if not found_optimum:
                #LINE_SEARCH
                mlp.eta,it_AWLS= AWLS(mlp,X,T,E,gradE_h,gradE_o,mlp.lambd,self.eta_start,self.eta_max,self.max_iter,self.m1,self.m2,
                               self.tau,self.mina,self.sfgrd,l_bfgs=False)

                self.it_AWLS_list.append(it_AWLS)

                #print("Epoca %s) Eta = %s"%(epoch+1,mlp.eta))

                #AGGIORNAMENTO
                dW_o_new = -mlp.eta * gradE_o + mlp.alfa * mlp.dW_o_old
                mlp.W_o = mlp.W_o + dW_o_new

                dW_h_new = -mlp.eta * gradE_h + mlp.alfa * mlp.dW_h_old
                mlp.W_h = mlp.W_h + dW_h_new

                mlp.dW_o_old = dW_o_new
                mlp.dW_h_old = dW_h_new

                # per stampa per ogni epoca
                if not suppress_print:
                    if mlp.classification:
                        print(
                            "Epoch %s/%s) Eta = %s ||gradE||/ ||gradE_0|| = %s,TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                                epoch + 1,n_epochs, mlp.eta,norm_gradE/norm_gradE_0, E, error_MSE_val, accuracy, accuracy_val))
                    else:
                        print(
                            "Epoch %s/%s) Eta = %s ||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL ((MEE) : %s" % (
                                epoch + 1, n_epochs,mlp.eta,norm_gradE/norm_gradE_0,E, error_MSE_val, error_MEE, error_MEE_val))

                epoch += 1
                #CONTROLLO EPOCHE
                if epoch >= n_epochs:
                    done_max_epochs = True


        #CALCOLO ERRORE DOPO L'ULTIMO AGGIORNAMENTO

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
                        norm_gradE / norm_gradE_0,mlp.errors_tr[-1], mlp.errors_vl[-1], mlp.accuracies_tr[-1], mlp.accuracies_vl[-1]))
            else:
                print(
                    "Final Results:||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL (MEE) : %s" % (
                        norm_gradE /norm_gradE_0,mlp.errors_tr[-1], mlp.errors_vl[-1], mlp.errors_mee_tr[-1], mlp.errors_mee_vl[-1]))
        """
        if found_optimum:
            vettore_hidden = np.reshape(mlp.W_h,(-1,1))
            vettore_out = np.reshape(mlp.W_o,(-1,1))
            vettore_finale = np.concatenate((vettore_hidden,vettore_out),axis=0)
            print("TROVATO OTTIMO:\nE = %3f\nnorma gradE/gradE_0=%s, W_star=\n%s"%(E,norm_gradE/norm_gradE_0,vettore_finale.T))

        elif done_max_epochs:

            print("Terminato per numero massimo di iterazioni")
            vettore_hidden = np.reshape(mlp.W_h, (-1, 1))
            vettore_out = np.reshape(mlp.W_o, (-1, 1))
            vettore_finale = np.concatenate((vettore_hidden, vettore_out), axis=0)
            print("VALORI FINALI(NON OTTIMI):\nE = %3f\nnorma gradE/gradE_0 =%s\nW_star=\n%s" % (
            E, norm_gradE / norm_gradE_0, vettore_finale.T))
        """
        return len(mlp.errors_tr)