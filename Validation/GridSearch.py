import sys
sys.path.append("../")
from Utilities.Utility import *
import numpy as np
from MLP.MLP import *
from Trainers.TrainBackprop import *
import math

"""
Per multiple minima: si esegueguono piu trials
- Class/Regress
"""

"""
Esegue il train del modello n_trials volte, ogni volta partendo da un punto diverso.
Restituisce informazioni statistiche sul processo effettuato, quali ad esempio MSE medio e varianza, rispetto
al numero di trials effettuati.
Restituisce inoltre il modello con errore di generalizzazione minore, tra tutti quelli utilizzati durante
i vari trials.

    :param n_feature : Numero di features
    :param X_tr : Matrice contenente i dati di training
    :param T_tr: Matrice con i target dei dati di training
    :param X_val Matrice contenente i dati di validation
    :param T_val: Matrice con i target dei dati di validation
    :param n_epochs : Numero massimo di epoche
    :param hidden_act : Funzione di attivazione dell'hidden layer
    :param output_act : Funzione di attivazione dell'output layer
    :param eta : Learning rate
    :param alfa : Momentum
    :param n_hidden : Numero neuroni nell'hidden layer
    :param weight: Indica che i pesi sono inizializzati in [-weight, weight]
    :param lambd : Penalty term
    :param n_trials : Indica il numero di trials da effettuare
    :param classification: Se true, mlp deve risolvere un problema di classificazione(usato per sapere se
                            riempire lista di accuracy o di MEE)
    :param trainer: Indica quale algoritmo di ottimizzazione usare per la fase di training
    :param eps : Valore per indicare accuratezza massima desiderata. Usata solo per BPLS E L-BFGS
    
    :return best_mlp: Modello con migliore capacità di generalizzazione
    :return mean_err_tr: lista dei TR MSE medi (un elemento per epoca)
    :return std_err_tr: lista della deviazione standare di TR MSE
    :return mean_error_MEE_tr: lista di TR MEE medio 
    :return std_error_MEE_tr : lista di deviazione standard di TR MEE 
    :return mean_err_vl: lista di VL MSE medio 
    :return std_err_vl: lista di deviazione standard di VL MSE
    :return mean_error_MEE_vl: lista di VL MEE medio 
    :return std_error_MEE_vl: lista di std di VL MEE
    :return avg_epochs_done: Numero medio di epoche effettuate
"""

def run_trials(n_features, X_tr, T_tr, X_vl, T_vl, n_epochs, hidden_act, output_act,
               eta, alfa, n_hidden, weight, lambd,
               n_trials, classification=True,trainer=TrainBackprop(),eps=1e-7):
    best_vl_error = 1e10
    best_mlp = None
    best_idx = -1

    # Classificazione/Regressione:
    errors_tr = np.zeros((n_trials, n_epochs+1))
    errors_vl = np.zeros((n_trials, n_epochs+1))
    if classification:
        acc_tr = np.zeros((n_trials, n_epochs+1))
        acc_vl = np.zeros((n_trials, n_epochs+1))
    else:
        errors_MEE_tr = np.zeros((n_trials, n_epochs+1))
        errors_MEE_vl = np.zeros((n_trials, n_epochs+1))

    avg_epochs_done = 0
    for trial in range(n_trials):

        #print(100 * '-')
        #print("Trial %s/%s: " % (trial + 1, n_trials))
        mlp = MLP(n_features, n_hidden, T_tr.shape[1], hidden_act, output_act, eta=eta, alfa=alfa, lambd=lambd,
                  fan_in_h=True, range_start_h=-weight, range_end_h=weight,
                  classification=classification, trainer=trainer)

        epoch_done=mlp.trainer.train(mlp,addBias(X_tr), T_tr, addBias(X_vl), T_vl, n_epochs, eps, suppress_print=True)
        avg_epochs_done += epoch_done

        # Classificazione/Regressione:
        errors_tr[trial, :epoch_done] = mlp.errors_tr
        errors_vl[trial, :epoch_done] = mlp.errors_vl
        errors_tr[trial, epoch_done:] = mlp.errors_tr[-1]
        errors_vl[trial, epoch_done:] = mlp.errors_vl[-1]
        if classification:
            acc_tr[trial,epoch_done:] = mlp.accuracies_tr[-1]
            acc_vl[trial,epoch_done:] = mlp.accuracies_vl[-1]
            acc_tr[trial, :epoch_done] = mlp.accuracies_tr
            acc_vl[trial, :epoch_done] = mlp.accuracies_vl
        else:
            errors_MEE_tr[trial, :epoch_done] = mlp.errors_mee_tr
            errors_MEE_vl[trial, :epoch_done] = mlp.errors_mee_vl
            errors_MEE_tr[trial,epoch_done:] = mlp.errors_mee_tr[-1]
            errors_MEE_vl[trial,epoch_done:] = mlp.errors_mee_vl[-1]

        # Se il migliore => prendo lui come modello (vedi slide matematica Validation part 2)
        ## Classificazione/Regressione:
        if classification:  # class==MONK; uso MSE
            if best_mlp is None:
                best_mlp = mlp
                best_vl_error = mlp.errors_vl[-1]  # Vedi slide Multipla minima (min sull'error vl MSE)
                best_idx = trial + 1
            elif mlp.errors_vl[-1] < best_vl_error:
                #print("\nTROVATO ERRORE MIGLIORE = %s -> %s\n" % (best_vl_error, mlp.errors_vl[-1]))
                best_mlp = mlp
                best_vl_error = mlp.errors_vl[-1]  # Vedi slide Multipla minima (min sull'error vl MSE)
                best_idx = trial + 1
        else:  # regressione==CUP; uso MEE
            if best_mlp is None:
                best_mlp = mlp
                best_vl_error = mlp.errors_mee_vl[-1]  # MEE
                best_idx = trial + 1
            elif mlp.errors_vl[-1] < best_vl_error:
                #print("\nTROVATO ERRORE MIGLIORE = %s -> %s\n" % (best_vl_error, mlp.errors_vl[-1]))
                best_mlp = mlp
                best_vl_error = mlp.errors_mee_vl[-1]  # MEE
                best_idx = trial + 1

    # Per avere una stima dell'error TR/VL (MSE) e dell'accuracy TR/VL se class,else MEE TR/VL se regressione
    ## Nota: Fondamentale stime(MSE)+LearningCurve => elementi di validazione nella fase di progettazione!
    mean_err_tr = np.mean(errors_tr, axis=0, keepdims=True).T  # Media
    std_err_tr = np.std(errors_tr, axis=0, keepdims=True).T  # sqm (radice varianza)
    mean_err_vl = np.mean(errors_vl, axis=0, keepdims=True).T  # Media
    std_err_vl = np.std(errors_vl, axis=0, keepdims=True).T  # sqm (radice varianza)

    if classification:
        mean_acc_tr = np.mean(acc_tr, axis=0, keepdims=True).T  # Media
        std_acc_tr = np.std(acc_tr, axis=0, keepdims=True).T  # sqm (radice varianza)
        mean_acc_vl = np.mean(acc_vl, axis=0, keepdims=True).T  # Media
        std_acc_vl = np.std(acc_vl, axis=0, keepdims=True).T  # sqm (radice varianza)
    else:
        mean_error_MEE_tr = np.mean(errors_MEE_tr, axis=0, keepdims=True).T  # Media
        std_error_MEE_tr = np.std(errors_MEE_tr, axis=0, keepdims=True).T  # sqm (radice varianza)
        mean_error_MEE_vl = np.mean(errors_MEE_vl, axis=0, keepdims=True).T  # Media
        std_error_MEE_vl = np.std(errors_MEE_vl, axis=0, keepdims=True).T  # sqm (radice varianza)

    avg_epochs_done = math.ceil(avg_epochs_done/n_trials)

    #print(100 * "-")
    #print("Returning model number ", best_idx)
    #print(100 * "-")
    #print("STATISTICS:")
    if classification:
        """
        print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f" % (
            mean_err_tr[-1], std_err_tr[-1], mean_acc_tr[-1],
            std_acc_tr[-1], mean_err_vl[-1], std_err_vl[-1],
            mean_acc_vl[-1], std_acc_vl[-1]))
            """

    else:
        print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f\nEpochs = %s" % (
            mean_err_tr[-1], std_err_tr[-1], mean_error_MEE_tr[-1],
            std_error_MEE_tr[-1], mean_err_vl[-1], std_err_vl[-1],
            mean_error_MEE_vl[-1], std_error_MEE_vl[-1],avg_epochs_done))

    #print(100 * "-")
    #print("\n")
    if classification:
        return best_mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl
    else:
        return best_mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl, std_error_MEE_vl,\
               avg_epochs_done
