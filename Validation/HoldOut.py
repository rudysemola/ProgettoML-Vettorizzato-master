import sys
sys.path.append("../")
from Validation.GridSearch import *


"""
Effettua la HOLD OUT.
Restituisce la configurazione migliore degli iperparametri.

NOTA1: NON EFFETTUA IL RETRAINING FINALE SULL'INTERO (TR+VL) set
NOTA2: NON EFFETTUA LO SPLITTING TRA I DATI INTERNI (TR/VL) E TEST(INTERNO), MA SOLO QUELLO TRA TR/VL
ENTRAMBI FATTI SUI FILE "FINALI" (vedi file test_kfoldCV.py e test_HoldOut.py)

:param X: Matrice di input
:param T: Matrice di target
:param n_epochs : numero di epoche di training
:param hidden_act: Funzione attivazione hidden layer
:param output_act: Funzione attivazione output layer
:param eta_values: Insieme di valori da provare per eta
:param alfa_values: Insieme di valori da provare per alfa
:param hidden_values: Insieme di valori da provare per il numero di hidden units
:param weight_values: Insieme di valori da provare per l'intervallo di inizializzazione dei pesi
:param lambda_values: Insieme di valori da provare per lambda
:param n_trials : Numero di volte che viene effettuato il train (multiple minima)

:return best_eta,best_alfa,best_hidden,best_lambda,best_weight : migliore configurazione trovata
:return best_mean_vl_error : media del miglior validation error trovato
    - Classificazione => uso MSE
    - Regressione => uso MEE

REMEMBER=> Hyperparameter (esaustiva):
    # eta
    # alfa
    # hidden (in generale numero neuroni)
    # weight (Initialize weights by random values near zero)
    # lambda
    # on-line/batch/miniBatch !
    # stopping criteria (piu per CM...)
    # n_trials per multipla minima (forzatura)
    # n_epochs non fisso (forzatura)
    # k della kcrossvalidation (forzatura)

"""


def do_HoldOut(n_features, X, T, n_epochs, hidden_act, output_act, eta_values, alfa_values, hidden_values,
               weight_values, lambda_values, n_trials, classifications=True):

    X_tr, T_tr, X_vl, T_vl = split_data_train_validation(X, T)

    best_mean_vl_error = 1e10
    best_std_vl_error = 0

    # Hyperparameter (esaustiva):
    best_eta = 0  # PRIMA GRID SEARCH
    best_alfa = 0  # PRIMA GRID SEARCH
    best_hidden = 0
    best_weight = 0
    best_lambda = 0  # PRIMA GRID SEARCH
    # on-line/batch/miniBatch !
    # stopping criteria (piu per CM...)
    # n_trials per multipla minima (forzatura)
    # n_epochs non fisso (forzatura)

    # Classificazione:
    if classifications:

        """
        PER OGNI CONFIGURAZIONE...
        """
        for eta in eta_values:
            for alfa in alfa_values:
                for hidden in hidden_values:  # FISSIAMO PER ORA
                    for weight in weight_values:  # FISSIAMO PER ORA
                        for lambd in lambda_values:
                            print(100 * '-')
                            print("Provo eta=%s alfa=%s #hidden=%s weight=%s lambda = %s" % (
                                eta, alfa, hidden, weight, lambd))

                            """
                            Seleziono modello che min [errore_(MSE)_vl]
                            """
                            mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl = run_trials(
                                n_features, X_tr, T_tr, X_vl, T_vl, n_epochs, hidden_act, output_act, eta, alfa,
                                hidden, weight, lambd, n_trials, classifications)

                            """
                            Prendo e stimo  VL error medio  per questa configurazione
                            """
                            mean_vl_error = mean_err_vl[-1]
                            print("VL ERROR: %3f" % mean_vl_error)

                            """
                            Controllo se VL error medio ottenuto e il migliore al momento.
                            """
                            if mean_vl_error < best_mean_vl_error:
                                print(
                                    "\nTROVATO ERRORE MIGLIORE = %3f -> %3f\n" % (
                                        best_mean_vl_error, mean_vl_error))
                                best_mean_vl_error = mean_vl_error

                                best_eta = eta
                                best_alfa = alfa
                                best_hidden = hidden
                                best_lambda = lambd
                                best_weight = weight

    # Regressione:
    else:
        """
        PER OGNI CONFIGURAZIONE...
        """
        for eta in eta_values:
            for alfa in alfa_values:
                for hidden in hidden_values:  # FISSIAMO PER ORA
                    for weight in weight_values:  # FISSIAMO PER ORA
                        for lambd in lambda_values:
                            print(100 * '-')
                            print("Provo eta=%s alfa=%s #hidden=%s weight=%s lambda = %s" % (
                                eta, alfa, hidden, weight, lambd))

                            """
                            Seleziono modello che min [errore_(MEE)_vl]
                            """
                            mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl, std_error_MEE_vl = run_trials(
                                n_features, X_tr, T_tr, X_vl, T_vl, n_epochs, hidden_act, output_act, eta, alfa,
                                hidden, weight, lambd, n_trials, classifications)

                            """
                            Prendo e stimo  VL error medio  per questa configurazione
                            """
                            mean_vl_error = mean_error_MEE_vl[-1]
                            print("VL ERROR: %3f" % mean_vl_error)

                            """
                            Controllo se VL error medio ottenuto e il migliore al momento.
                            """
                            if mean_vl_error < best_mean_vl_error:
                                print(
                                    "\nTROVATO ERRORE MIGLIORE = %3f -> %3f\n" % (
                                        best_mean_vl_error, mean_vl_error))
                                best_mean_vl_error = mean_vl_error

                                best_eta = eta
                                best_alfa = alfa
                                best_hidden = hidden
                                best_lambda = lambd
                                best_weight = weight

    print()
    print(100 * "-")
    print("CONFIGURAZIONE SCELTA eta=%s alfa=%s #hidden=%s weight=%s lambda=%s" % (
        best_eta, best_alfa, best_hidden, best_weight, best_lambda))
    print("BEST VL ERROR: %3f +- %3f" % (best_mean_vl_error, best_std_vl_error))
    print(100 * "-")
    print("FINE HOLD_OUT")
    print(100 * "-")
    print()
    return best_eta, best_alfa, best_hidden, best_lambda, best_weight, best_mean_vl_error



