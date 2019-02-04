import sys
sys.path.append("../")
from Validation.KFold import *
from Validation.GridSearch import *
import matplotlib.pyplot as plt
from Utilities.Utility import *
from Trainers.TrainBackprop import *
import math

"""
KFold per regressione
Per descrizione parametri vedi file KFoldCV
"""
def KFoldRegression(n_features, X, T, k, n_epochs, hidden_act, output_act, eta_values, alfa_values, hidden_values,
                    weight_values, lambda_values, n_trials, shuffle=True, title_plot = "ML CUP", save_path_plot="../Plots/cup",
                    save_path_results="../Results_CSV/cup",window_size = 1,trainer=TrainBackprop(),eps=1e-7):
    if shuffle:
        X, T = shuffle_matrices(X, T)

    folds = kFold(X, T, k)

    best_mean_vl_error = 1e10
    best_std_vl_error = 0

    # Hyperparameter (esaustiva):
    best_eta = 0
    best_alfa = 0
    best_hidden = 0
    best_weight = 0
    best_lambda = 0

    """
            PER OGNI CONFIGURAZIONE...
            """
    for eta in eta_values:
        for alfa in alfa_values:
            for hidden in hidden_values:
                for weight in weight_values:
                    for lambd in lambda_values:
                        print(100 * '-')
                        print("Provo eta=%s alfa=%s #hidden=%s weight=%s lambda = %s" % (
                            eta, alfa, hidden, weight, lambd))

                        avg_epochs = 0
                        """
                        Apro il file
                        """
                        with open("%s_eta_%s_alpha_%s_lambd_%s_hidd_%s_weight_%s_trials_%s_k_%s.csv" % (
                                save_path_results, eta, alfa, lambd, hidden, weight,n_trials,k), "w") as f:

                            """
                            Scrivo prime info su file
                            """
                            f.write("Metrica usata per model selection = MEE\n")
                            f.write("Eta = %s Alpha = %s Lambda = %s Numero hidden units = %s Weight initialization = [-%s,+%s] trials = %s k = %s\n" % (
                                    eta, alfa, lambd, hidden, weight, weight,n_trials,k))

                            """
                            PER OGNI FOLD: ...
                            """

                            """
                            Matrici di k elementi.
                            La riga i contiene la lista degli errori medi e std fatti durante il train, usando il fold i-esimo come vl set.
                            Servono per poter produrre le curve medie per ogni configurazione
                            """

                            mean_tr_err_matrix = np.zeros((k, n_epochs + 1))
                            mean_tr_err_MEE_matrix = np.zeros((k, n_epochs + 1))
                            mean_vl_err_matrix = np.zeros((k, n_epochs + 1))
                            mean_vl_err_MEE_matrix = np.zeros((k, n_epochs + 1))

                            for (idx, fold_for_vl) in enumerate(folds):
                                print("FOLD ", idx + 1)
                                X_tr, T_tr, X_vl, T_vl = split_dataset(X, T, folds, idx)

                                mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, \
                                std_err_vl, mean_error_MEE_vl, std_error_MEE_vl,epochs = run_trials(
                                    n_features, X_tr, T_tr, X_vl, T_vl, n_epochs, hidden_act, output_act, eta, alfa,
                                    hidden,
                                    weight, lambd, n_trials,classification=False,trainer=trainer,eps=eps)


                                print("FOLD %s: VL ERROR = %3f Epoche %s" % (idx + 1, mean_error_MEE_vl[-1],epochs))

                                avg_epochs += epochs
                                """
                                Scrivo risultato su fold idx
                                """
                                f.write(
                                    "FOLD %s: TR ERROR (MSE)= %3f +- %3f TR ERROR (MEE) = %3f +- %3f VL ERROR (MSE) = %3f +- %3f VL ERROR (MEE) = %3f +- %3f\n" % (
                                        idx + 1,
                                        mean_err_tr[-1], std_err_tr[-1], mean_error_MEE_tr[-1], std_error_MEE_tr[-1],
                                        mean_err_vl[-1], std_err_vl[-1], mean_error_MEE_vl[-1], std_error_MEE_vl[-1]
                                    ))

                                print(100 * "-")
                                print()

                                """
                                Riempio la riga idx-esima delle matrici che servono per il plot
                                """

                                mean_tr_err_matrix[idx] = mean_err_tr.T
                                mean_tr_err_MEE_matrix[idx] = mean_error_MEE_tr.T
                                mean_vl_err_matrix[idx] = mean_err_vl.T
                                mean_vl_err_MEE_matrix[idx] = mean_error_MEE_vl.T

                            """
                            COSTRUISCO LE LISTE DI MEDIA E VARIANZA DA USARE PER CREARE LA LEARNING CURVE
                            """
                            mean_err_tr_list = np.mean(mean_tr_err_matrix, axis=0, keepdims=True).T  # Media
                            std_err_tr_list = np.std(mean_tr_err_matrix, axis=0,
                                                     keepdims=True).T  # sqm (radice varianza)

                            mean_err_vl_list = np.mean(mean_vl_err_matrix, axis=0, keepdims=True).T  # Media
                            std_err_vl_list = np.std(mean_vl_err_matrix, axis=0,
                                                     keepdims=True).T  # sqm (radice varianza)

                            mean_err_tr_MEE_list = np.mean(mean_tr_err_MEE_matrix, axis=0, keepdims=True).T  # Media
                            std_err_tr_MEE_list = np.std(mean_tr_err_MEE_matrix, axis=0,
                                                         keepdims=True).T  # sqm (radice varianza)
                            mean_err_vl_MEE_list = np.mean(mean_vl_err_MEE_matrix, axis=0, keepdims=True).T  # Media
                            std_err_vl_MEE_list = np.std(mean_vl_err_MEE_matrix, axis=0,
                                                         keepdims=True).T  # sqm (radice varianza)

                            """
                            STAMPO INFORMAZIONI COMPLESSIVE RIGUARDO ALLA CONFIGURAZIONE
                            """
                            print(
                                "INFORMAZIONI COMPLESSIVE: TR MSE = %3f +- %3f TR MEE = %3f +- %3f VL MSE = %3f +- %3f VL MEE = %3f +- %3f" % (
                                    mean_err_tr_list[-1], std_err_tr_list[-1], mean_err_tr_MEE_list[-1],
                                    std_err_tr_MEE_list[-1],
                                    mean_err_vl_list[-1], std_err_vl_list[-1], mean_err_vl_MEE_list[-1],
                                    std_err_vl_MEE_list[-1]
                                ))


                            avg_epochs = math.ceil(avg_epochs/k)

                            """
                            Salvo info finali della configurazione sui vari fold
                            """
                            f.write(
                                "RISULTATO FINALE SUL FOLD: TR MSE = %3f +- %3f TR MEE = %3f +- %3f VL MSE = %3f +- %3f VL MEE = %3f +- %3f Epoche medie:%s\n" % (
                                    mean_err_tr_list[-1], std_err_tr_list[-1], mean_err_tr_MEE_list[-1], std_err_tr_MEE_list[-1],
                                    mean_err_vl_list[-1], std_err_vl_list[-1], mean_err_vl_MEE_list[-1],
                                    std_err_vl_MEE_list[-1],avg_epochs
                                ))



                            """
                            FACCIO LA LEARNING CURVE
                            """

                            fig = plt.figure()
                            """
                            st = plt.suptitle("%s\neta=%s alpha=%s lambda=%s n_hidden=%s weight=%s trials=%s k=%s" % (
                                title_plot, eta, alfa, lambd, hidden, weight,n_trials,k))
                            """
                            st = plt.suptitle("%s\nlambda=%s n_hidden=%s weight=%s trials=%s k=%s" % (
                                title_plot, lambd, hidden, weight, n_trials, k))
                            plt.subplot(2, 1, 1)
                            plt.plot(mean_err_tr_list, label='Training Error', ls="-")

                            plt.plot(mean_err_vl_list, label='Validation Error', ls="dashed")

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_tr_list - std_err_tr_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_tr_list + std_err_tr_list, n_epochs + 1, -1),
                                             color="b", alpha=0.2)

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_vl_list - std_err_vl_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_vl_list + std_err_vl_list, n_epochs + 1, -1),
                                             color="orange", alpha=0.2)

                            ylim_sup = mean_err_vl_list[-1] + window_size
                            ylim_inf = max([mean_err_vl_list[-1] - window_size, 0])

                            plt.ylim([ylim_inf,ylim_sup])
                            plt.ylabel('MSE')
                            plt.grid(True)
                            plt.xlabel('epoch')
                            plt.legend(loc='upper right', prop={'size': 12})

                            plt.subplot(2, 1, 2)
                            plt.plot(mean_err_tr_MEE_list, label='Training MEE', ls="-")

                            plt.plot(mean_err_vl_MEE_list, label='Validation MEE', ls="dashed")

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_tr_MEE_list - std_err_tr_MEE_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_tr_MEE_list + std_err_tr_MEE_list, n_epochs + 1, -1),
                                             color="b", alpha=0.2)

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_vl_MEE_list - std_err_vl_MEE_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_vl_MEE_list + std_err_vl_MEE_list, n_epochs + 1, -1),
                                             color="orange", alpha=0.2)

                            plt.ylabel('MEE')
                            plt.grid(True)


                            ylim_sup = mean_err_vl_MEE_list[-1] + window_size
                            ylim_inf = max([mean_err_vl_MEE_list[-1] - window_size,0])



                            plt.ylim([ylim_inf,ylim_sup])
                            plt.xlim([0,avg_epochs])
                            plt.xlabel('epoch')
                            plt.legend(loc='upper right', prop={'size': 12})
                            plt.subplots_adjust(hspace=0.5)

                            """
                            SALVO LA LEARNING CURVE SU FILE
                            """
                            plt.savefig("%s_eta_%s_alpha_%s_lambd_%s_hidd_%s_weight_%s.jpg" % (
                                save_path_plot, eta, alfa, lambd, hidden, weight))
                            # plt.show()
                            plt.close(fig)



                            """
                            FACCIO LA SECONDA CURVA (NON SCALATA!!!)
                            
                            """

                            fig = plt.figure()
                            st = plt.suptitle("%s\neta=%s alpha=%s lambda=%s n_hidden=%s weight=%s trials=%s k=%s" % (
                                title_plot, eta, alfa, lambd, hidden, weight, n_trials, k))
                            plt.subplot(2, 1, 1)
                            plt.plot(mean_err_tr_list, label='Training Error', ls="-")

                            plt.plot(mean_err_vl_list, label='Validation Error', ls="dashed")

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_tr_list - std_err_tr_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_tr_list + std_err_tr_list, n_epochs + 1, -1),
                                             color="b", alpha=0.2)

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_vl_list - std_err_vl_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_vl_list + std_err_vl_list, n_epochs + 1, -1),
                                             color="orange", alpha=0.2)

                            #plt.ylim([0, 10])
                            plt.ylabel('MSE')
                            plt.grid(True)
                            plt.xlabel('epoch')
                            plt.legend(loc='upper right', prop={'size': 12})

                            plt.subplot(2, 1, 2)
                            plt.plot(mean_err_tr_MEE_list, label='Training MEE', ls="-")

                            plt.plot(mean_err_vl_MEE_list, label='Validation MEE', ls="dashed")

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_tr_MEE_list - std_err_tr_MEE_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_tr_MEE_list + std_err_tr_MEE_list, n_epochs + 1, -1),
                                             color="b", alpha=0.2)

                            plt.fill_between(range(0, n_epochs + 1),
                                             np.reshape(mean_err_vl_MEE_list - std_err_vl_MEE_list, n_epochs + 1, -1),
                                             np.reshape(mean_err_vl_MEE_list + std_err_vl_MEE_list, n_epochs + 1, -1),
                                             color="orange", alpha=0.2)

                            plt.ylabel('MEE')
                            plt.grid(True)
                            # plt.ylim([0,10])
                            plt.xlabel('epoch')
                            plt.legend(loc='upper right', prop={'size': 12})
                            plt.subplots_adjust(hspace=0.5)

                            """
                            SALVO LA LEARNING CURVE SU FILE
                            """
                            plt.savefig("%s_eta_%s_alpha_%s_lambd_%s_hidd_%s_weight_%s_normale.jpg" % (
                                save_path_plot, eta, alfa, lambd, hidden, weight))
                            # plt.show()
                            plt.close(fig)

                            """
                            Controllo se VL error medio finale di questa configurazione è il migliore al momento.
                            """

                            if mean_err_vl_MEE_list[-1] < best_mean_vl_error:
                                print("\nTROVATO ERRORE MIGLIORE = %3f -> %3f\n" % (
                                best_mean_vl_error, mean_err_vl_MEE_list[-1]))

                                best_mean_vl_error = mean_err_vl_MEE_list[-1]
                                best_std_vl_error = std_err_vl_MEE_list[-1]

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
    print("FINE K_FOLD CV")
    print(100 * "-")
    print()
    return best_eta, best_alfa, best_hidden, best_lambda, best_weight, best_mean_vl_error, best_std_vl_error

