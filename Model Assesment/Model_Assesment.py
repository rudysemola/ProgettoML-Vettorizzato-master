"""
File usato per effettuare:
-retraining modello finale
-assessment modello finale su test interno
-produrre risultati per il blind test
"""

import sys

sys.path.append("../")

import numpy as np
from Utilities.Utility import *
from Trainers.TrainBackprop import *
from Validation.GridSearch import *
from matplotlib import pyplot as plt
from ML_CUP.LoadDataset import *
from MLP.Activation_Functions import *

"""
Salve su file gli elementi della matrice M.
La formattazione è quella necessaria per il blind test

:param M : Matrice da salvare su file, contenente i risultati del blind test
:param path_test: Path del file su cui salvare i risultati del blind test
"""


def saveBlindTestResults(M, path_test):
    with open(path_test, "w")as f:
        f.write("#\n")
        f.write("#\n")
        f.write("#\n")
        f.write("#\n")

        id = 1

        for (idx_row, row) in enumerate(M):
            f.write("%s," % (id))
            for (idx_col, element) in enumerate(row):
                f.write("%s" % (element))
                if not idx_col == M.shape[1] - 1:
                    f.write(",")

            f.write("\n")
            id += 1

    return


"""
Carica il dataset usato come internal test set.
:return X_retrain, T_retrain,X_test,T_test
"""


def load_internal_test():
    P_retrain = loadMatrixFromFile("../Datasets/DatasetTrVl.csv")
    X_retrain = P_retrain[:, : - 2]
    T_retrain = P_retrain[:, -2:]

    P_test = loadMatrixFromFile("../Datasets/TestSetInterno.csv")
    X_test = P_test[:, : - 2]
    T_test = P_test[:, -2:]

    return X_retrain, T_retrain, X_test, T_test


"""
Effettua il retraining del modello finale sull'intero dev set, testandolo su test set interno.
Restituisce il modello finale.
"""


def model_assesment(n_features,
                    n_epochs, hidden_act, output_act, eta, alfa, n_hidden, weight, lambd,
                    n_trials, classification=False, trainer=TrainBackprop(),
                    title_plot="Plot Finale", save_path_plot="../RisultatiFinali/cupFinale",
                    save_path_results="../RisultatiFinali/cupFinale.txt"):
    window_size = 1

    X_retrain, T_retrain, X_test, T_test = load_internal_test()

    best_mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_test, \
    std_err_test, mean_error_MEE_test, std_error_MEE_test, epochs = run_trials(
        n_features, X_retrain, T_retrain, X_test, T_test, n_epochs, hidden_act, output_act,
        eta, alfa, n_hidden, weight, lambd,
        n_trials, classification=classification, trainer=trainer)

    """
    SALVO RISULTATI STATISTICI SU FILE
    """

    with open(save_path_results, "w") as f:
        f.write("TR MSE : %3f +- %3f, TS MSE : %3f +- %3f TR MEE : %3f +- %3f TS MEE : %3f +- %3f" % (
            mean_err_tr[-1], std_err_tr[-1], mean_err_test[-1], std_err_test[-1],
            mean_error_MEE_tr[-1], std_error_MEE_tr[-1], mean_error_MEE_test[-1], std_error_MEE_test[-1]
        ))

    print("STIMA FINALE:\nTR MSE : %f +- %f, TS MSE : %f +- %f TR MEE : %f +- %f TS MEE : %f +- %f" % (
        mean_err_tr[-1], std_err_tr[-1], mean_err_test[-1], std_err_test[-1],
        mean_error_MEE_tr[-1], std_error_MEE_tr[-1], mean_error_MEE_test[-1], std_error_MEE_test[-1]
    ))

    """
    FACCIO LA LEARNING CURVE
    """

    fig = plt.figure()
    st = plt.suptitle("%s\neta=%s alpha=%s lambda=%s n_hidden=%s weight=%s trials=%s" % (
        title_plot, eta, alfa, lambd, n_hidden, weight, n_trials))
    plt.subplot(2, 1, 1)
    plt.plot(mean_err_tr, label='Training Error', ls="-")

    plt.plot(mean_err_test, label='Test Error', ls="dashed")

    plt.fill_between(range(0, n_epochs + 1),
                     np.reshape(mean_err_tr - std_err_tr, n_epochs + 1, -1),
                     np.reshape(mean_err_tr + std_err_tr, n_epochs + 1, -1),
                     color="b", alpha=0.2)

    plt.fill_between(range(0, n_epochs + 1),
                     np.reshape(mean_err_test - std_err_test, n_epochs + 1, -1),
                     np.reshape(mean_err_test + std_err_test, n_epochs + 1, -1),
                     color="orange", alpha=0.2)

    ylim_sup = mean_err_test[-1] + window_size
    ylim_inf = max([mean_err_test[-1] - window_size, 0])

    plt.ylim([ylim_inf, ylim_sup])
    plt.ylabel('MSE')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='upper right', prop={'size': 12})

    plt.subplot(2, 1, 2)
    plt.plot(mean_error_MEE_tr, label='Training MEE', ls="-")

    plt.plot(mean_error_MEE_test, label='Test MEE', ls="dashed")

    plt.fill_between(range(0, n_epochs + 1),
                     np.reshape(mean_error_MEE_tr - std_error_MEE_tr, n_epochs + 1, -1),
                     np.reshape(mean_error_MEE_tr + std_error_MEE_tr, n_epochs + 1, -1),
                     color="b", alpha=0.2)

    plt.fill_between(range(0, n_epochs + 1),
                     np.reshape(mean_error_MEE_test - std_error_MEE_test, n_epochs + 1, -1),
                     np.reshape(mean_error_MEE_test + std_error_MEE_test, n_epochs + 1, -1),
                     color="orange", alpha=0.2)

    plt.ylabel('MEE')
    plt.grid(True)

    ylim_sup = mean_error_MEE_test[-1] + window_size
    ylim_inf = max([mean_error_MEE_test[-1] - window_size, 0])

    plt.ylim([ylim_inf, ylim_sup])
    plt.xlabel('epoch')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.subplots_adjust(hspace=0.5)

    """
    SALVO LA LEARNING CURVE SU FILE
    """
    plt.savefig("%s_eta_%s_alpha_%s_lambd_%s_hidd_%s_weight_%s.jpg" % (
        save_path_plot, eta, alfa, lambd, n_hidden, weight))
    # plt.show()
    plt.close(fig)

    return best_mlp


"""
Esegue il blind test e ne salva i risultati su file.
:param mlp : modello finale
:param path_blind_results: Path del file in cui salvare i risultati del blind test
"""


def blind_test(mlp, path_blind_results):
    X_test_blind = load_cup_dataset_blind("../Datasets/ML-CUP18-TS.csv")
    mlp.feedforward(addBias(X_test_blind))
    saveBlindTestResults(mlp.Out_o, path_blind_results)
    return


if __name__ == "__main__":
    eta = 0.0525
    alfa = 0.9
    hidden = 33
    weight = 0.7
    lambd = 0.0075
    n_epochs = 8000
    n_trials = 7
    k = 3
    n_features = 10
    window_size = 1
    classifications = False
    trainer = TrainBackprop()

    final_model = model_assesment(n_features, n_epochs, TanhActivation(), LinearActivation(),
                                  eta, alfa, hidden, weight, lambd, n_trials,
                                  classifications, trainer)

    blind_test(final_model, "../Blind Test Result/team-name_ML-CUP18-TS.csv")
