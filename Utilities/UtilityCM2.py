import sys

sys.path.append("../")

from Utilities.Utility import *
from Validation.GridSearch import *
import csv
import time


"""
Trasforma 2 matrici in un unico vettore [X|Y]
"""


def matrix2vec(X, Y):
    X_vett = np.reshape(X, (-1, 1))
    Y_vett = np.reshape(Y, (-1, 1))
    vect = np.concatenate((X_vett, Y_vett), axis=0)
    return vect


"""
Trasforma due vettori in una unica matrice
"""


def vec2matrix(X, shape_h, shape_o):
    W_h = X[:(shape_h[0] * shape_h[1])]
    W_o = X[-(shape_o[0] * shape_o[1]):]
    W_h = np.reshape(W_h, (shape_h[0], shape_h[1]))
    W_o = np.reshape(W_o, (shape_o[0], shape_o[1]))

    return W_h, W_o


"""
Calcola funzione obiettivo e relativo gradiente nel punto w
:param mlp : Rete neurale
:param X : Matrice dei dati di training
:param T : Matrice dei target di training
:param w : Punto in cui calcolare funzione e gradiente

:return f: Valore della funzione in w
:return grad_f: Gradiente della funzione in w
"""


def evaluate_function(mlp, X, T, w):
    W_h_init = np.copy(mlp.W_h)
    W_o_init = np.copy(mlp.W_o)

    # Mi metto nel punto w
    W_h, W_o = vec2matrix(w, mlp.W_h.shape, mlp.W_o.shape)
    mlp.W_h = W_h
    mlp.W_o = W_o

    # Valuto funzione
    mlp.feedforward(X)
    mse = compute_Error(T, mlp.Out_o)
    norm_w = np.linalg.norm(mlp.W_h) ** 2 + np.linalg.norm(mlp.W_o) ** 2
    loss = mse + (0.5 * mlp.lambd * norm_w)

    # Valuto gradiente
    m_grad_mse_o, m_grad_mse_h = mlp.backpropagation(X, T)
    grad_mse_o = - m_grad_mse_o
    grad_mse_h = - m_grad_mse_h

    grad_o = grad_mse_o + (mlp.lambd * mlp.W_o)
    grad_h = grad_mse_h + (mlp.lambd * mlp.W_h)

    grad = matrix2vec(grad_h, grad_o)

    mlp.W_h = W_h_init
    mlp.W_o = W_o_init

    return loss, grad


"""
Sposta i pesi della rete nel punto w
:param w : Punto in cui mettere i pesi della rete
"""


def update_weights(mlp, w):
    W_h, W_o = vec2matrix(w, mlp.W_h.shape, mlp.W_o.shape)
    mlp.W_h = W_h
    mlp.W_o = W_o
    return


"""
Restituisce il punto in cui si trovano i pesi di mlp

:param mlp : Rete neurale

:return w: Vettore dei pesi attuali.
"""


def get_current_point(mlp):
    w = matrix2vec(mlp.W_h, mlp.W_o)
    return w



"""
Serve per fare le prove per CM
"""
def perform_test(n_features, X, T, X_val, T_val, n_epochs,
                                 hidden_act, output_act,
                                 eta, alfa, n_hidden, weight, lambd, n_trials,
                                 classification, trainer, eps,
                                 path_results
                                 ):

    fieldnames_trial = ['Trial','Iterazioni Totali', 'Iterazioni spese in Line Search', 'Trovato ottimo' ,'Ottimo']

    avg_epochs_done = 0 #Numero medio di epoche fatte
    avg_it_AWLS_done = 0 #Numero medio di AWLS fatte
    n_converged = 0 #Numero di trial che hanno raggiunto l'ottimo
    time_tot  = 0 #Tempo totale in cui sono stati eseguiti tutti i trials

    errors_tr = np.zeros((n_trials, n_epochs + 1))
    errors_vl = np.zeros((n_trials, n_epochs + 1))

    if classification:
        acc_tr = np.zeros((n_trials, n_epochs+1))
        acc_vl = np.zeros((n_trials, n_epochs+1))
    else:
        errors_MEE_tr = np.zeros((n_trials, n_epochs+1))
        errors_MEE_vl = np.zeros((n_trials, n_epochs+1))

    start = time.time()


    for trial in range(n_trials):

        mlp = MLP(n_features, n_hidden, T.shape[1], hidden_act, output_act, eta=eta, alfa=alfa, lambd=lambd,
                  fan_in_h=True, range_start_h=-weight, range_end_h=weight,
                  classification=classification, trainer=trainer)

        epoch_done,ls_iters,converged,optimum, hyperparam = mlp.trainer.train(mlp, addBias(X), T, addBias(X_val), T_val, n_epochs, eps,
                                       suppress_print=True)

        avg_epochs_done += epoch_done
        avg_it_AWLS_done += ls_iters
        if converged:
            n_converged += 1

        #Salvo le info x ogni trial su file csv

        if trial == 0:
            with open(path_results,"w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames_trial)
                    for key, item in hyperparam.items():
                        f.write("#%s:%s\n" % (key, item))
                    writer.writeheader()

        with open(path_results, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_trial)
            writer.writerow({'Trial':trial+1,'Iterazioni Totali': epoch_done, 'Iterazioni spese in Line Search': ls_iters,
                             'Trovato ottimo': converged, 'Ottimo': optimum})

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

    with open(path_results, "a", newline='') as f:
        f.write("\n\n\t\t\t\t\tRISULTATI COMPLESSIVI")


    end = time.time()
    time_tot = end - start

    avg_epochs_done = math.ceil(avg_epochs_done /n_trials)
    avg_it_AWLS_done = math.ceil(avg_it_AWLS_done/n_trials)

    fieldnames_total = ['Trials totali','Iterazioni medie','# Iterazioni LS medie', '# Trials Converged', 'MSE medio',
                        'Std MSE', 'Tempo totale']

    with open(path_results, "a", newline='') as f:
        f.write("\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames_total)
        writer.writeheader()
        writer.writerow(
            {'Trials totali': n_trials, 'Iterazioni medie': avg_epochs_done, '# Iterazioni LS medie': avg_it_AWLS_done,
             '# Trials Converged': n_converged, 'MSE medio': float(mean_err_tr[-1]), 'Std MSE': float(std_err_tr[-1]),
             'Tempo totale':time_tot})

    if classification:
        return mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl
    else:
        return mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl, std_error_MEE_vl
