import sys

sys.path.append("../")
from matplotlib import pyplot as plt
from Trainers.TrainBackpropLS_new import *
from MLP.Activation_Functions import *
from MLP.MLP import *
from Validation.GridSearch import *
import time

if __name__ == '__main__':
    "Variabili necessari"
    n_features = 10
    n_out = 2
    classification = False

    "Iperparametri ML: configurazione finale"
    eta = 1
    alpha = 0.8
    lambd = 0.0075
    n_hidden = 33
    n_trials = 3

    "Iperparametri CM A1"  # per ora valori di default
    eta_start = 0.05
    eta_max = 2
    max_iter = 100
    m1 = 0.001
    m2 = 0.9
    tau = 0.9
    sfgrd = 0.001
    mina = 1e-16
    n_epochs = 1000
    eps = 1e-4

    "Criteri di arresto impiegati dall'algoritmo"
    done_max_epochs = False  # Fatte numero massimo iterazioni
    found_optimum = False  # Gradiente minore o uguale a eps_prime

    "Criteri di arresto impiegati su AWLS"
    done_max_iters = False   # Fatte numero massimo iterazioni (AWLS!)
    reached_eta_max = False  # raggiunto eta massimo
    wolfe_satisfied = False  # AW sodisfatta

    "Caricamento dataset per sperimenti"
    P = loadMatrixFromFile("../Datasets/DatasetTrVl.csv")
    X = P[:, : - 2]
    T = P[:, -2:]

    "Per stampe salvandole su file"
    title = "../RisultatiCM/bpls_"
    title = title + time.strftime("%d-%m-%Y-%H%M%S") + ".csv"

    title_stat = "../RisultatiCM/bpls_stat_"
    title_stat = title_stat + time.strftime("%d-%m-%Y-%H%M%S") + ".csv"

    "Scelta e Inserimento iperparametri Algoritmo alla rete - Fase di progetto e prove sperimentali"
    trainer = TrainBackPropLS(
    eta_start=eta_start, eta_max=eta_max, max_iter=max_iter, m1=m1, m2=m2, tau=tau,
    sfgrd=sfgrd, mina=mina, path_results=title)

    "traininig Alg. su numero di volte pari a n_trial"
    """
    best_mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl, std_error_MEE_vl,\
    avg_epochs_done = run_trials(n_features, X, T, X, T, n_epochs=n_epochs,
                                 hidden_act=TanhActivation(), output_act=LinearActivation(),
                                 eta=eta, alfa=alpha, n_hidden=n_hidden, weight=0.7, lambd=lambd, n_trials=n_trials,
                              classification=classification, trainer=trainer, eps=eps)
    """

    mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl, \
    std_error_MEE_vl=perform_test(n_features, X, T, X, T, n_epochs=n_epochs,
               hidden_act=TanhActivation(), output_act=LinearActivation(),
               eta=eta, alfa=alpha, n_hidden=n_hidden, weight=0.7, lambd=lambd, n_trials=n_trials,
               classification=classification, trainer=trainer, eps=eps,path_results=title_stat)
    # Dettaglio
    plt.subplot(2, 1, 1)
    plt.plot(mean_err_tr)
    plt.grid(True)
    plt.ylim([mean_err_tr[-1] - 0.5, mean_err_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(2, 1, 2)
    plt.plot(mean_error_MEE_tr)
    plt.grid(True)
    plt.ylim([mean_error_MEE_tr[-1] - 0.5, mean_error_MEE_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.show()

    # Generale
    plt.subplot(2, 1, 1)
    plt.plot(mean_err_tr)
    plt.grid(True)
    # plt.ylim([mlp.errors_tr[-1] - 0.5, mlp.errors_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(2, 1, 2)
    plt.plot(mean_error_MEE_tr)
    plt.grid(True)
    # plt.ylim([mlp.errors_mee_tr[-1] - 0.5, mlp.errors_mee_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.show()

"""
    #PLOT DELLE LEARNING CURVE...
    plt.plot(mlp.errors_tr)
    plt.plot(mlp.errors_vl)
    plt.grid(True)
    plt.ylim((0,10))
    plt.show()
    plt.plot(mlp.errors_mee_tr)
    plt.plot(mlp.errors_mee_vl)
    plt.grid(True)
    plt.ylim((0, 10))
    plt.show()
    #PLOT DEL GRADIENTE...
    plt.loglog(mlp.gradients)
    plt.grid(True)
   # plt.ylim((0, 10))
    plt.show()
    print(mlp.errors_mee_tr)
    #e = compute_obj_function(mlp,addBias(X),T,lambd)
    #gradE_h,gradE_o = compute_gradient(mlp,addBias(X),T,lambd)
    #phi,phi_p =f2phi(eta,mlp,addBias(X),T,gradE_h,gradE_o,lambd)
    #print("Phi = %s, Phi primo = %s"%(phi,phi_p))
    #mlp.trainer.train(mlp, addBias(X), Y, addBias(X_val), Y_val, 500, 1e-6)
    #(mlp,X,T,phi_0,gradE_h,gradE_o,lambd,eta_start=1,eta_max=20,max_iter=100,m1=0.001,m2=0.9,tau = 0.9)
    #eta_star = AWLS(mlp,addBias(X),T,e,gradE_h,gradE_o,lambd,eta_start=0.01,eta_max=4,max_iter=100)
    print("Eta_star trovato: ",eta_star)
"""
