import sys

sys.path.append("../")
from matplotlib import pyplot as plt
from Trainers.TrainBackpropLS_new import *
from MLP.Activation_Functions import *
from MLP.MLP import *

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

    "Iperparametri CM A1"  # per ora valori di default
    eta_start = 0.1
    eta_max = 2
    max_iter = 100
    m1 = 0.0001
    m2 = 0.9
    tau = 0.9
    sfgrd = 0.001
    mina = 1e-16
    n_epochs = 10000
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

    "Scelta e Inserimento iperparametri Algoritmo alla rete - Fase di progetto e prove sperimentali"
    trainer = TrainBackPropLS(eta_start=eta, eta_max=2, max_iter=100, m1=0.0001, m2=0.9, tau=0.9,
                              sfgrd=0.0001, mina=1e-16)
    mlp = MLP(n_features, n_hidden, n_out, TanhActivation(), LinearActivation(), lambd=lambd, eta=eta, alfa=alpha,
              trainer=trainer, classification=classification)

    "traininig Alg."
    mlp.trainer.train(mlp, addBias(X), T, addBias(X), T, n_epochs=10000, eps=1e-4, suppress_print=True)

    """
    Si effettuano i plot
    """
    # Dettaglio
    plt.subplot(2, 1, 1)
    plt.plot(mlp.errors_tr)
    plt.grid(True)
    plt.ylim([mlp.errors_tr[-1] - 0.5, mlp.errors_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(2, 1, 2)
    plt.plot(mlp.errors_mee_tr)
    plt.grid(True)
    plt.ylim([mlp.errors_mee_tr[-1] - 0.5, mlp.errors_mee_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.show()

    # Generale
    plt.subplot(2, 1, 1)
    plt.plot(mlp.errors_tr)
    plt.grid(True)
    # plt.ylim([mlp.errors_tr[-1] - 0.5, mlp.errors_tr[-1] + 1])
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(2, 1, 2)
    plt.plot(mlp.errors_mee_tr)
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
