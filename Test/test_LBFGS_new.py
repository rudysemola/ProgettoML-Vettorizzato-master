import sys
sys.path.append("../")
from matplotlib import pyplot as plt
from Trainers.L_BFGS_new import *
from MLP.Activation_Functions import *
from MLP.MLP import *
if __name__ == '__main__':

    n_features = 10
    n_hidden = 33
    n_out = 2
    eta = 1
    alpha = 0.4
    lambd = 0.0075
    #lambd = 0
    classification = False

    P = loadMatrixFromFile("../Datasets/DatasetTrVl.csv")
    X = P[:, : - 2]
    T = P[:, -2:]

    trainer = LBFGS(eta_start=eta,eta_max=2,max_iter_AWLS_train=100,m1=0.0001,m2=0.9,tau=0.9,
                    sfgrd=0.0001,mina=1e-16,m=2000)
    mlp = MLP(n_features, n_hidden, n_out, TanhActivation(), LinearActivation(),lambd=lambd, eta=eta, alfa=alpha,trainer=trainer,classification=classification)

    #train(self, mlp, X, T, X_val, T_val, n_epochs=1000, eps=1e-6, threshold=0.5, suppress_print=False):
    mlp.trainer.train(mlp,addBias(X),T,addBias(X),T,n_epochs=10000,eps=1e-5)

    plt.subplot(2,1,1)
    plt.plot(mlp.errors_tr)
    plt.grid(True)
    plt.ylim([mlp.errors_tr[-1]-0.5, mlp.errors_tr[-1]+1])
    plt.xlabel("Epochs")

    plt.ylabel("MSE")

    plt.subplot(2, 1, 2)
    plt.plot(mlp.errors_mee_tr)
    plt.grid(True)
    plt.ylim([mlp.errors_mee_tr[-1] - 0.5, mlp.errors_mee_tr[-1] + 1])
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

