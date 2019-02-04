from Validation.HoldOut import *
from Validation.KFoldCV import *
from matplotlib import pyplot as plt
from MLP.Activation_Functions import *


eta_values = [0.02,0.01,0.05]
alfa_values = [0.8,0.6,0.7,0.5]
hidden_values = [12,20,30]
weight_values = [0.7]
lambda_values = [0.01,0.1,0.001]
n_epochs = 1000
n_trials = 3
n_features = 10
k = 5
classifications = False

hold_out = False




"fase di splitting Tr-Vl/TS  oppure load matrici nei file"
P = loadMatrixFromFile("../Datasets/DatasetTrVl.csv")
X = P[:, : - 2]
T = P[:, -2:]
print(X.shape)
print(T.shape)
# print(X[1, :])
# print(T[1, :])

if hold_out:
    "HOLD OUT: TECNICA DI VALIDAZIONE"
    best_eta, best_alfa, best_hidden, best_lambda, best_weight, best_mean_vl_error = do_HoldOut(n_features, X, T,
                                                                                                n_epochs,
                                                                                                TanhActivation(),
                                                                                                LinearActivation(),
                                                                                                eta_values, alfa_values,
                                                                                                hidden_values,
                                                                                                weight_values,
                                                                                                lambda_values, n_trials,
                                                                                                classifications)

    "RETRAINING TR+VL data"
    # NOTA: mi interessano solo quelli di vl (perche retrain su tutto!)
    # NON MI PIACE QUESTO...
    X_tr, T_tr, X_vl, T_vl = split_data_train_validation(X, T)  # TR=75%, VL=25% (

    # ReTrain su tutto X,T!
    mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl = run_trials(
        n_features, X, T, X_vl, T_vl, n_epochs, TanhActivation(),  LinearActivation(),
        best_eta, best_alfa, best_hidden, best_weight, best_lambda, n_trials, classifications)

    # Print & plot su Regressione
    print("TR ERR = %3f TR ACC = %3f VL ERR = %3f VL ACC = %3f" % (mlp.errors_tr[-1], mlp.errors_mee_tr[-1],
                                                                   mlp.errors_vl[-1], mlp.errors_mee_vl[-1]))

    # Print & plot su Regressione
    st = plt.suptitle(
        "Best model Regression Hold Out\neta=%s alpha=%s lambda=%s n_hidden=%s" % (
        mlp.eta, mlp.alfa, mlp.lambd, mlp.n_hidden))
    plt.subplot(2, 1, 1)
    plt.plot(mlp.errors_tr, label='Training Error', ls="-")
    plt.plot(mlp.errors_vl, label='Validation Error', ls="dashed")
    plt.ylabel('loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.subplot(2, 1, 2)
    plt.plot(mlp.errors_mee_tr, label='Training MEE', ls="-")
    plt.plot(mlp.errors_mee_vl, label='Validation MEE', ls="dashed")
    plt.ylabel('MEE')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='lower right', prop={'size': 12})
    plt.show()

else:
    "KFOLD CV: TECNICA DI VALIDAZIONE"
    best_eta, best_alfa, best_hidden, best_lambda, best_weight, best_mean_vl_error, best_std_vl_error = kFoldCV(
        n_features, X, T, k, 500,
        TanhActivation(), LinearActivation(),
        eta_values, alfa_values, hidden_values, weight_values, lambda_values, n_trials, classifications)

    "RETRAINING TR+VL data"
    # NOTA: mi interessano solo quelli di vl (perche retrain su tutto!)
    # NON MI PIACE QUESTO...
    X_tr, T_tr, X_vl, T_vl = split_data_train_validation(X, T)  # TR=75%, VL=25% (

    # ReTrain su tutto X,T!
    mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl = run_trials(
        n_features, X, T, X_vl, T_vl, n_epochs, TanhActivation(),  LinearActivation(),
        best_eta, best_alfa, best_hidden, best_weight, best_lambda, n_trials, classifications)

    # Print & plot su Regressione
    print("TR ERR = %3f TR ACC = %3f VL ERR = %3f VL ACC = %3f" % (mlp.errors_tr[-1], mlp.errors_mee_tr[-1],
                                                                   mlp.errors_vl[-1], mlp.errors_mee_vl[-1]))

    # Print & plot su Regressione
    st = plt.suptitle(
        "Best model Regression KCV\neta=%s alpha=%s lambda=%s n_hidden=%s" % (
        mlp.eta, mlp.alfa, mlp.lambd, mlp.n_hidden))
    plt.subplot(2, 1, 1)
    plt.plot(mlp.errors_tr, label='Training Error', ls="-")
    plt.plot(mlp.errors_vl, label='Validation Error', ls="dashed")
    plt.ylabel('loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.subplot(2, 1, 2)
    plt.plot(mlp.errors_mee_tr, label='Training MEE', ls="-")
    plt.plot(mlp.errors_mee_vl, label='Validation MEE', ls="dashed")
    plt.ylabel('MEE')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='lower right', prop={'size': 12})
    plt.show()
