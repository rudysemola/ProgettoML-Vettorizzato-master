import sys
sys.path.append("../")

from Trainers.Training import *
from Utilities.Utility import *

""""
Effettua lo standard SGD con momentum usando stepsize fissato.
"""
class TrainBackprop(Training):

    def train(self,mlp,X, T, X_val, T_val, n_epochs = 1000, eps = 10 ^ (-3), threshold = 0.5, suppress_print = False):
        assert X.shape[0] == T.shape[0]
        # 1) Init pesi e iperparametri // fatto nel costruttore

        # 4) Condizioni di arresto
        error_MSE = 100
        for epoch in range(n_epochs):

            # 2) Effettuo la feedfoward;
            #   calcolo MSE class/regress (Learning Curve TR/VL), accuracy(accuracy curve TR/VL, class)/ MEE (regress);
            #   calcolo delta_W usando backpropagation
            mlp.feedforward(X)
            # print "n_output:", self.n_output
            # print "OUT_o", self.Out_o.shape
            # print "Target", T.shape
            error_MSE = compute_Error(T, mlp.Out_o)
            if mlp.classification:
                accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
                mlp.errors_tr.append(error_MSE)
                mlp.accuracies_tr.append(accuracy)
            else:
                error_MEE = compute_Regr_MEE(T, mlp.Out_o)
                mlp.errors_tr.append(error_MSE)
                mlp.errors_mee_tr.append(error_MEE)

            dW_o, dW_h = mlp.backpropagation(X, T)

            # CALCOLO IL VALIDATION ERROR
            mlp.feedforward(X_val)
            error_MSE_val = compute_Error(T_val, mlp.Out_o)
            if mlp.classification:
                accuracy_val = compute_Accuracy_Class(T_val, convert2binary_class(mlp.Out_o, threshold))
                mlp.errors_vl.append(error_MSE_val)
                mlp.accuracies_vl.append(accuracy_val)
            else:
                error_MEE_val = compute_Regr_MEE(T_val, mlp.Out_o)
                mlp.errors_vl.append(error_MSE_val)
                mlp.errors_mee_vl.append(error_MEE_val)

            # 3) Upgrade weights

            # TODO: LINE SEARCH-> CM+ML...
            # TODO: A0
            # TODO: A1
            # TODO: A2

            # A1
            """
            if opt_a1:
                loss = error_MSE  # lamba=0
                self.eta = AWLS(self, X, T, loss, -dW_h, -dW_o, 0)
            """

            # self.eta = AWLS(self,X,T,error_MSE,dW_h,dW_o)
            dW_o_new = mlp.eta * dW_o + mlp.alfa * mlp.dW_o_old
            mlp.W_o = mlp.W_o + dW_o_new - (mlp.lambd * mlp.W_o)

            dW_h_new = mlp.eta * dW_h + mlp.alfa * mlp.dW_h_old
            mlp.W_h = mlp.W_h + dW_h_new - (mlp.lambd * mlp.W_h)

            mlp.dW_o_old = dW_o_new
            mlp.dW_h_old = dW_h_new

            # per stampa per ogni epoca
            if not suppress_print:
                if mlp.classification:
                    print(
                        "Epoch %s/%s) TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                            epoch + 1, n_epochs, error_MSE, error_MSE_val, accuracy, accuracy_val))
                else:
                    print(
                        "Epoch %s/%s) TR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL ((MEE) : %s" % (
                            epoch + 1, n_epochs, error_MSE, error_MSE_val, error_MEE, error_MEE_val))

        # CALCOLO ERRROR E ACCURACY/MEE FINALI (metto nelle liste)
        mlp.feedforward(X)
        error_MSE = compute_Error(T, mlp.Out_o)
        if mlp.classification:
            accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
            mlp.errors_tr.append(error_MSE)
            mlp.accuracies_tr.append(accuracy)
        else:
            error_MEE = compute_Regr_MEE(T, mlp.Out_o)
            mlp.errors_tr.append(error_MSE)
            mlp.errors_mee_tr.append(error_MEE)

        mlp.feedforward(X_val)
        error_MSE_val = compute_Error(T_val, mlp.Out_o)
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
                    "Final Results: TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                        mlp.errors_tr[-1], mlp.errors_vl[-1], mlp.accuracies_tr[-1], mlp.accuracies_vl[-1]))
            else:
                print(
                    "Final Results: TR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL (MEE) : %s" % (
                        mlp.errors_tr[-1], mlp.errors_vl[-1], mlp.errors_mee_tr[-1], mlp.errors_mee_vl[-1]))


        return len(mlp.errors_tr)