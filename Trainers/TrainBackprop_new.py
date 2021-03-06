import sys

sys.path.append("../")

from Trainers.Training import *
from Utilities.UtilityCM2 import *

""""
Effettua lo standard SGD con momentum usando stepsize fissato.
"""


class TrainBackprop2(Training):

    def __init__(self, path_results="../RisultatiCM/bp.csv"):
        self.w_prec = None
        self.w_new = None

        self.gradE = 0
        self.gradE_0 = 0
        self.epsilon_prime = 0
        self.path_results = path_results
        self.norm_gradE = 0
        self.norm_gradE_0 = 0
        self.hyperparameters = {}

    def train(self, mlp, X, T, X_val, T_val, n_epochs=1000, eps=10 ^ (-3), threshold=0.5, suppress_print=False):
        assert X.shape[0] == T.shape[0]

        epoch = 0
        done_max_epochs = False  # Fatte numero massimo iterazioni
        found_optimum = False  # Gradiente minore o uguale a eps_prime

        fieldnames = ['Iterazione', 'Eta', 'Errore', 'Gradiente']

        while (not done_max_epochs) and (not found_optimum):
            if epoch == 0:

                self.w_prec = get_current_point(mlp)
                self.w_new = self.w_prec
                E, self.gradE = evaluate_function(mlp, X, T, self.w_new)

                if mlp.classification:
                    accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
                    mlp.errors_tr.append(E)
                    mlp.accuracies_tr.append(accuracy)
                else:
                    error_MEE = compute_Regr_MEE(T, mlp.Out_o)
                    mlp.errors_tr.append(E)
                    mlp.errors_mee_tr.append(error_MEE)

                self.norm_gradE_0 = np.linalg.norm(self.gradE)

                self.epsilon_prime = eps * self.norm_gradE_0
                self.norm_gradE = self.norm_gradE_0

                mlp.gradients.append(self.norm_gradE / self.norm_gradE_0)


                self.hyperparameters = {
                    'alpha': mlp.alfa,
                    'eta': mlp.eta,
                    'epsilon': eps,
                    'epsilon_prime': self.epsilon_prime,
                    'n_epochs': n_epochs
                }

                file_exists = os.path.isfile(self.path_results)
                if not file_exists:
                    with open(self.path_results, "w") as f:

                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        for key, item in self.hyperparameters.items():
                            f.write("#%s:%s\n" % (key, item))

                        writer.writeheader()
                else:
                    with open(self.path_results, "a") as f:
                        f.write("\n\n")

            else:
                self.w_new = get_current_point(mlp)
                E, self.gradE = evaluate_function(mlp, X, T, self.w_new)

                if mlp.classification:
                    accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
                    mlp.errors_tr.append(E)
                    mlp.accuracies_tr.append(accuracy)
                else:
                    error_MEE = compute_Regr_MEE(T, mlp.Out_o)
                    mlp.errors_tr.append(E)
                    mlp.errors_mee_tr.append(error_MEE)

                self.norm_gradE = np.linalg.norm(self.gradE)

                mlp.gradients.append(self.norm_gradE / self.norm_gradE_0)

            # CALCOLO IL VALIDATION ERROR
            error_MSE_val, _ = evaluate_function(mlp, X_val, T_val, self.w_new)

            if mlp.classification:
                accuracy_val = compute_Accuracy_Class(T_val, convert2binary_class(mlp.Out_o, threshold))
                mlp.errors_vl.append(error_MSE_val)
                mlp.accuracies_vl.append(accuracy_val)
            else:
                error_MEE_val = compute_Regr_MEE(T_val, mlp.Out_o)
                mlp.errors_vl.append(error_MSE_val)
                mlp.errors_mee_vl.append(error_MEE_val)

            # CONTROLLO GRADIENTE
            if self.norm_gradE < self.epsilon_prime:
                found_optimum = True

            if not found_optimum:

                d = - self.gradE

                # AGGIORNAMENTO

                w_new = self.w_new + mlp.eta * d + (mlp.alfa * (self.w_new - self.w_prec))

                self.w_prec = self.w_new
                self.w_new = w_new

                update_weights(mlp, self.w_new)

                # per stampa per ogni epoca
                if not suppress_print:
                    if mlp.classification:
                        print(
                            "Epoch %s/%s) Eta = %s ||gradE||/ ||gradE_0|| = %s,TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                                epoch + 1, n_epochs, mlp.eta, self.norm_gradE / self.norm_gradE_0, E,
                                error_MSE_val, accuracy, accuracy_val))
                    else:
                        print(
                            "Epoch %s/%s) Eta = %s ||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL ((MEE) : %s" % (
                                epoch + 1, n_epochs, mlp.eta, self.norm_gradE / self.norm_gradE_0,
                                E, error_MSE_val, error_MEE, error_MEE_val))

                with open(self.path_results, "a",newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({'Iterazione':epoch+1,
                                     'Eta':mlp.eta,'Errore':E,'Gradiente':self.norm_gradE / self.norm_gradE_0})


                print("Iterazione %s) Eta = %s New Error %s New Gradient %s" % (
                    epoch, mlp.eta, E, self.norm_gradE / self.norm_gradE_0))
                epoch += 1
                # CONTROLLO EPOCHE
                if epoch >= n_epochs:
                    done_max_epochs = True

        #CALCOLO ERRORE DOPO L'ULTIMO AGGIORNAMENTO

        E, _ = evaluate_function(mlp, X, T, self.w_new)

        if mlp.classification:
            accuracy = compute_Accuracy_Class(T, convert2binary_class(mlp.Out_o, threshold))
            mlp.errors_tr.append(E)
            mlp.accuracies_tr.append(accuracy)
        else:
            error_MEE = compute_Regr_MEE(T, mlp.Out_o)
            mlp.errors_tr.append(E)
            mlp.errors_mee_tr.append(error_MEE)
            error_MSE_val, _ = evaluate_function(mlp, X_val, T_val, self.w_new)

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
                    "Final Results: ||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                        self.norm_gradE / self.norm_gradE_0, mlp.errors_tr[-1], mlp.errors_vl[-1],
                        mlp.accuracies_tr[-1], mlp.accuracies_vl[-1]))
            else:
                print(
                    "Final Results:||gradE||/ ||gradE_0|| = %s\nTR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL (MEE) : %s" % (
                        self.norm_gradE / self.norm_gradE_0, mlp.errors_tr[-1], mlp.errors_vl[-1],
                        mlp.errors_mee_tr[-1], mlp.errors_mee_vl[-1]))

        if found_optimum:
            print("Trovato ottimo")
        elif done_max_epochs:
            print("Terminato il numero massimo di iterazioni disponibili")

        return len(mlp.errors_tr), 0, found_optimum, mlp.errors_tr[-1],self.hyperparameters
