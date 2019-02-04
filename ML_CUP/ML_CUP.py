import sys
sys.path.append("../")

"""
File usato per effettuare le grid search per la ML CUP
"""

from Validation.KFoldCV import *
from matplotlib import pyplot as plt
from MLP.Activation_Functions import *
from Trainers.TrainBackprop import *

#eta_values = [0.03,0.05,0.07]
eta_values = [0.0525]
#alfa_values = [0.5,0.7]
alfa_values = [0.9]
#alfa_values = [0.5,0.7]
hidden_values = [28,30]
#hidden_values = [10,12,15,20,25,30]
weight_values = [0.7]
lambda_values = [0.0065]
#lambda_values = [0]
n_epochs = 8000 #FARLE VARIARE TRA 1000 5000 E 10000
n_trials = 7

k = 3 #/5/10
n_features = 10
window_size = 1 #fa il plot tra [max(0,vl_err - window_size,vl_err + window_size]

classifications = False

title_plot = "ML CUP"
save_path_plot="../Plots/cup"
save_path_results="../Results_CSV/cup"

P = loadMatrixFromFile("../Datasets/DatasetTrVl.csv")
X = P[:, : - 2]
T = P[:, -2:]
trainer = TrainBackprop()

"KFOLD CV: TECNICA DI VALIDAZIONE"
best_eta, best_alfa, best_hidden, best_lambda, best_weight, best_mean_vl_error, best_std_vl_error = kFoldCV(
    n_features, X, T, k, n_epochs,
    TanhActivation(), LinearActivation(),
    eta_values, alfa_values, hidden_values, weight_values, lambda_values, n_trials, classifications,title_plot=title_plot,
    save_path_plot=save_path_plot,save_path_results=save_path_results,window_size=window_size)

