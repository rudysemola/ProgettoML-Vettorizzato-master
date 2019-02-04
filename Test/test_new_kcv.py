from Validation.KFoldCV import *
from Monks.Monk import *
from matplotlib import pyplot as plt
from MLP.Activation_Functions import *
from ML_CUP.LoadDataset import *

eta_values = [0.8,0.7,0.4,0.2]
alfa_values = [0.8]
hidden_values =[3,5,10,20]
weight_values = [0.7]
lambda_values = [0]
n_epochs = 500
n_trials = 3
k=5
n_features = 10
classifications = False

P = loadMatrixFromFile("../Datasets/DatasetTrVl.csv")
X = P[:, : - 2]
T = P[:, -2:]

"KFOLD CV"  # Classificazione
"""
best_eta,best_alfa,best_hidden,best_lambda,best_weight,best_mean_vl_error,best_std_vl_error=kFoldCV(n_features,X,T,k,500,
    TanhActivation(),SigmoidActivation(),
    eta_values,alfa_values,hidden_values,weight_values,lambda_values,n_trials, classifications,title_plot="Monk 1",save_path_plot="../Plots/monk1",
                                                                                                    save_path_results="../Results_CSV/monk1")
"""
best_eta,best_alfa,best_hidden,best_lambda,best_weight,best_mean_vl_error,best_std_vl_error=kFoldCV(n_features,X,T,k,500,
    TanhActivation(),LinearActivation(),
    eta_values,alfa_values,hidden_values,weight_values,lambda_values,n_trials, classifications,title_plot="ML_CUP",save_path_plot="../Plots/cup",
                                                                                                    save_path_results="../Results_CSV/cup")
