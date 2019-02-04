"""

File usato per effettuare prove su Monk1
"""
import sys
sys.path.append("../")

from Monks.Monk import *
from MLP.Activation_Functions import *
from matplotlib import pyplot as plt
from MLP.MLP import *
from Validation.GridSearch import *

import time


X, Y = load_monk("../Datasets/monks-1.train")
X_val, Y_val = load_monk("../Datasets/monks-1.test")

"""
mlp = MLP(17,3,1,TanhActivation(),SigmoidActivation(),
          eta=0.8,alfa=.8,
          lambd=0,
          fan_in_h=True,range_start_h=-0.2,range_end_h=0.2)
"""

eta = 0.75
alfa = 0.9
lambd = 0
hidden = 3
trials = 100
weight = 0.7

start = time.time()
mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl \
    , std_error_MEE_vl = run_trials(17,X,Y,X_val,Y_val,500,TanhActivation(),SigmoidActivation(),
        eta=eta,alfa=eta,n_hidden= hidden,weight=weight,
           lambd=lambd,n_trials=trials,classification=True)
end = time.time()

print("TR MSE: %3f +- %s VL MSE: %3f +- %3f TR ACC %3f +- %3f VL ACC %3f +- %3f"%(
    float(mean_err_tr[-1]), float(std_err_tr[-1]),float(mean_err_vl[-1]),float(std_err_vl[-1]),
    float(mean_error_MEE_tr[-1]),float(std_error_MEE_tr[-1]),float(mean_error_MEE_vl[-1]),
    float(std_error_MEE_vl[-1])))

with open("../Risultati_Monk/Monk1_medio.csv","w") as f:
    f.write("TR MSE: %3f +- %s VL MSE: %3f +- %3f TR ACC %3f +- %3f VL ACC %3f +- %3f"%(
    float(mean_err_tr[-1]), float(std_err_tr[-1]),float(mean_err_vl[-1]),float(std_err_vl[-1]),
    float(mean_error_MEE_tr[-1]),float(std_error_MEE_tr[-1]),float(mean_error_MEE_vl[-1]),
    float(std_error_MEE_vl[-1])))

with open("../Risultati_Monk/Monk1_best.csv","w") as f:
    f.write("TR MSE: %3f VL MSE: %3f TR ACC %3f  VL ACC %3f"%(
    float(mlp.errors_tr[-1]), float(mlp.errors_vl[-1]),float(mlp.accuracies_tr[-1]),
    float(mlp.accuracies_vl[-1])
   ))

with open("../Risultati_Monk/Monk1_time.csv","w") as f:
    f.write("VECTORIZED TIME ELAPSED = %3f sec per epoch"%((end-start)/(500*trials)))

st = plt.suptitle("Monk 1\neta = %s alpha = %s, hidden = %s lambd = %s"%(eta,alfa,hidden,lambd))
plt.subplot(2,1,1)
plt.plot(mlp.errors_tr,label='Training Error',ls="-")
plt.plot(mlp.errors_vl,label='Validation Error',ls="dashed")
plt.ylabel('MSE')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})

plt.subplot(2,1,2)
plt.plot(mlp.accuracies_tr,label='Training Accuracy',ls="-")
plt.plot(mlp.accuracies_vl,label='Validation Accuracy',ls="dashed")
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})
plt.savefig("../Risultati_Monk/Monk1_best.jpg")


fig = plt.figure()
st = plt.suptitle("Monk 1\neta = %s alpha = %s, hidden = %s lambd = %s"%(eta,alfa,hidden,lambd))
plt.subplot(2, 1, 1)
plt.plot(mean_err_tr, label='Training Error', ls="-")
plt.plot(mean_err_vl, label='Validation Error', ls="dashed")
plt.fill_between(range(0, 501),np.reshape(mean_err_tr - std_err_tr, 501, -1),
                        np.reshape(mean_err_tr + std_err_tr, 501, -1),
                                        color="b", alpha=0.2)
plt.fill_between(range(0, 501),
np.reshape(mean_err_vl - std_err_vl, 501, -1),
        np.reshape(mean_err_vl + std_err_vl, 501, -1),
                            color="orange", alpha=0.2)
plt.ylabel('MSE')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})


plt.subplot(2, 1, 2)
plt.plot(mean_error_MEE_tr, label='Training Accuracy', ls="-")
plt.plot(mean_error_MEE_vl, label='Validation Accuracy', ls="dashed")
plt.fill_between(range(0, 501),np.reshape(mean_error_MEE_tr - std_error_MEE_tr, 501, -1),
                        np.reshape(mean_error_MEE_tr + std_error_MEE_tr, 501, -1),
                                        color="b", alpha=0.2)
plt.fill_between(range(0, 501),
np.reshape(mean_error_MEE_vl - std_error_MEE_vl, 501, -1),
        np.reshape(mean_error_MEE_vl + std_error_MEE_vl, 501, -1),
                            color="orange", alpha=0.2)
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})
plt.savefig("../Risultati_Monk/Monk1_medio.jpg")