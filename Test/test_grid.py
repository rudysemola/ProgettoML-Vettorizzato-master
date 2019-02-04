from Monks.Monk import *
from matplotlib import pyplot as plt
from Validation.GridSearch import *
from MLP.Activation_Functions import *
from MLP.MLP import *

eta_values = [0.7,0.8]
alfa_values = [0.7,0.8]
hidden_values =[2,3]
weight_values = [0.7]
lambda_values = [0,0.01,0.03]

n_trials = 10
X1, Y1 = load_monk("../Datasets/monks-1.train")
X_val1, Y_val1 = load_monk("../Datasets/monks-1.test")

X2, Y2 = load_monk("../Datasets/monks-2.train")
X_val2, Y_val2 = load_monk("../Datasets/monks-2.test")

X3, Y3 = load_monk("../Datasets/monks-3.train")
X_val3, Y_val3 = load_monk("../Datasets/monks-3.test")


best_mlp,best_mean_err_tr,best_std_err_tr,best_mean_acc_tr,best_std_acc_tr,best_mean_err_vl,best_std_err_vl,best_mean_acc_vl,best_std_acc_vl = gridSearch(
    17,X3,Y3,X_val3,Y_val3,500,
    TanhActivation(),SigmoidActivation(),
    eta_values,alfa_values,hidden_values,weight_values,lambda_values,n_trials)

st = plt.suptitle("Monk 1(Mean Curve)\neta=%s alpha=%s lambda=%s n_hidden=%s"%(best_mlp.eta,best_mlp.alfa,best_mlp.lambd,best_mlp.n_hidden))
plt.subplot(2, 1, 1)
plt.plot(best_mean_err_tr,label='Training Error',ls="-")
plt.plot(best_mean_err_vl,label='Validation Error',ls="dashed")


plt.fill_between(range(0,501),np.reshape(best_mean_err_tr - best_std_err_tr,501,-1),np.reshape(best_mean_err_tr + best_std_err_tr,501,-1),
                 color="b",alpha=0.4)


plt.fill_between(range(0,501),np.reshape(best_mean_err_vl - best_std_err_vl,501,-1),np.reshape(best_mean_err_vl + best_std_err_vl,501,-1),
               color="orange",alpha=0.4)


plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})
plt.subplot(2, 1, 2)
plt.plot(best_mean_acc_tr,label='Training Accuracy',ls="-")
plt.plot(best_mean_acc_vl,label='Validation Accuracy',ls="dashed")


plt.fill_between(range(0,501),np.reshape(best_mean_acc_tr - best_std_acc_tr,501,-1),np.reshape(best_mean_acc_tr + best_std_acc_tr,501,-1),
                 color="b",alpha=0.4)


plt.fill_between(range(0,501),np.reshape(best_mean_acc_vl - best_std_acc_vl,501,-1),np.reshape(best_mean_acc_vl + best_std_acc_vl,501,-1),
               color="orange",alpha=0.4)


plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})
plt.show()

st = plt.suptitle("Monk 1(Best model)\neta=%s alpha=%s lambda=%s n_hidden=%s"%(best_mlp.eta,best_mlp.alfa,best_mlp.lambd,best_mlp.n_hidden))
plt.subplot(2, 1, 1)
plt.plot(best_mlp.errors_tr,label='Training Error',ls="-")
plt.plot(best_mlp.errors_vl,label='Validation Error',ls="dashed")
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})
plt.subplot(2, 1, 2)
plt.plot(best_mlp.accuracies_tr,label='Training Accuracy',ls="-")
plt.plot(best_mlp.accuracies_vl,label='Validation Accuracy',ls="dashed")
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})
plt.show()