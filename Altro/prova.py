from Trainers.TrainBackPropLS_old import *
from Monks.Monk import *
from MLP.MLP import *
from MLP.Activation_Functions import *

def but_the_heidi_is(X,Y):

    sum = 0
    Y = Y.T

    for i in range(X.shape[0]):
        sum += X[i,:] * Y[:,i]

    return sum


X = np.random.rand(2,2)
Y = np.random.rand(2,3)

X_vett = np.reshape(X,(-1,1))
Y_vett = np.reshape(Y,(-1,1))
Z = np.concatenate((X_vett,Y_vett),axis = 0)

print("X matrice",X)
print("Y matrice",Y)
print("Z=",Z.T)

X, Y = load_monk("../Datasets/monks-2.train")
X_val, Y_val = load_monk("../Datasets/monks-2.test")
n_features = 17
n_hidden = 3
n_out = 1
eta = 0.7
alpha = 0.7
lambd = 0

mlp = MLP(n_features, n_hidden, n_out, TanhActivation(), SigmoidActivation(), lambd=lambd, eta=eta, alfa=alpha,
          trainer=TrainBackPropLS())
mlp.trainer.train(mlp, addBias(X), Y, addBias(X_val), Y_val, 500, 1e-6)






