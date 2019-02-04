import sys
sys.path.append("../")
from MLP.MLP import *
from Utilities.Utility import *

"""
PER CM
"""

"""
Mette due matrici M, N di dimensione rispettivamente pari a m,n, in un unico vettore di dimensione (m+n,1)
"""

def compute_obj_function(mlp,X,T,lambd):
    mlp.feedforward(X)
    mse = compute_Error(T,mlp.Out_o)
    norm_w = np.linalg.norm(mlp.W_h)**2 + np.linalg.norm(mlp.W_o)**2
    loss = mse + (0.5*lambd* norm_w)
    return loss

"""
PER CM
"""
def compute_gradient(mlp,X, T,lambd):

    m_grad_mse_o, m_grad_mse_h = mlp.backpropagation(X,T)
    grad_mse_o = - m_grad_mse_o
    grad_mse_h = - m_grad_mse_h
    grad_o = grad_mse_o + (lambd * mlp.W_o)
    grad_h = grad_mse_h + (lambd * mlp.W_h)
    return grad_h, grad_o

"""
Trasforma 2 matrici in un unico vettore [X|Y]
"""
def matrix2vec(X,Y):
    X_vett = np.reshape(X, (-1, 1))
    Y_vett = np.reshape(Y, (-1, 1))
    vect = np.concatenate((X_vett, Y_vett), axis=0)
    return vect

"""
Bias già inserito
Trasforma due vettori in una unica matrice
"""
def vec2matrix(X,shape_h,shape_o):
    W_h = X[:(shape_h[0] * shape_h[1])]
    W_o = X[-(shape_o[0] * shape_o[1]):]
    W_h = np.reshape(W_h, (shape_h[0], shape_h[1]))
    W_o = np.reshape(W_o, (shape_o[0], shape_o[1]))

    return W_h, W_o


