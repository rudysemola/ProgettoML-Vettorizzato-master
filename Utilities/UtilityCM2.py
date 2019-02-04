from Utilities.Utility import *
import sys

sys.path.append("../")
"""
Trasforma 2 matrici in un unico vettore [X|Y]
"""


def matrix2vec(X, Y):
    X_vett = np.reshape(X, (-1, 1))
    Y_vett = np.reshape(Y, (-1, 1))
    vect = np.concatenate((X_vett, Y_vett), axis=0)
    return vect


"""
Trasforma due vettori in una unica matrice
"""


def vec2matrix(X, shape_h, shape_o):
    W_h = X[:(shape_h[0] * shape_h[1])]
    W_o = X[-(shape_o[0] * shape_o[1]):]
    W_h = np.reshape(W_h, (shape_h[0], shape_h[1]))
    W_o = np.reshape(W_o, (shape_o[0], shape_o[1]))

    return W_h, W_o


"""
Calcola funzione obiettivo e relativo gradiente nel punto w
:param mlp : Rete neurale
:param X : Matrice dei dati di training
:param T : Matrice dei target di training
:param w : Punto in cui calcolare funzione e gradiente

:return f: Valore della funzione in w
:return grad_f: Gradiente della funzione in w
"""


def evaluate_function(mlp, X, T, w):
    W_h_init = np.copy(mlp.W_h)
    W_o_init = np.copy(mlp.W_o)

    # Mi metto nel punto w
    W_h, W_o = vec2matrix(w, mlp.W_h.shape, mlp.W_o.shape)
    mlp.W_h = W_h
    mlp.W_o = W_o

    # Valuto funzione
    mlp.feedforward(X)
    mse = compute_Error(T, mlp.Out_o)
    norm_w = np.linalg.norm(mlp.W_h) ** 2 + np.linalg.norm(mlp.W_o) ** 2
    loss = mse + (0.5 * mlp.lambd * norm_w)

    # Valuto gradiente
    m_grad_mse_o, m_grad_mse_h = mlp.backpropagation(X, T)
    grad_mse_o = - m_grad_mse_o
    grad_mse_h = - m_grad_mse_h

    grad_o = grad_mse_o + (mlp.lambd * mlp.W_o)
    grad_h = grad_mse_h + (mlp.lambd * mlp.W_h)

    grad = matrix2vec(grad_h, grad_o)

    mlp.W_h = W_h_init
    mlp.W_o = W_o_init

    return loss, grad


"""
Sposta i pesi della rete nel punto w
:param w : Punto in cui mettere i pesi della rete
"""


def update_weights(mlp, w):
    W_h, W_o = vec2matrix(w, mlp.W_h.shape, mlp.W_o.shape)
    mlp.W_h = W_h
    mlp.W_o = W_o
    return


"""
Restituisce il punto in cui si trovano i pesi di mlp

:param mlp : Rete neurale

:return w: Vettore dei pesi attuali.
"""


def get_current_point(mlp):
    w = matrix2vec(mlp.W_h, mlp.W_o)
    return w
