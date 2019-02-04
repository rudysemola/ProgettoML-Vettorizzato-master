from Utilities.UtilityCM import *
from MLP.MLP import *
from Monks.Monk import *
from MLP.Activation_Functions import *
"""
def matrix2vec(X,Y):
    X_vett = np.reshape(X, (-1, 1))
    Y_vett = np.reshape(Y, (-1, 1))
    vect = np.concatenate((X_vett, Y_vett), axis=0)
    return vect

def compute_obj_function(mlp,X,T,lambd):
    mlp.feedforward(X)
    mse = compute_Error(T,mlp.Out_o)
    norm_w = np.linalg.norm(mlp.W_h)**2 + np.linalg.norm(mlp.W_o)**2
    loss = mse + (0.5*lambd* norm_w)
    return loss


PER CM

def compute_gradient(mlp,X, T,lambd):

    m_grad_mse_o, m_grad_mse_h = mlp.backpropagation(X,T)
    grad_mse_o = - m_grad_mse_o
    grad_mse_h = - m_grad_mse_h
    grad_o = grad_mse_o + (lambd * mlp.W_o)
    grad_h = grad_mse_h + (lambd * mlp.W_h)
    return grad_h, grad_o
"""
"""
X ha già il bias
gradE = gradiente di E (NON MENO GRADIENTE)
phi_p_eta = <gradE_new_vec, -gradE_vec)>

"""
def f2phi(eta,mlp,X,T,gradE_h,gradE_o,lambd):

    #PESI ATTUALI
    W_h_current = mlp.W_h
    W_o_current = mlp.W_o

    print("W_h",W_h_current)
    print("W_o",W_o_current)

    #SPOSTO I PESI LUNGO DELTA_W ( = - GRADIENTE)
    mlp.W_h = mlp.W_h - (eta * gradE_h)
    mlp.W_o = mlp.W_o - (eta * gradE_o)

    #CALCOLO E(w + alpha* delta_W)
    phi_eta = compute_obj_function(mlp,X,T,lambd)

    # GRADIENTE CALCOLATO NEL NUOVO PUNTO
    gradE_h_new, gradE_o_new = compute_gradient(mlp,X,T,lambd)

    #METTO I GRADIENTI SOTTO FORMA DI VETTORE PER POTER FARE IL PRODOTTO SCALARE

    gradE_vec = matrix2vec(gradE_h,gradE_o)
    gradE_new_vec = matrix2vec(gradE_h_new,gradE_o_new)

    phi_p_eta = float(np.dot(gradE_new_vec.T, -gradE_vec))

    #RIMETTO I PESI COME ERANO ALL'INIZIO DELLA FUNZIONE
    mlp.W_h = W_h_current
    mlp.W_o = W_o_current

    print("W_h dopo phi", mlp.W_h)
    print("W_o dopo phi", mlp.W_o)

    return phi_eta, phi_p_eta


"""
Controlla se la condizione di armijio è soddisfatta
"""
def check_armijio(phi_zero,phi_prime_zero,phi_alpha,m1,alpha):

    assert m1 < 1
    assert m1 > 0

    return phi_alpha <= (phi_zero + m1*alpha*phi_prime_zero)

"""
Controlla se la condizione di strong Wolfe è soddisfatta
"""
def check_strong_wolfe(phi_prime_alpha,phi_prime_zero,m2):

    assert m2 < 1
    assert m2 > 0

    return abs(phi_prime_alpha) <= -m2 * phi_prime_zero


"""
AWLS
 NOTA: ordine dW_h, dW_o come matrici sul foglio...
"""
"""
def AWLS(mlp, X, T, loss, grad_W_h, grad_W_o, lambd, alpha_0=0.01, max_it=100, m1=0.001, m2=0.8, tau=0.925, epsilon=1e-6,mina=1e-12):
    phi_0 = loss
    gradE = vectorize(grad_W_h, grad_W_o)
    phi_p_0 = - np.linalg.norm(gradE)**2
    alpha = alpha_0
    do_quadratic_int = False

    for it in range(max_it):
        phi_alpha, phi_p_alpha = f2phi(alpha, mlp, X, T, grad_W_h, grad_W_o,lambd) #TODO... modifica func f2phi...
        arm = check_armijio(phi_0, phi_p_0, phi_alpha, m1, alpha)
        s_wolf = check_strong_wolfe(phi_p_alpha, phi_p_0, m2)

        if arm and s_wolf: # love by RS e sopprattutto Michele =) (NON LO STIA A SENTI' PROFFE !!!!!)
            print("Soddisfatta AW")
            break

        if phi_p_alpha >= 1e-12:
            #   chiama Quadratic interpolation!
            print("phi_p_alpha >= 0")
            do_quadratic_int = True
            break

        print("Iterazione %s: Alpha = %s" % (it, alpha))
        alpha = alpha / tau

    if do_quadratic_int:
        print("Iterazione %s: Alpha = %s" % (it, alpha))
        alpha = quadratic_interpolation(alpha, mlp, X, T, grad_W_h, grad_W_o, gradE, lambd,phi_0, phi_p_0, m1, m2, max_it,
                                    epsilon, mina)

    if not do_quadratic_int:
        print("Fine numero massimo iterazioni")

    return alpha


def quadratic_interpolation(alpha_0, mlp, X, T, grad_W_h, grad_W_o,gradE,lambd,phi_0,phi_p_0,
                            m1=0.001,m2=0.9,max_it=100,epsilon = 1e-6,mina=1e-12, sfgrd =0.01):

    alpha_sx = 0
    alpha_dx = alpha_0
    phi_p_sx = phi_p_0
    phi_dx,phi_p_dx = f2phi(alpha_dx, mlp, X, T, grad_W_h, grad_W_o,lambd)
    # norm_gradE = np.linalg.norm(gradE)
    epsilon_prime = 1e-12
    alpha = alpha_0

    print("[QUAD INT]Faccio interpolazione in [0,%s]"%(alpha_0))
    for it in range(max_it):

        if math.fabs(phi_p_dx) <= epsilon_prime:
            print("[QUAD INT]Trovato ottimo: derivata dx = ",math.fabs(phi_p_dx))
            break

        if alpha_dx - alpha_sx <= mina:
            print("[QUAD INT]Alpha sx e dx troppo vicini")
            break

        alpha = ((alpha_sx * phi_p_dx) - (alpha_dx* phi_p_sx)) / (phi_p_dx  - phi_p_sx)
        # a = max( [ am * ( 1 + sfgrd ) min( [ as * ( 1 - sfgrd ) a ] ) ] );
        alpha = max([alpha_sx*(1+sfgrd), min([alpha_dx*(1-sfgrd), alpha])])
        print("[QUAD INT]Iterazione %s) Alpha Interpolato = %s"%(it,alpha))

        phi_alpha, phi_p_alpha = f2phi(alpha, mlp, X, T, grad_W_h, grad_W_o,lambd)

        arm = check_armijio(phi_0, phi_p_0, phi_alpha, m1, alpha)
        s_wolf = check_strong_wolfe(phi_p_alpha, phi_p_0, m2)

        if arm and s_wolf:  # love by RS e sopprattutto Michele =) (NON LO STIA A SENTI' PROFFE !!!!!)
            print("[QUAD INT] Soddisfatto AW")
            break

        if phi_p_alpha < 0:
            alpha_sx = alpha
            phi_p_sx = phi_p_alpha

        else:

            alpha_dx = alpha
            phi_p_dx = phi_p_alpha
            if alpha_dx <= mina:
                print("[QUAD INT]Alpha dx troppo piccolo")
                break

    return alpha
"""
"""
-----------------
TEST VARI.....
----------------
"""
if __name__ == '__main__':
    M = np.array([
        [1,2,3],
        [4,5,6]
    ])

    N = np.array([
        [7,8],
        [9,10],
        [11,12],
        [13,14]
    ])

    #print(vectorize(M,N))
    #print(vectorize(M,N).shape)


    X = np.array([
        [1, -2,3],
        [4, -5,6]
    ])

    Y = np.array([
        [0,1],
        [1,0]
    ])

    X, Y = load_monk("monks-2.train")
    X_val, Y_val = load_monk("monks-2.test")
    n_features = 17
    n_hidden = 3
    n_out = 1
    eta = 0.7
    alpha = 0.7
    lambd = 0

    mlp = MLP(n_features,n_hidden,n_out,TanhActivation(),SigmoidActivation(),lambd=lambd,eta=eta,alfa=alpha,trainer=TrainBackprop())
    mlp.trainer.train(mlp,addBias(X), Y, addBias(X_val), Y_val, 500, 1e-4)
    """"
    print("Wh =\n",mlp.W_h)
    print("Wo =\n",mlp.W_o)
    print ("size Wh:", mlp.W_h.shape)
    print ("size Wo:", mlp.W_o.shape)
    mlp.feedforward(addBias(X))
    err = compute_Error(Y,mlp.Out_o)
    print("Errore = ",err)
    grad_W_o,grad_W_h = mlp.backpropagation(addBias(X),Y)
    print("gradWh =\n",grad_W_h)
    print("gradWo =\n",grad_W_o)
    print ("size gradWh:", grad_W_h.shape)
    print ("size gradWo:", grad_W_o.shape)

    phi, phip = f2phi(alpha,mlp,X,Y,grad_W_h,grad_W_o)
    print("phi(alpha) ", phi)
    print("phi'(alpha) ", phip)


    a = check_armijio(0,1,0.1,0.11,0.2)
    print(a)

    a = check_strong_wolfe(0, 1, 0.1)
    print(a)
    """
""""
    mlp.feedforward(addBias(X))
    grad_W_h, grad_W_o  =  compute_gradient(mlp,X, Y,lambd)
    loss = compute_obj_function(mlp,X,Y,lambd)
    alpha = AWLS(mlp, X, Y, loss, grad_W_h, grad_W_o,lambd,alpha_0=alpha, m1=0.001)
    print("Miglior alpha con AWLS= ",alpha)
"""

