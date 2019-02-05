import sys

sys.path.append("../")

from Utilities.UtilityCM2 import *

"""
Calcola phi(eta) = f(w+eta*d)
Calcola phi'(eta) = <grad(w+eta*d),d>

:param w: punto iniziale
:param eta : stepsize
:param d : direzione in cui muoversi
:param mlp : Rete neurale
:param X : Matrice dati training
:param T: Matrice target di training

:return phi_eta : Valore di phi(eta)
:return phi_p_eta : Valore di phi'(eta)
"""


def compute_phi(w, eta, d, mlp, X, T):
    w_new = w + eta * d

    f, grad_f = evaluate_function(mlp, X, T, w_new)

    phi_eta = f

    phi_p_eta = np.dot(grad_f.T, d)

    return float(phi_eta), float(phi_p_eta)


"""
Controlla se la condizione di armijio è soddisfatta

:param eta : Stepsize
:param phi_0 : Valore di phi(0)
:param phi_p_0: Valore di phi'(0)
:param phi_eta : Valore di phi(eta)
:param m1 : Valore del parametro necessario per valutare condizione

:return armijo : Se true, indica che condizione di armijo è soddisfatta
"""


def check_armijo(eta, phi_0, phi_p_0, phi_eta, m1):
    assert m1 < 1
    assert m1 > 0

    armijo = phi_eta <= phi_0 + m1 * eta * phi_p_0
    return armijo


"""
Controlla se la condizione di strong wolfe è soddisfatta

:param phi_p_0: Valore di phi'(0)
:param phi_eta : Valore di phi(eta)
:param m2 : Valore del parametro necessario per valutare condizione

:return wolfe : Se true, indica che condizione di strong wolfe è soddisfatta
"""


def check_strong_wolfe(phi_p_0, phi_p_eta, m2):
    # print("[WOLFE] phi'(eta) = %3f m2*phi'(0) = %3f phi'(0) = %3f"%(abs(phi_p_eta),m2*abs(phi_p_0),abs(phi_p_0)))
    wolfe = (abs(phi_p_eta) <= m2 * abs(phi_p_0))
    return wolfe


"""
Effettua Armijo Wolfe Line Search"
"""


def AWLS(mlp, X, T, d, lambd, eta_start=1, eta_max=20, max_iter=100, m1=0.001, m2=0.9,
         tau=0.9, mina=1e-16, sfgrd=0.001, debug=False, l_bfgs=False, epsilon=1e-7):

    assert eta_start > 0
    assert eta_max > eta_start
    assert lambd >= 0
    assert tau < 1
    assert max_iter > 0
    assert m1 > 0 and m1 < 1
    assert m2 > 0 and m2 > m1

    """
    Condizioni di arresto
    """

    done_max_iters = False
    reached_eta_max = False
    wolfe_satisfied = False
    done_interpolation = False

    """
    Calcolo phi(0), phi'(0)
    """
    w = get_current_point(mlp)
    phi_0, phi_p_0 = compute_phi(w, 0, d, mlp, X, T)

    """
    Mi servono per mantenere informazioni sulle iterate durante lo svolgimento dell'algoritmo
    """
    eta_prec = 0  # metto eta_0 = 0
    phi_eta_prec = phi_0  # phi(eta_0) = phi(0)
    phi_p_eta_prec = phi_p_0
    eta = eta_start
    eta_star = eta_start

    it = 1
    arm_satisfied = False

    while (not done_max_iters) and (not reached_eta_max) and (not wolfe_satisfied) and (not done_interpolation):

        phi_eta, phi_p_eta = compute_phi(w, eta, d, mlp, X, T)

        if debug:
            print("[AWLS] Iterazione %s) Eta = %3f Eta_Max = %3f Phi(eta) =%3f Phi'(eta)=%3f" %
                  (it, eta, eta_max, phi_eta, phi_p_eta))

        if abs(phi_p_eta) <= epsilon:
            wolfe_satisfied = True

        else:
            if check_armijo(eta, phi_0, phi_p_0, phi_eta, m1):
                # print("Armijo soddisfatta")
                arm_satisfied = True

            if (not arm_satisfied) or (phi_eta >= phi_eta_prec and it > 1):
                eta_star, it_zoom = zoom(mlp, X, T, d, lambd, eta_prec, eta, phi_eta_prec, phi_eta,
                                         phi_p_eta_prec, phi_p_eta,
                                         phi_0, phi_p_0, m1, m2, max_iter - it, mina, sfgrd, l_bfgs)

                it += it_zoom
                done_interpolation = True

            else:
                if arm_satisfied and not done_interpolation:
                    if check_strong_wolfe(phi_p_0, phi_p_eta, m2):
                        # print("Wolfe soddisfatta")
                        eta_star = eta
                        wolfe_satisfied = True

                    elif phi_p_eta >= 0:
                        eta_star, it_zoom = zoom(mlp, X, T, d, lambd, eta, eta_prec, phi_eta, phi_eta_prec, phi_p_eta,
                                                 phi_p_eta_prec,
                                                 phi_0, phi_p_0, m1, m2, max_iter - it, mina, sfgrd, l_bfgs)
                        it += it_zoom
                        done_interpolation = True

                    else:
                        """
                        Aggiorno eta ed eta_prec
                        """
                        tmp = eta
                        eta = eta / tau
                        eta_prec = tmp
                        phi_eta_prec = phi_eta
                        phi_p_eta_prec = phi_p_eta

                        if eta > eta_max:
                            reached_eta_max = True
                            eta_star = eta_max

                        it += 1

                        if not l_bfgs:
                            if it >= max_iter:
                                done_max_iters = True
                                eta_star = eta

    """
    if done_max_iters:
        print("Raggiunto il numero massimo di iterazioni")
    elif done_interpolation:
        print("Effettuata interpolazione")
    elif reached_eta_max:
        print("Raggiunto eta massimo")
    elif wolfe_satisfied:
        print("Soddisfatte le condizioni di Armijo-Wolfe")
    """
    return eta_star, it


"""
Effettua la fase interpolazione
"""


def zoom(mlp, X, T, d, lambd, eta_l, eta_h, phi_eta_l, phi_eta_h, phi_p_eta_l, phi_p_eta_h, phi_0, phi_p_0, m1, m2,
         max_iters, mina, sfgrd, debug=False, l_bfgs=False):
    # print("ENTRO IN ZOOM")
    eta_low = eta_l
    eta_high = eta_h
    phi_eta_low = phi_eta_l
    phi_eta_high = phi_eta_h
    phi_p_eta_low = phi_p_eta_l
    phi_p_eta_high = phi_p_eta_h

    eta_star = eta_low

    satisfied_aw = False
    done_max_iters = False
    too_close = False

    it = 0
    while (not satisfied_aw) and (not done_max_iters) and (not too_close):
        """
        if debug:
            print("[ZOOM] Iterazione %s) Faccio interpolazione in [%s,%s]"%(it+1,eta_low,eta_high))
        """
        if abs(eta_high - eta_low) <= mina:
            eta_star = eta_low
            too_close = True

        else:
            eta = ((eta_low * phi_p_eta_high) - (eta_high * phi_p_eta_low)) / (phi_p_eta_high - phi_p_eta_low)
            eta = max([min([eta_low, eta_high]) * (1 + sfgrd), min([max([eta_low, eta_high]) * (1 - sfgrd), eta])])

            if debug:
                print("[ZOOM] Eta interpolato = ", eta)

            w = get_current_point(mlp)
            phi_eta, phi_p_eta = compute_phi(w, eta, d, mlp, X, T)

            if (not check_armijo(eta, phi_0, phi_p_0, phi_eta, m1)) or (phi_eta >= phi_eta_low):
                eta_high = eta
                phi_eta_high = phi_eta
                phi_p_eta_high = phi_p_eta

            else:

                if check_strong_wolfe(phi_p_0, phi_p_eta, m2):
                    eta_star = eta
                    satisfied_aw = True

                elif phi_p_eta * (eta_high - eta_low) >= 0:
                    eta_high = eta_low
                    phi_eta_high = phi_eta_low
                    phi_p_eta_high = phi_p_eta_low

                eta_low = eta
                phi_eta_low = phi_eta
                phi_p_eta_low = phi_p_eta

            it += 1

            if not l_bfgs:
                if it >= max_iters:
                    done_max_iters = True
                    eta_star = eta

    if debug:
        if too_close:
            print("[ZOOM] Terminato per intervallo troppo piccolo")

        elif done_max_iters:
            print("[ZOOM] Terminato il numero massimo di iterazioni")

        elif satisfied_aw:
            print("[ZOOM] Soddisfatte AW")

        print("[ZOOM] Eta restituito = ", eta_star)

    return eta_star, it
