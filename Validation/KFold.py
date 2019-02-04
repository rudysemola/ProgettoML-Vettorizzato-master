import sys
sys.path.append("../")
import numpy as np

"""
Divide in k parti le matrici X degli input e T degli output.
Restituisce una lista di k elementi contenente come elementi una tupla, che rappresenta l'intervallo [start_i, end_i).
[start_i, end_i) rappresenta gli estremi dell'intervallo degli indici da cui estrarre l'i-esimo fold.

Ad esempio, se [start_i, end_i) = [3,5), allora per estrarre tutti e soli gli elementi di questo fold, e sufficiente fare
X[3:5]. Stessa cosa con la matrice dei target.

In questo modo si risparmia spazio, poiche non si memorizzano esplicitamente le matrici di ogni singolo fold ma si tiene traccia solo degli indici
che delimitano il fold.

:param X : Matrice degli input
:param T : Matrice degli output
:param k : intero > 0 che specifica in quante parti vanno divise le matrici.

:return folds: Lista che contiene gli intervalli rappresentanti ciascun fold 
"""


def kFold(X, T, k):
    assert k > 1, "k deve essere maggiore di 1"
    assert X.shape[0] == T.shape[0], "Le due matrici devono avere lo stesso numero di righe"

    n_examples = X.shape[0]
    assert k <= n_examples, "k deve essere minore o uguale al numero di righe delle due matrici" \
                            ""
    fold_dim = int(np.floor(n_examples / k))
    folds_idx = []

    start_i = 0

    """
    Ogni fold, tranne l'ultimo, appartiene a [start_i, start_i + fold_dim)
    L'ultimo fold appartiene a [start_i, end)
    
    """
    for fold_i in range(k):

        if not fold_i == k - 1:
            folds_idx.append((start_i, start_i + fold_dim))
            start_i += fold_dim
        else:
            folds_idx.append((start_i, n_examples))

    return folds_idx


"""
Restituisce il contenuto dell'i-esimo fold.

:param X : Matrice di input
:param T : Matrice dei target
:param folds: lista contenente gli estremi dell'intervallo di ogni fold
:param i : indice del fold da estrarre

:return X_i : sottomatrice di X corrispondente al fold i
:return T_i : sottomatrice di T corrispondente al fold i
"""


def get_fold(X, T, folds, i):
    assert i >= 0, "i deve essere maggiore o uguale a zero"
    assert i < len(folds), "i deve essere minore del numero di folds"

    fold_i = folds[i]
    start_i = fold_i[0]
    end_i = fold_i[1]

    return X[start_i:end_i, :], T[start_i:end_i, :]


"""
Dato l'indice del fold da usare come VL set, costruisce il TR set ed il VL set.

:param X : Matrice input
:param T : Matrice target
:param folds : Rappresentazione dei folds come intervalli
:param idx_vl : indice del fold da usare come VL set

:return X_tr : Sottomatrice di X da usare come TR set
:return T_tr : Sottomatrice di T da usare come TR set
:return X_vl : Sottomatrice di X da usare come VL set
:return T_vl : Sottomatrice di T da usare come VL set


"""


def split_dataset(X, T, folds, idx_vl):
    assert idx_vl >= 0, "l'indice del fold per vl deve essere maggiore o uguale a zero"
    assert idx_vl < len(folds), "indice del fold per vl deve essere minore del numero di folds"

    "Validation set"
    X_vl, T_vl = get_fold(X, T, folds, idx_vl)

    "Ottengo il TR set come X_tr = X - X_vl e T_tr=T-T_vl"
    fold_interval = folds[idx_vl]

    rows_to_delete = range(fold_interval[0], fold_interval[1])
    X_tr = np.delete(X, rows_to_delete, axis=0)
    T_tr = np.delete(T, rows_to_delete, axis=0)

    return X_tr, T_tr, X_vl, T_vl
