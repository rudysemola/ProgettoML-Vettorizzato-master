"""
Questo file contiene varie funzioni utilizzate dagli altri moduli del progetto.

"""
import sys
sys.path.append("../")

import numpy as np
from sklearn.model_selection import train_test_split
from MLP import *
import os

"""
Spitto internal set: TR e VL set, ma su  M = [X, T]
Effettua lo shuffling delle due matrici X e T (ma su M!)
separo M

:parameter:
X: dataset TR+VL (NO TEST INTERNO!)
T: target TR+VL

:return: X_tr, T_tr, X_vl, T_vl
"""


def split_data_train_validation(X, T, test_size=0.25):
    M = np.concatenate((X, T), axis=1)
    M_tr, M_vl = train_test_split(M, test_size=test_size)  # shuffle=True , train_size=0.75
    X_shuffled_tr = M_tr[:, :X.shape[1]]
    T_shuffled_tr = M_tr[:, -T.shape[1]:]
    X_shuffled_vl = M_vl[:, :X.shape[1]]
    T_shuffled_vl = M_vl[:, -T.shape[1]:]
    return X_shuffled_tr, T_shuffled_tr, X_shuffled_vl, T_shuffled_vl

""""
Aggiunge il bias ad una generica matrice M.
A partire dalla matrice M, costruisce la matrice M' = [1 | M].
La matrice M passata in input non viene modificata.

:param M : Matrice di origine tramite cui comporre la nuova matrice M'.
:return : La matrice M' = [1 | M].
"""


def addBias(M):
    M_bias = np.ones((M.shape[0], M.shape[1] + 1))
    M_bias[:, 1:] = np.copy(M)
    return M_bias


"""
Rimuove il bias da una generica matrice M.
A partire dalla matrice M = [1 | M'], ricostruisce la matrice M' e la restituisce in output.
La matrice M non viene modificata.

:param M : Matrice a cui rimuovere il bias.
:return : Matrice M'.
"""


def removeBias(M):
    M_prime = np.copy(M[:, 1:])
    return M_prime


"""
Costruisce una matrice random di dimensione (n_rows * n_cols + 1) i cui elementi sono
compresi nel range [range_start, range_end].

----------------------------------
Viene aggiunta una colonna per il bias.
----------------------------------

Se fan_in == True, allora gli elementi sono compresi nell'intervallo [range_start / n_cols, range_end / n_cols]
Per le matrici dei pesi:
    n_rows = # elementi nel layer corrente 
    n_cols = # elementi nel layer precedente

Gli elementi sono generati a partire da una distribuzione uniforme sull'intervallo specificato.

--------------------------
NOTA: LA MATRICE DEI PESI CONTIENE GIA' IL BIAS!!!   ;)
-------------------------

:param n_rows : numero di righe
:param n_cols : numero di colonne
:param fan_in : bool. Se True, allora viene applicato il fan_in.
:param range_start : estremo di inizio dell'intervallo
:param range_end : estremo di fine dell'intervallo 
"""


def init_Weights(n_rows, n_cols, fan_in=False, range_start=-0.7, range_end=0.7):
    assert range_start < range_end
    assert n_rows > 0
    assert n_cols > 0

    if fan_in:
        M = np.random.uniform(range_start / n_cols, range_end / n_cols, (n_rows, n_cols + 1))
    else:
        M = np.random.uniform(range_start, range_end, (n_rows, n_cols + 1))

    return M


"""
Calcola l'errore misurato mediante l'MSE.

MSE calcolato come ([frobenius_norm(T - OUT)]**2 / n_examples)

:param T : Matrice dei Target
:param OUT : Matrice degli Output della rete neurale

:return : MSE(T,OUT)
"""


def compute_Error(T, OUT):
    assert T.shape == OUT.shape
    n_examples = T.shape[0]
    return 0.5 * (np.linalg.norm(T - OUT, 'fro') ** 2) / n_examples


"""
Calcola l'accuracy della rete neurale, dati target e predizioni della rete neurale.
L'accuracy e calcola come [ (numero di elementi classificati correttamente) / #esempi]
La matrice OUT contiene le predizioni della rete (0/1 nel caso di classificazione binaria)

:param T : Matrice contenente i target.
:param OUT: Matrice contenente le predizioni della rete
:return Accuracy

"""


def compute_Accuracy_Class(T, OUT):
    assert T.shape == OUT.shape
    n_examples = T.shape[0]

    n_misclass = np.sum((T - OUT) ** 2)
    n_correct_class = n_examples - n_misclass
    return n_correct_class / n_examples


"""
Data una generica matrice M, calcola la classe di ogni elemento.
Opera solo su "classificazione binaria" (0/1).
Se un elemento e maggiore o uguale a threshold, la classe assegnata e 1, 0 altrimenti.
La matrice M non viene modificata.

:param M : Matrice di cui si desidera calcolare la classe di ogni suo elemento
:param threshold : Soglia di decisione. Default = 0.5
:return M_prime: Matrice contenente la classificazione degli elementi di M

"""


def convert2binary_class(M, threshold=0.5):
    M_prime = np.zeros(M.shape)
    M_prime[M >= threshold] = 1
    return M_prime


"""
Calcola l'accuracy misurata come MEE.
FORMULA CON MATRICI!
D -> differenza matriciale Out -T;
Q -> quadrato elemento per elemento;
S -> sommo le righe (axis=1); // diventa un vettore
R -> radice quadrata elemento per elemento;
scal -> eseguo somma elementi vettore (axis=0);
mee = divido per N=num_pattern
"""


def compute_Regr_MEE(T, OUT):
    assert T.shape == OUT.shape
    n_examples = T.shape[0]
    """
    D = OUT - T
    Q = D ** 2
    S = np.sum(Q, 1)
    R = np.sqrt(S)
    """
    R = np.linalg.norm(OUT - T,axis=1)
    scal = np.sum(R, 0)
    mee = scal / n_examples

    return mee


"""
Salva la matrice M nel file specificato.
M viene salvata nel file usando la seguente formattazione:
numero righe di M
numero colonne di M
ogni riga nel file rappresenta una riga di M (gli elementi sono separati da una virgola)
:param filename : path del file su cui salvare la matrice
:param M : matrice da salvare
"""


def saveMatrix2File(filename, M):
    with open(filename, "w") as f:
        f.write(str(M.shape[0]) + "\n")  # scrivo numero di righe
        f.write(str(M.shape[1]) + "\n")  # scrivo numero di colonne

        for (row_idx, row) in enumerate(M):
            for (el_idx, element) in enumerate(row):

                # scrivo ogni elemento della riga row di M su questa riga, separando ogni elemento usando la virgola
                f.write(str(element))
                if not el_idx == M.shape[1] - 1:
                    f.write(",")
                else:
                    f.write("\n")

    return


"""
Carico la matrice dal file specificato
:param filename : path del file da cui caricare la matrice
:return M : Matrice letta dal file
(DI PROVA: NON ANCORA COME VUOLE IL PROFESSORE!)
"""


def loadMatrixFromFile(filename):
    with open(filename) as f:

        n_rows = 0
        n_cols = 0
        M = None
        current_row = 0

        for (idx, line) in enumerate(f):

            if idx == 0:
                n_rows = int(line.rstrip("\n"))

            elif idx == 1:
                n_cols = int(line.rstrip("\n"))

                M = np.zeros((n_rows, n_cols))

            else:
                elements = line.rstrip("\n").split(",")
                current_col = 0
                for element in elements:
                    M[current_row, current_col] = float(element)
                    current_col += 1

                current_row += 1

        return M


"""
Effettua lo shuffling delle due matrici X e T
"""


def shuffle_matrices(X, T):
    M = np.concatenate((X, T), axis=1)
    np.random.shuffle(M)
    X_shuffled = M[:, :X.shape[1]]
    T_shuffled = M[:, -T.shape[1]:]
    return np.reshape(X_shuffled, (-1, X.shape[1])), np.reshape(T_shuffled, (-1, T.shape[1]))


"""
Analizza i file csv prodotti come risultati dalla kFoldCV.
Trova il file contenente il minore errore di validazione.
Restituisce il nome del file e il suo vl error

:param path_folder: Path della cartella in cui sono contenuti i csv da analizzare
:param n_folds: numero di folds usati nella kCV (parametro k)
 
"""
"""
def analyze_result_csv(path_folder,n_folds):

    best_vl_error = 1e5
    best_file = ""
    path = path_folder+"/"
    fileList = os.listdir(path_folder)
    for file in fileList:

       if not file == "__init__.py":
           with open(path+file) as f:
            print(path+file)
            for i in range(n_folds +2):
                f.readline()

            line_result = f.readline()
            line=line_result.rstrip("\n").split(" ")
            #print(line[-3:])
            vl_error = float(line[-3])

            if vl_error < best_vl_error:
                best_vl_error = vl_error
                best_file = path+file

    return best_file, best_vl_error
"""