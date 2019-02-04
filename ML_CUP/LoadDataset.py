import sys
sys.path.append("../")

import numpy as np
"""
Carica il dataset di TR della cup da filename.
:param filename: path file da cui caricare dataset
:param row_to_ignore: numero di righe iniziali da non considerare
"""

def load_cup_dataset(filename,n_rows_to_ignore = 10):
    """
    Carica il dataset della MLCUP.

    :param filename: path del file da caricare
    :return X : Matrice degli input (n_examples, n_features)
    :return T : Matrice dei target (n_examples, 2)
    """

    with open(filename) as f:

        current_row = 1
        patterns_loaded = [] #contiene i vari esempi letti fino ad ora
        n_examples = 0

        for line in f:

            #deve ignorare le prime righe nel caricare il dataset
            if current_row <= n_rows_to_ignore:
                current_row += 1

            else:

                #legge una riga e la processa
                example = line.rstrip("\n").split(",")[1:]
                n_examples += 1


                #aggiunge l'esempio letto in patterns_loaded
                patterns_loaded.append(example)


    # Faccio diventare patterns_loaded una matrice (n_esempi, -1)

    patterns_loaded = np.reshape(np.array(patterns_loaded), (n_examples,-1))

    # Converto patterns_loaded da matrice di stringhe a matrice di float.
    # Metto il risultato in M
    M = np.zeros(patterns_loaded.shape)

    for (row_idx,row) in enumerate(patterns_loaded):
        for (col_idx,pattern) in enumerate(row):

           M[row_idx,col_idx] = float(pattern)

    # Splitto M in X e T

    X = M[:,:-2]
    T = M[:,-2:]

    return X, T


"""
Carica il dataset di TS della cup da filename.
:param filename: path file da cui caricare dataset
:param row_to_ignore: numero di righe iniziali da non considerare
"""
def load_cup_dataset_blind(filename,n_rows_to_ignore = 10):
    """
    Carica il dataset della MLCUP.

    :param filename: path del file da caricare
    :return X : Matrice degli input (n_examples, n_features)
    :return T : Matrice dei target (n_examples, 2)
    """

    with open(filename) as f:

        current_row = 1
        patterns_loaded = [] #contiene i vari esempi letti fino ad ora
        n_examples = 0

        for line in f:

            #deve ignorare le prime righe nel caricare il dataset
            if current_row <= n_rows_to_ignore:
                current_row += 1

            else:

                #legge una riga e la processa
                example = line.rstrip("\n").split(",")[1:]
                n_examples += 1


                #aggiunge l'esempio letto in patterns_loaded
                patterns_loaded.append(example)


    # Faccio diventare patterns_loaded una matrice (n_esempi, -1)

    patterns_loaded = np.reshape(np.array(patterns_loaded), (n_examples,-1))

    # Converto patterns_loaded da matrice di stringhe a matrice di float.
    # Metto il risultato in M
    M = np.zeros(patterns_loaded.shape)

    for (row_idx,row) in enumerate(patterns_loaded):
        for (col_idx,pattern) in enumerate(row):

           M[row_idx,col_idx] = float(pattern)

    return M





