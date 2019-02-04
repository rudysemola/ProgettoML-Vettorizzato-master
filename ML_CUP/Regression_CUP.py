import sys
sys.path.append("../")

"""
File per eseguire lo splitting dei dati
Da fare solo una volta!!!

    - TS DA NON TOCCARE FINO ALLA FINE!!!
    - TR+VL  da usare per la model selection con tenciche di validazione a scelta (HoldOut o KCV)

"""

from ML_CUP.LoadDataset import *
from Utilities.Utility import *

X,T = load_cup_dataset("ML-CUP18-TR.csv")
X_trvl, T_trvl, X_ts, T_ts = split_data_train_validation(X, T, 216)
M_ts = np.concatenate((X_ts, T_ts), axis=1)
M_trvl = np.concatenate((X_trvl, T_trvl), axis=1)

saveMatrix2File("TestSetInterno.csv", M_ts)
saveMatrix2File("DatasetTrVl.csv", M_trvl)
