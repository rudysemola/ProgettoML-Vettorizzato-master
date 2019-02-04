from Validation.KFold import *
from Utilities.Utility import *
X = np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3],
    [4,4,4],
    [5,5,5]
])
Y = np.ones((5,1))

folds = kFold(X,Y,3)
X_i ,Y_i = get_fold(X,Y,folds,0)
print(X_i)
print(Y_i)

split_dataset(X,Y,folds,0)

for (idx, fold_for_vl) in enumerate(folds):
    print("FOLD ", idx + 1)
    X_tr, T_tr, X_vl, T_vl = split_dataset(X, Y, folds, idx)
    print("X_tr=\n",X_tr)
    print("T_tr=\n", T_tr)
    print("X_vl=\n", X_vl)
    print("T_vl=\n",T_vl)