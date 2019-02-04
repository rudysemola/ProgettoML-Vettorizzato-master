import sys
sys.path.append("../")

class Training:
    """
    Specifica la segnatura del metodo train usato dai vari ottimizzatori.

    :param mlp: Modello di cui fare il training
    :param X : Matrice contenente i dati di training
    :param T: Matrice con i target dei dati di training
    :param X_val Matrice contenente i dati di validation
    :param T_val: Matrice con i target dei dati di validation
    :param n_epochs : Numero massimo di epoche
    :param eps : Valore per indicare accuratezza massima desiderata. Usata solo per BPLS E L-BFGS
    :param threshold : Indica il threshold da usare per classificare
    :param suppress_print: Se true, stampa solo informazioni necessarie e non di debugging
    """
    def train(self,mlp, X, T, X_val, T_val, n_epochs=1000, eps=1e-6, threshold=0.5, suppress_print=False):
        raise NotImplementedError("Metodo astratto")