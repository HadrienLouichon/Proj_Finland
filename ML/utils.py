import numpy as np

def predict(X_new, R, B):
    Dx_new = np.array([[np.linalg.norm(x - r) for r in R] for x in X_new])
    Dy_new = Dx_new @ B  # distances aux étiquettes de référence
    y_pred = np.argmin(Dy_new, axis=1) + 1  # les classes sont numérotées à partir de 1
    return y_pred
