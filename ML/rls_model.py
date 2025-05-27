import numpy as np

def compute_distances(X, R):
    # Euclidean distances entre X (N x D) et R (K x D)
    N = X.shape[0]
    K = R.shape[0]
    D = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            D[i, k] = np.linalg.norm(X[i] - R[k])
    return D


def init_rls(Dx0, Dy0):
    P0 = np.linalg.inv(Dx0.T @ Dx0)
    B0 = P0 @ Dx0.T @ Dy0
    return P0, B0


def update_rls(Dx_i, Dy_i, P_prev, B_prev):
    Px_i = P_prev @ Dx_i.T
    I = np.eye(Dx_i.shape[0])
    denom_inv = np.linalg.inv(I + Dx_i @ Px_i)
    P_i = P_prev - Px_i @ denom_inv @ Dx_i @ P_prev
    K_i = P_i @ Dx_i.T
    B_i = B_prev + K_i @ (Dy_i - Dx_i @ B_prev)
    return P_i, B_i
