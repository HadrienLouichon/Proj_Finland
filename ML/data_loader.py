import numpy as np
import scipy.io as sp
from sklearn.preprocessing import StandardScaler


def load_salinas_data(corrected_path, gt_path):
    data = sp.loadmat(corrected_path)['salinasA_corrected']
    gt = sp.loadmat(gt_path)['salinasA_gt']
    return data, gt


def prepare_data(data, gt, labeled_only=True):
    h, w, d = data.shape
    data = data.reshape(-1, d)
    labels = gt.reshape(-1)

    if labeled_only:
        mask = labels > 0
        data = data[mask]
        labels = labels[mask]

    # Normalisation
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, labels


def split_reference(X, y, num_refs=10):
    # Prend `num_refs` exemples par classe comme référence
    ref_idx = []
    unique_labels = np.unique(y)
    for label in unique_labels:
        class_idx = np.where(y == label)[0]
        np.random.shuffle(class_idx)
        ref_idx.extend(class_idx[:num_refs])

    R = X[ref_idx]
    T = y[ref_idx]
    return R, T, ref_idx
