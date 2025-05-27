import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

def display_salinas_image(mat_path, bands=(10, 130, 160)):
    """
    Affiche une image en fausses couleurs à partir de 3 bandes hyperspectrales.

    Parameters:
        mat_path (str): Chemin vers le fichier SalinasA_corrected.mat
        bands (tuple): Indices des bandes à utiliser pour RGB (entre 0 et 203)
    """
    data = sp.loadmat(mat_path)['salinasA_corrected']  # shape: (83, 86, 204)

    # Normalisation entre 0 et 1 pour chaque bande
    rgb = np.stack([data[:, :, b] for b in bands], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"Salinas-A (Bandes RGB = {bands})")
    plt.axis("off")
    plt.show()
