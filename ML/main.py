import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import load_salinas_data
from rls_model import compute_distances, init_rls, update_rls
from utils import predict
from visualize import display_salinas_image

# Chargement et préparation des données
X, y = load_salinas_data("ML/SalinasA_corrected.mat", "ML/SalinasA_gt.mat")
#print("X :", X)
#print("y :", y) 
# ⚠️ Supprimer les classes avec moins de 2 instances
unique, counts = np.unique(y, return_counts=True)
#print(unique, counts)
valid_classes = unique[counts >= 2]
#print("Valid classes:", valid_classes)
mask = np.isin(y, valid_classes)
#print(mask)
X = X[mask]
y = y[mask]
print("X :",X)
print(X.shape, y.shape)

display_salinas_image("ML/SalinasA_corrected.mat")

# Split avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Sélection des points de référence (1 par classe)
unique_classes = np.unique(y_train)
reference_idx = np.concatenate([
    np.random.choice(np.where(y_train == c)[0], size=min(3, np.sum(y_train == c)), replace=False)
    for c in unique_classes
])
R = X_train[reference_idx]
T = y_train[reference_idx]

# Matrices de distances initiales
Dx0 = compute_distances(X_train, R)
Dy0 = compute_distances(y_train.reshape(-1,1), T.reshape(-1,1))

# Initialisation du modèle RLS
P, B = init_rls(Dx0, Dy0)

# Boucle de mise à jour récursive (simulée)
for i in range(len(X_train)):
    xi = X_train[i:i+1]
    yi = y_train[i:i+1]
    dx_i = compute_distances(xi, R)
    dy_i = compute_distances(yi.reshape(-1,1), T.reshape(-1,1))
    P, B = update_rls(dx_i, dy_i, P, B)

# Prédictions sur le jeu de test
y_pred = predict(X_test, R, B)

# Évaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))





