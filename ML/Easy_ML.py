import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- 1. Chargement des données Salinas-A ---
def load_salinas_A():
    data = scipy.io.loadmat('ML/SalinasA_corrected.mat')['salinasA_corrected']
    labels = scipy.io.loadmat('ML/SalinasA_gt.mat')['salinasA_gt']
    X = data.reshape(-1, data.shape[-1])
    y = labels.reshape(-1)
    mask = y > 0  # remove background (label 0)
    return X[mask], y[mask], labels  # Return full label map for visualization

# --- 2. Sélection des points de référence ---
def select_reference_points(X_train, y_train, n_components=10):
    R = []
    T = []
    for label in np.unique(y_train):
        X_class = X_train[y_train == label]
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_class)
        for i in range(n_components):
            scores = X_pca[:, i]
            sorted_indices = np.argsort(scores)
            min_ = X_class[sorted_indices[0]]
            median = X_class[sorted_indices[len(sorted_indices) // 2]]
            max_ = X_class[sorted_indices[-1]]
            R.extend([min_, median, max_])
            T.extend([label, label, label])
    return np.array(R), np.array(T)

# --- 3. Entraînement MLM ---
def train_mlm(X, y, R, T, input_metric='euclidean'):
    y = y.astype(float)
    T = T.astype(float)
    D_out = pairwise_distances(y[:, np.newaxis], T[:, np.newaxis], metric='euclidean')
    D_in = pairwise_distances(X, R, metric=input_metric)
    Bb = np.linalg.pinv(D_in).dot(D_out)
    return Bb

# --- 4. Prédiction MLM ---
def predict_mlm(X_new, T, R, Bb, input_metric='euclidean', n_neighbors=25):
    D_new = pairwise_distances(X_new, R, metric=input_metric)
    delta = D_new.dot(Bb)
    predictions = []
    for d in delta:
        sorted_indices = np.argsort(d)
        if n_neighbors == 1:
            predictions.append(T[sorted_indices[0]])
        else:
            closest_labels = T[sorted_indices[:n_neighbors]]
            predictions.append(mode(closest_labels, keepdims=True).mode[0])
    return np.array(predictions)

# --- 5. Visualisation PCA ---
def plot_pca_projection(X, y, title='PCA Projection'):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', s=1)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# --- 6. Cross-validation ---
def run_cross_validation(X, y, k=5, n_neighbors=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        R, T = select_reference_points(X_train, y_train)
        Bb = train_mlm(X_train, y_train, R, T)
        y_pred = predict_mlm(X_test, T, R, Bb, n_neighbors=n_neighbors)
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        print(f"Fold accuracy: {acc:.4f}")
    print(f"\nMean Accuracy: {np.mean(acc_scores):.4f}")
    return acc_scores

# --- 7. Matrice de confusion ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# --- 8. Benchmark temps ---
def benchmark_training_prediction(X_train, y_train, X_test, y_test, n_neighbors=3):
    start_train = time.time()
    R, T = select_reference_points(X_train, y_train)
    Bb = train_mlm(X_train, y_train, R, T)
    end_train = time.time()

    start_pred = time.time()
    y_pred = predict_mlm(X_test, T, R, Bb, n_neighbors=n_neighbors)
    end_pred = time.time()

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {end_train - start_train:.4f}s")
    print(f"Prediction Time: {end_pred - start_pred:.4f}s")
    return acc, y_pred

# --- 9. Affichage des cartes avec légende par patch couleur ---
def plot_comparison_maps(true_labels, pred_labels, shape, class_names=None, class_colors=None, class_values=None):
    true_reshaped = true_labels.reshape(shape)
    pred_reshaped = pred_labels.reshape(shape)
    errors = (true_reshaped != pred_reshaped) & (true_reshaped != 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if class_colors is not None:
        cmap = ListedColormap(class_colors)
    else:
        cmap = 'tab20'

    im0 = axes[0].imshow(true_reshaped, cmap=cmap, vmin=min(class_values), vmax=max(class_values))
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    im1 = axes[1].imshow(pred_reshaped, cmap=cmap, vmin=min(class_values), vmax=max(class_values))
    axes[1].set_title("Predicted Labels")
    axes[1].axis('off')

    im2 = axes[2].imshow(errors, cmap='Reds')
    axes[2].set_title("Prediction Errors")
    axes[2].axis('off')

    # Légende personnalisée avec patchs
    if class_names and class_colors:
        patches = [Patch(color=color, label=label) for color, label in zip(class_colors, class_names)]
        fig.legend(handles=patches, loc='lower center', ncol=6)

    plt.tight_layout()
    plt.show()

# --- 10. Exécution principale ---
if __name__ == "__main__":
    X, y, full_labels = load_salinas_A()
    print("Salinas-A loaded:", X.shape, y.shape)

    plot_pca_projection(X, y, title="Salinas-A PCA Projection")

    print("\n--- Cross-validation ---")
    run_cross_validation(X, y, k=3, n_neighbors=3)

    print("\n--- Benchmark ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    acc, y_pred = benchmark_training_prediction(X_train, y_train, X_test, y_test, n_neighbors=3)

    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(y_test, y_pred)

    # --- Image des prédictions et des vrais labels ---
    mask = full_labels.reshape(-1) > 0
    predicted_full = np.zeros(full_labels.shape, dtype=int)
    R, T = select_reference_points(X, y)
    Bb = train_mlm(X, y, R, T)
    predicted_all = predict_mlm(X, T, R, Bb, n_neighbors=3)
    predicted_full[mask.reshape(full_labels.shape)] = predicted_all.reshape(-1)

    class_values = np.array([1, 10, 11, 12, 13, 14])
    class_names = ["Brocoli", "Corn", "Lettuce4wk", "Lettuce5wk", "Lettuce6wk", "Lettuce7wk"]
    class_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    plot_comparison_maps(full_labels, predicted_full, full_labels.shape, class_names=class_names, class_colors=class_colors, class_values=class_values)
