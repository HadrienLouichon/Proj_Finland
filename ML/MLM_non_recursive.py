import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, euclidean_distances
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import mode
from matplotlib.patches import Patch
from itertools import product

# --- Load Salinas-A data ---
def load_salinas_A():
    data = scipy.io.loadmat('/home/palomies/Documents/Proj_Finland/data/SalinasA.mat')['salinasA_corrected']
    #print(data)
    #print("Shape of Salinas-A data:", data.shape)
    labels = scipy.io.loadmat('/home/palomies/Documents/Proj_Finland/data/SalinasA_gt.mat')['salinasA_gt']
    #print(labels)
    #print("Shape of Salinas-A labels:", labels.shape)
    X = data.reshape(-1, data.shape[-1])
    y = labels.reshape(-1)

    mask = y > 0  # remove background (label 0)
    return X[mask], y[mask], labels, data

# --- Reference points selection ---
def select_reference_points(X_train, y_train, n_components):
    R = [] # Selected reference points list
    T = [] # Corresponding labels list
    for label in np.unique(y_train): # Select reference points for each label
        X_class = X_train[y_train == label]
        #print(f"Processing class {label} with {X_class.shape[0]} samples.")
        #print(X_class)
        pca = PCA(n_components=n_components) # Compute number of components with PCA from the training data.
        X_pca = pca.fit_transform(X_class)
        #print(f"PCA components shape for class {label}: {X_pca.shape}")
        
        for i in range(n_components): # Select 3*n_components training points for each label.
            sorted_indices = np.argsort(X_pca[:, i]) # Argument sort component.
            mini = X_class[sorted_indices[0]]
            median = X_class[sorted_indices[len(sorted_indices) // 2]]
            maxi = X_class[sorted_indices[-1]]
            R.extend([mini, median, maxi])
            T.extend([label, label, label])
        #print(len(R), len(T))
    return np.array(R), np.array(T)

# --- MLM Training phase ---
def train_mlm(X, y, R, T):
    y = y.astype(float) # Convert labels to float for distance calculations
    T = T.astype(float)
    
    Distance_out = euclidean_distances(y[:, np.newaxis], T[:, np.newaxis]) # Output distances
    #print("Output distances shape:", Distance_out.shape)
    Distance_in = euclidean_distances(X, R) # Input distances
    #print("Input distances shape:", Distance_in.shape)
    
    # Approximation of the coefficients using OLS.
    B = np.linalg.pinv(Distance_in).dot(Distance_out)
    #print("B shape:", B.shape)
    #print(B)
    return B

# --- MLM prediction phase ---
def predict_mlm(X_new, T, R, B, n_neighbours):
    D_new = euclidean_distances(X_new, R) # New distances between new data and R.
    
    # Solve the equation : 
    delta = D_new.dot(B)
    predictions = []
    for d in delta:
        sorted_indices = np.argsort(d)
        if n_neighbours != 1:
            closest_labels = T[sorted_indices[:n_neighbours]]
            predictions.append(mode(closest_labels, keepdims=True).mode[0]) # Select the mode of the neighbours
        else:
            predictions.append(T[sorted_indices[0]]) # Select nearest neighbour
    return np.array(predictions)

# --- Visualisation of the PCA ---
def plot_pca_projection(X, y, title='PCA Projection'):
    pca = PCA(n_components=25)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', s=1)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# --- Cross-validation ---
def run_cross_validation(X, y, X_aug, Y_aug, n_neighbours, n_components, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores = []

    for train_index, test_index in skf.split(X, y):
        X_test, y_test = X[test_index], y[test_index]
        R, T = select_reference_points(X_aug, Y_aug, n_components)
        B = train_mlm(X_aug, Y_aug, R, T)
        y_pred = predict_mlm(X_test, T, R, B, n_neighbours)
        
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        print(f"Fold accuracy: {acc:.6f}")
    print(f"\nMean Accuracy: {np.mean(acc_scores):.6f}")
    return acc_scores

# --- Confusion Matrix ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# --- Benchmark training ---
def benchmark_training_prediction(X_train, y_train, X_test, y_test, X_aug, Y_aug, n_neighbours, n_components):
    start_train = time.time()
    R, T = select_reference_points(X_aug, Y_aug, n_components)
    B = train_mlm(X_aug, Y_aug, R, T)
    end_train = time.time()

    start_pred = time.time()
    y_pred = predict_mlm(X_test, T, R, B, n_neighbours)
    end_pred = time.time()

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.6f}")
    print(f"Training Time: {end_train - start_train:.6f}s")
    print(f"Prediction Time: {end_pred - start_pred:.6f}s")
    return acc, y_pred

# --- Comparison maps with color to differiate the labels ---
def plot_comparison_maps(true_labels, pred_labels, shape, class_names, class_colors, class_values):
    true_reshaped = true_labels.reshape(shape)
    pred_reshaped = pred_labels.reshape(shape)
    errors = (true_reshaped != pred_reshaped) & (true_reshaped != 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if class_colors is not None:
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

    if class_names and class_colors:
        patches = [Patch(color=color, label=label) for color, label in zip(class_colors, class_names)]
        fig.legend(handles=patches, loc='lower center', ncol=6)

    plt.tight_layout()
    plt.show()

def best_parameters(X, y, X_aug, Y_aug, input_metric, iteration, k_folds=3):
    
    n_components_list = [i for i in range(1,iteration)]
    n_neighbours_list = [i for i in range(1,iteration)]
    print(n_components_list, n_neighbours_list)

    results = []
    for n_comp, n_neigh in product(n_components_list, n_neighbours_list):
        print(f"Testing n_components = {n_comp}, n_neighbours = {n_neigh}")
        scores = run_cross_validation(X, y, X_aug, Y_aug, n_neigh, n_comp, input_metric, k=k_folds)
        mean_acc = np.mean(scores)
        results.append((n_comp, n_neigh, mean_acc))

    # Convert results to numpy array for plotting
    results = np.array(results)
    best_idx = np.argmax(results[:, 2])
    best_n_comp = int(results[best_idx, 0])
    best_n_neigh = int(results[best_idx, 1])
    best_acc = results[best_idx, 2]

    print(f"\nBest parameters: n_components = {best_n_comp}, n_neighbours = {best_n_neigh} → Accuracy = {best_acc:.4f}")

    # Plot accuracy heatmap
    acc_matrix = np.zeros((len(n_components_list), len(n_neighbours_list)))
    for i, n_comp in enumerate(n_components_list):
        for j, n_neigh in enumerate(n_neighbours_list):
            acc = next((acc for c, n, acc in results if c == n_comp and n == n_neigh), 0)
            acc_matrix[i, j] = acc

    plt.figure(figsize=(10, 6))
    sns.heatmap(acc_matrix, annot=True, xticklabels=n_neighbours_list, yticklabels=n_components_list, cmap='viridis')
    plt.xlabel("n_neighbours")
    plt.ylabel("n_components")
    plt.title("Accuracy Heatmap (Cross-Validation)")
    plt.show()

    return best_n_neigh, best_n_comp

# --- Data augmentation pour image hyperspectrale ---
def augment_image(image, labels, angle=90, saturation_factor=1.0, brightness_offset=0.0):
    augmented_image = image.copy().astype(np.float32)
    augmented_labels = labels.copy()

    # Rotation (par tranche spectrale)
    rotated = np.stack([rotate(augmented_image[:, :, i], angle, reshape=False, mode='nearest')
                        for i in range(augmented_image.shape[-1])], axis=-1)

    # Saturation : multiplicateur global par bande
    rotated *= saturation_factor

    # Décalage de luminosité : ajout d'un offset par pixel
    rotated += brightness_offset

    # Rotation des labels pour rester cohérent avec l'image
    rotated_labels = rotate(augmented_labels, angle, reshape=False, order=0, mode='nearest')

    # Clip pour rester dans les valeurs valides
    rotated = np.clip(rotated, 0, 65535)  # Plage typique uint16

    return rotated.astype(np.uint16), rotated_labels.astype(np.int32)

# --- Apply many augmentations ---
def generate_augmented_dataset(original_image, original_labels):
    angles = [0, 90, 180, 270]
    saturations = [0.9, 1.0, 1.1]
    brightness = [-100, 0, 100]

    X_list = []
    y_list = []

    for angle in angles:
        for sat in saturations:
            for b in brightness:
                aug_img, aug_lbl = augment_image(original_image, original_labels, angle=angle, saturation_factor=sat, brightness_offset=b)
                X = aug_img.reshape(-1, aug_img.shape[-1])
                y = aug_lbl.reshape(-1)
                mask = y > 0
                X_list.append(X[mask])
                y_list.append(y[mask])
                print(f"Augmentation: angle={angle}, saturation={sat}, brightness={b}, samples={np.sum(mask)}")

    X_augmented = np.vstack(X_list)
    y_augmented = np.hstack(y_list)
    mask = y_augmented > 0  # Remove background
    return X_augmented[mask], y_augmented[mask]

# --- Main ---
if __name__ == "__main__":
    
    input_metric = 'euclidean'

    # Load Salinas-A data & PCA projection
    X, y, full_labels, data = load_salinas_A()
    print("Shape of Salinas-A loaded:", X.shape, y.shape) # 7138 values of 204 spectral components
    plot_pca_projection(X, y, title="Salinas-A PCA Projection")

    X_aug, Y_aug = generate_augmented_dataset(data, full_labels)
    print("Shape of augmented data:", X_aug.shape, Y_aug.shape) # Augmented data shape

    # Look for the best parameters
    #n_neighbours, n_components = best_parameters(X, y, X_aug, Y_aug, input_metric, 40, k_folds=3) #: best = 27 (neighbours), 26 (components)
    n_neighbours = 27
    n_components = 26

    # Cross validation, benchmark & confusion matrix
    print("\n--- Cross-validation ---")
    run_cross_validation(X, y, X_aug, Y_aug, n_neighbours, n_components, k=5)

    print("\n--- Benchmark ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    acc, y_pred = benchmark_training_prediction(X_train, y_train, X_test, y_test, X_aug, Y_aug, n_neighbours, n_components)

    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(y_test, y_pred)

    # Comparison maps
    mask = full_labels.reshape(-1) > 0
    predicted_full = np.zeros(full_labels.shape, dtype=int)
    R, T = select_reference_points(X, y, n_components)
    B = train_mlm(X_aug, Y_aug, R, T)
    predicted_all = predict_mlm(X, T, R, B, n_neighbours)
    predicted_full[mask.reshape(full_labels.shape)] = predicted_all.reshape(-1)

    class_values = np.array([1, 10, 11, 12, 13, 14])
    class_names = ["Brocoli", "Corn", "Lettuce4wk", "Lettuce5wk", "Lettuce6wk", "Lettuce7wk"]
    class_colors = ['#1F77B4', '#F7B6D2', '#C7C7C7', '#BCBD22', '#17BECF', '#9EDAE5']

    plot_comparison_maps(full_labels, predicted_full, full_labels.shape, class_names=class_names, class_colors=class_colors, class_values=class_values)
