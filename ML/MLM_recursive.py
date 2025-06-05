import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Salinas-A data ---
def load_salinas_A():
    data = scipy.io.loadmat('ML/SalinasA_corrected.mat')['salinasA_corrected']
    #print(data)
    #print("Shape of Salinas-A data:", data.shape)
    labels = scipy.io.loadmat('ML/SalinasA_gt.mat')['salinasA_gt']
    #print(labels)
    #print("Shape of Salinas-A labels:", labels.shape)
    
    # Remove the background : class (0)
    #mask_background = labels > 0
    #data[~mask_background], labels[~mask_background] = 0, 0
    #print("Data shape after removing background:", data.shape)
    # Separate training and testing data, each 3rd row for training, the others for testing
    training_data = data[::3, :, :]
    training_labels = labels[::3, :]
    mask = np.ones(data.shape[0], dtype=bool)
    mask[::3] = False
    testing_data = data[mask, :, :]
    testing_labels = labels[mask, :]
    
    #print("Training data shape:", training_data.shape)
    #print("Testing data shape:", testing_data.shape)
    #print("Training labels shape:", training_labels.shape)
    #print("Testing labels shape:", testing_labels.shape)
    return training_data, training_labels, testing_data, testing_labels

# Selection of the reference points (3 per class, randomly selected)
def select_reference_points(training_data, training_labels):
    classes = np.unique(training_labels[training_labels > 0])
    #print(classes)
    #print(type(classes))
    R = [] # Reference points, 3 per classes selected randomly
    T = [] # Labels of reference points

    for label in classes:
        class_indices = np.where(training_labels == label)[0] # Indices of the current class
        if len(class_indices) < 3:
            raise ValueError(f"Not enough points for the class {label}.")
        chosen_indices = np.random.choice(class_indices, size=3, replace=False)
        R.extend(training_data[chosen_indices])
        T.extend([label, label, label])
    
    #print("Reference points selected:", len(R))
    #print(R)
    return np.array(R), np.array(T)

# Reshape the data for distance calculations :
def reshape_data(data, labels):
    data = data.reshape(-1, data.shape[-1])  # Flatten the last dimension
    labels = labels.reshape(-1)  # Flatten the labels
    
    #print("Reshaped data shape:", data.shape)
    #print("Reshaped labels shape:", labels.shape)
    return data, labels

# Build the distances matrices :
def build_distance_matrices(data, R, labels, T):
    labels = labels.astype(float)  
    T = T.astype(float)

    Distance_out = euclidean_distances(labels[:, np.newaxis], T[:, np.newaxis])
    Distance_in = euclidean_distances(data, R)
    
    #print("Distance_out shape:", Distance_out.shape)
    #print("Distance_in shape:", Distance_in.shape)
    return Distance_out, Distance_in

# Initialization 
def rls_initialization(Distance_out, Distance_in):
    P0 = np.linalg.pinv(Distance_in.T @ Distance_in)
    B0 = P0 @ Distance_in.T @ Distance_out
    
    print("P0 shape:", P0.shape)
    print("B0 shape:", B0.shape)
    return P0, B0

# Update the RLS model
def rls_update(New_distance_out, New_distance_in, P, B):
    P_x = P @ New_distance_in.T
    P_new = P - P_x @ np.linalg.inv(np.eye(New_distance_in.shape[0]) + New_distance_in @ P_x) @ New_distance_in @ P
    K_i = P_new @ New_distance_in.T
    B_new = B + K_i @ (New_distance_out - New_distance_in @ B)
    
    #print("P_new shape:", P_new.shape)
    #print("B_new shape:", B_new.shape)
    return P_new, B_new

def predict_label(X_new, R, T, B):
    Dx_new = euclidean_distances(X_new, R)
    predictions = Dx_new @ B
    predicted_labels = np.argmin(predictions, axis=1)
    return T[predicted_labels]

# Main function to run the RLS model
if __name__ == "__main__":
    # Load the Salinas-A dataset
    training_data, training_labels, testing_data, testing_labels = load_salinas_A()
    # Select reference points
    R, T = select_reference_points(training_data, training_labels)
    # Reshape the data for distance calculations
    training_data, training_labels = reshape_data(training_data, training_labels)
    testing_data, testing_labels = reshape_data(training_data, training_labels)
    R, T = reshape_data(R, T)
    # Build the distance matrices
    Distance_out, Distance_in = build_distance_matrices(training_data, R, training_labels, T)
    # Initialize the RLS model
    P0, B0 = rls_initialization(Distance_out, Distance_in)

    for i in range(training_data.shape[0]):
        data = training_data[i]
        label = training_labels[i]
        data, label = reshape_data(data, label)
        # Update the RLS model with new data
        New_distance_out, New_distance_in = build_distance_matrices(data, R, label, T)
        P_new, B_new = rls_update(New_distance_out, New_distance_in, P0, B0)
    print(P_new)
    print(B_new)
