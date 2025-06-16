import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# --- Load Salinas-A data ---
def load_salinas_A():
    data = scipy.io.loadmat('ML/SalinasA_corrected.mat')['salinasA_corrected']
    #print(data)
    print("Shape of Salinas-A data:", data.shape)
    labels = scipy.io.loadmat('ML/SalinasA_gt.mat')['salinasA_gt']
    #print(labels)
    print("Shape of Salinas-A labels:", labels.shape)
    
    # Remove the background : class (0)
    mask_background = labels > 0
    data[~mask_background], labels[~mask_background] = 0, 0
    #print("Data shape after removing background:", data.shape)
    # Separate training and testing data, each 3rd row for training, the others for testing
    training_data = data[::3, :, :]
    training_labels = labels[::3, :]
    mask = np.ones(data.shape[0], dtype=bool)
    mask[::3] = False
    testing_data = data[mask, :, :]
    testing_labels = labels[mask, :]
    
    print("Training data shape:", training_data.shape)
    print("Testing data shape:", testing_data.shape)
    #print("Training labels shape:", training_labels.shape)
    #print("Testing labels shape:", testing_labels.shape)
    return training_data, training_labels, testing_data, testing_labels

# Selection of the reference points (3 per class, randomly selected)
def select_reference_points(training_data, training_labels):
    #classes = np.unique(training_labels[training_labels > 0])
    classes = np.unique(training_labels)
    #print(classes)
    #print(type(classes))
    R = [] # Reference points, 3 per classes selected randomly
    T = [] # Labels of reference points

    for label in classes:
        class_indices = np.where(training_labels == label)[0]
        if len(class_indices) < 3:
            raise ValueError(f"Not enough points for the class {label}.")
        chosen_indices = np.random.choice(class_indices, size=3, replace=False)
        print("Points for class :", label, "are : ", chosen_indices)
        R.extend(training_data[chosen_indices])
        T.extend([label, label, label])
    
    print("Number of reference points selected:", len(R))
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

    #Distance_out = euclidean_distances(labels[:, np.newaxis], T[:, np.newaxis])
    #Distance_in = euclidean_distances(data, R)
    Distance_out = np.array([[np.linalg.norm(l - ref_label) for ref_label in T] for l in labels])
    Distance_in = np.array([[np.linalg.norm(d - ref_data) for ref_data in R] for d in data])
    
    #print("Distance_out shape:", Distance_out.shape)
    #print("Distance_in shape:", Distance_in.shape)
    return Distance_out, Distance_in

# Initialization 
def rls_initialization(Distance_out, Distance_in):
    P0 = np.linalg.pinv(Distance_in.T @ Distance_in)
    B0 = P0 @ Distance_in.T @ Distance_out

    #print("P0 shape:", P0.shape)
    #print("B0 shape:", B0.shape)
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

# Predict the label for new data points
def predict_label(X_new, R, T, B):
    #Dx_new = euclidean_distances(X_new, R)
    Dx_new = np.array([[np.linalg.norm(x - ref_data) for ref_data in R] for x in X_new])
    predictions = Dx_new @ B
    predicted_labels = np.argmin(predictions, axis=1)
    return T[predicted_labels][0]

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

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

# Main function to run the RLS model
if __name__ == "__main__":

    # Define class names, colors, and values for plots
    class_values = np.array([0, 1, 10, 11, 12, 13, 14])
    class_names = ["Background","Brocoli", "Corn", "Lettuce4wk", "Lettuce5wk", "Lettuce6wk", "Lettuce7wk"]
    class_colors = ['#1F77B4','#AEC7E8', '#7F7F7F', '#C7C7C7', '#BCBD22', '#17BECF', '#9EDAE5']

    # Load the Salinas-A dataset
    training_data, training_labels, testing_data, testing_labels = load_salinas_A()
    training_data, training_labels = reshape_data(training_data, training_labels)
    
    # Select reference points
    R, T = select_reference_points(training_data, training_labels)
    R, T = reshape_data(R, T)
    
    # Build the model from the training data
    Distance_out, Distance_in = build_distance_matrices(training_data, R, training_labels, T)
    P, B = rls_initialization(Distance_out, Distance_in)

    num_rows, num_cols = testing_data.shape[0], testing_data.shape[1]
    predictions_array = np.zeros((num_rows, num_cols), dtype=np.uint8)

    for i in range(testing_data.shape[0]):
        print("Processing row:", i+1)
        data = testing_data[i,:]
        label = testing_labels[i,:]
        for j in range(testing_data.shape[1]):
            pixel, pix_label = reshape_data(data[j], label[j])
            # Make predictions for the current data point
            predicted_label = predict_label(pixel, R, T, B)
            predictions_array[i,j] = predicted_label
            #print("Label to predict:", pix_label)
            #print("Predicted label for pixel:", predicted_label)
            # Update the RLS model with new data
        valid_pixels = label != 0
        #print(valid_pixels)
        data_valid = data[valid_pixels]
        label_valid = label[valid_pixels]
        New_distance_out, New_distance_in = build_distance_matrices(data_valid, R, label_valid, T)
        #New_distance_out, New_distance_in = build_distance_matrices(data, R, label, T)
        P, B = rls_update(New_distance_out, New_distance_in, P, B)

    print(accuracy_score(testing_labels.reshape(-1), predictions_array.reshape(-1)))
    plot_confusion_matrix(testing_labels.reshape(-1), predictions_array.reshape(-1))
    plot_comparison_maps(testing_labels, predictions_array, testing_labels.shape, class_names=class_names, class_colors=class_colors, class_values=class_values)

