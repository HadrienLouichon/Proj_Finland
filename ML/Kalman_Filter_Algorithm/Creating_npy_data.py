import numpy as np
import scipy.io

def reshape_data(data, labels):
    data = data.reshape(-1, data.shape[-1])
    labels = labels.reshape(-1)
    return data, labels

data = scipy.io.loadmat('Proj_Finland/ML/Kalman_Filter_Algorithm/SalinasA_corrected.mat')['salinasA_corrected']
labels = scipy.io.loadmat('Proj_Finland/ML/Kalman_Filter_Algorithm/SalinasA_gt.mat')['salinasA_gt']

training_data, training_labels = reshape_data(data, labels)

#mask_background = labels > 0
#data[~mask_background], labels[~mask_background] = 0, 0

classes = np.unique(labels)
X = []
Y = []

for label in classes:
    class_indices = np.where(labels == label)[0]
    if len(class_indices) < 5:
        raise ValueError(f"Not enough points for the class {label}.")
    chosen_indices = np.random.choice(class_indices, size=5, replace=False)
    print("Points for class :", label, "are : ", chosen_indices)
    X.extend(data[chosen_indices])
    Y.extend([label]*5)
print("Number of points selected:", len(X))

#print(training_data.shape)
#print(X)

np.save('Proj_Finland/ML/Kalman_Filter_Algorithm/salinas_x_test.npy', data)
np.save('Proj_Finland/ML/Kalman_Filter_Algorithm/salinas_x_train.npy', np.array(X))
np.save('Proj_Finland/ML/Kalman_Filter_Algorithm/salinas_y_test.npy', labels)
np.save('Proj_Finland/ML/Kalman_Filter_Algorithm/salinas_y_train.npy', np.array(Y))
