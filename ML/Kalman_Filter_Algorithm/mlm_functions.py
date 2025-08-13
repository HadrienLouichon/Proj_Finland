import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from numpy.linalg import pinv
from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.metrics import accuracy_score, f1_score



def select_random_subset(x_train, y_train, n_samples, random_seed=None):
    """
    Select a random subset of x_train and y_train.

    Parameters:
    -----------
    x_train : np.ndarray
        Training data of shape (H, W, bands) or (N_samples, bands)
    y_train : np.ndarray
        Corresponding labels of shape (H, W) or (N_samples,)
    n_samples : int
        Number of samples to select
    random_seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    x_subset : np.ndarray
        Selected samples, shape (n_samples, bands)
    y_subset : np.ndarray
        Selected labels, shape (n_samples, 1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # If x_train is image-shaped, flatten
    if x_train.ndim == 3:
        H, W, bands = x_train.shape
        x_flat = x_train.reshape(-1, bands)
        y_flat = y_train.flatten()
    else:
        x_flat = x_train
        y_flat = y_train

    # Remove background (y == 0) if desired
    valid_idx = np.where(y_flat > 0)[0]

    if len(valid_idx) < n_samples:
        raise ValueError(f"Requested {n_samples} samples, but only {len(valid_idx)} valid points available.")

    selected_idx = np.random.choice(valid_idx, size=n_samples, replace=False)

    x_subset = x_flat[selected_idx, :]
    y_subset = y_flat[selected_idx].reshape(-1, 1)

    return x_subset, y_subset


def select_R(x_train, y_train, n_per_class=5, random_seed=None):
    """
    Select representative samples (R) and labels (y_R).

    Parameters:
    -----------
    x_train : np.ndarray
        Training data, shape (n_samples, n_features) or (H, W, n_features)
    y_train : np.ndarray
        Ground truth labels, shape (n_samples,) or (H, W)
    n_per_class : int
        Number of samples to select per class
    random_seed : int or None
        Seed for reproducibility

    Returns:
    --------
    R : np.ndarray
        Selected samples, shape (n_classes * n_per_class, n_features)
    y_R : np.ndarray
        Selected labels, shape (n_classes * n_per_class, 1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # If input is image, flatten it
    if x_train.ndim == 3:
        H, W, bands = x_train.shape
        x_flat = x_train.reshape(-1, bands)
        y_flat = y_train.flatten()
    else:
        x_flat = x_train
        y_flat = y_train.flatten()

    classes = np.unique(y_flat)
    classes = classes[classes > 0]  # Ignore zero class if it's background

    R_list = []
    y_R_list = []

    for cls in classes:
        idx = np.where(y_flat == cls)[0]
        if len(idx) < n_per_class:
            raise ValueError(f"Not enough samples for class {cls}. Found {len(idx)}.")

        chosen_idx = np.random.choice(idx, size=n_per_class, replace=False)
        R_list.append(x_flat[chosen_idx, :])
        y_R_list.append(y_flat[chosen_idx])

    R = np.vstack(R_list)
    y_R = np.hstack(y_R_list).reshape(-1, 1)

    return R, y_R

def predict_classes(continuous_results, y_R, neighbours=0):
    order = np.argsort(continuous_results, axis=1)
    if neighbours > 0:
        neigh_labels = y_R[order[:, :neighbours]]
        modes = mode(neigh_labels, axis=1)[0]
        return modes.reshape(-1, 1)
    else:
        return y_R[order[:, 0]].reshape(-1, 1)
    
def init_model(X,Y,R, y_R):
    distances = euclidean_distances(X, R)
    P = distances.T @ distances
    bhat = pinv(P) @ distances.T @ euclidean_distances(Y, y_R)
    return bhat, P

def predict_model(bhat, x, R, y_R,n):
    d = euclidean_distances(x, R)
    yhat = d @ bhat
    anom_score = np.var(yhat,axis=1)
    y_pred = predict_classes(yhat, y_R,neighbours=n)
    return y_pred, anom_score

def recursive_ls(P_o, x_di, y_di, bhat_o):
    P_di = P_o @ x_di
    A = np.eye(x_di.shape[1]) + x_di.T @ P_di
    P_i = P_o - P_di @ pinv(A) @ (x_di.T @ P_o)
    K_i = P_i @ x_di
    residual = y_di.T - x_di.T @ bhat_o
    bhat_new = bhat_o + K_i @ residual
    return bhat_new, P_i



def kalman_fuse_init(b_init, P_init, Q):
    return b_init.copy(), P_init.copy(), Q

def kalman_fuse_predict(b_fuse, P_fuse, Q):
    # Prediction step
    b_pred = b_fuse
    P_pred = P_fuse + Q 
    return b_pred, P_pred

def kalman_fuse_update(b_pred, P_pred, z, R):
    # Update step with measurement z, covariance R
    K = P_pred @ pinv(P_pred + R)
    
    b_new = b_pred + 0.5 * K @ (z - b_pred)
    P_new = (np.eye(P_pred.shape[0]) - K) @ P_pred
    return b_new, P_new, K

def kalman_fuse_update_adaptive(b_pred, P_pred, z, R, gain_norm_history, regularization=1e-4):
    """
    Adaptive Kalman update step.

    Parameters:
    -----------
    b_pred : np.ndarray
        Predicted state
    P_pred : np.ndarray
        Predicted covariance
    z : np.ndarray
        Measurement (e.g., UAV model)
    R : np.ndarray
        Measurement covariance
    gain_norm_history : list
        Store norm(K) values
    regularization : float
        Added to P to avoid singularity

    Returns:
    --------
    b_new : np.ndarray
        Updated state
    P_new : np.ndarray
        Updated covariance
    K : np.ndarray
        Kalman gain
    log_det_P : float
        Log-determinant of updated P
    """
    # Regularize predicted covariance
    P_pred += regularization * np.eye(P_pred.shape[0])

    # Kalman gain
    K = P_pred @ pinv(P_pred + R)

    # Update step
    b_new = b_pred + K @ (z - b_pred)
    P_new = (np.eye(P_pred.shape[0]) - K) @ P_pred

    # Track gain norm
    gain_norm = np.linalg.norm(K, ord='fro')
    gain_norm_history.append(gain_norm)

    # Track log-determinant
    sign, log_det_P = np.linalg.slogdet(P_new)
    log_det_P = log_det_P if sign > 0 else -np.inf

    return b_new, P_new, K, log_det_P


def compute_metrics(y_true, y_pred, valid_idx):
    acc = accuracy_score(y_true[valid_idx], y_pred[valid_idx])
    f1 = f1_score(y_true[valid_idx], y_pred[valid_idx], average='weighted', zero_division=0)
    return acc, f1

def run_simulation(x_train, y_train, x_test, y_test, R, y_R, uav1_rows, uav2_rows,anom_threshold_b,anom_threshold_u):
    H_train, W_train, bands = x_train.shape
    H_test, W_test, _ = x_test.shape
    n = 5 #
    # anom_threshold_b = 0.7
    # anom_threshold_u = 1.5

    gain_norms = []
    log_det_P_trace = []


    train_res = np.zeros((H_train,W_train))

    x_test_flat = x_test.reshape(-1, bands)
    y_test_flat = y_test.flatten()
    valid_idx = y_test_flat > 0

    #x_subset, y_subset = select_random_subset(x_train, y_train, n_samples=  7000, random_seed=42)

    #bhat1, P1 = init_model(x_subset, y_subset ,R, y_R)
    #bhat2, P2 = init_model(x_subset, y_subset ,R, y_R)

    bhat1, P1 = bhat1a, P1a = init_model(R, y_R,R, y_R)
    bhat2, P2 = bhat2a, P2a  = init_model(R, y_R,R, y_R)

    small_identity_matrix = 1e-6 * np.eye(bhat1.shape[0])
    b_fuse, P_fuse, Q = kalman_fuse_init(bhat1, P1, Q=small_identity_matrix)

    # Predict
    
    
    metrics = {
        'uav1_before': {'acc': [], 'f1': []},
        'uav2_before': {'acc': [], 'f1': []},
        'uav1_after': {'acc': [], 'f1': []},
        'uav2_after': {'acc': [], 'f1': []},
        'base': {'acc': [], 'f1': []}
    }

    test_results = {'base':[]}

    models = {
        'uav1_before':[],
        'uav2_before': [],
        'uav1_after': [],
        'uav2_after': [],
        'base': []
    }

    for row in range(H_train):
        x_row1 = uav1_rows[row,:,:]
        x_row2 = uav2_rows[row,:,:]
        
        # UAV 1

       
        #pred1 = predict_model(bhat1, x_row1, R, y_R,n)
      

        pred1, as1 = predict_model(bhat1, x_row1, R, y_R,n)
        anomscore1 = (as1>anom_threshold_b)& (as1<anom_threshold_u)
       
        if np.any(anomscore1):
            
            mask1 = anomscore1
                
            pred1n = pred1[mask1]
            x_row1n = x_row1[mask1,:]
            y_target = euclidean_distances(pred1n, y_R)
            d_row1 = euclidean_distances(x_row1n, R)
            bhat1, P1 = recursive_ls(P1, d_row1.T, y_target.T, bhat1)

        models['uav1_before'].append(bhat1)

        # UAV 2
        
        #pred2 = predict_model(bhat2, x_row2, R, y_R,n)

        pred2, as2 = predict_model(bhat2, x_row2, R, y_R,n)
        anomscore2 = (as2>anom_threshold_b)& (as2<anom_threshold_u)
        if np.any(anomscore2):

            mask2 =  anomscore2
            pred2n = pred2[mask2]
            x_row2n = x_row2[mask2,:]
            y_target = euclidean_distances(pred2n, y_R)
            d_row2 = euclidean_distances(x_row2n, R)
            bhat2, P2 = recursive_ls(P2, d_row2.T, y_target.T, bhat2)
            
        models['uav2_before'].append(bhat2)

        train_res[row,0:pred1.shape[0]] = pred1.reshape(-1)
        train_res[row,pred1.shape[0]::] = pred2.reshape(-1)

        # Predict before base without update


        pred1a,astmp = predict_model(bhat1a, x_row1, R, y_R,n)
        y_targeta = euclidean_distances(pred1a, y_R)
        d_row1a = euclidean_distances(x_row1, R)
        bhat1a, P1a = recursive_ls(P1a, d_row1a.T, y_targeta.T, bhat1a)

        pred2a,astmp = predict_model(bhat2a, x_row2, R, y_R,n)
        y_targeta = euclidean_distances(pred2a, y_R)
        d_row2a = euclidean_distances(x_row2, R)
        bhat2a, P2a = recursive_ls(P2a, d_row2a.T, y_targeta.T, bhat2a)

        pred1aa,astmp = predict_model(bhat1a, x_test_flat, R, y_R,n)
        pred2aa,astmp = predict_model(bhat2a, x_test_flat, R, y_R,n)

        acc1, f1_1 = compute_metrics(y_test_flat, pred1aa, valid_idx)
        acc2, f1_2 = compute_metrics(y_test_flat, pred2aa, valid_idx)
        metrics['uav1_before']['acc'].append(acc1)
        metrics['uav1_before']['f1'].append(f1_1)
        metrics['uav2_before']['acc'].append(acc2)
        metrics['uav2_before']['f1'].append(f1_2)




        
        
    
     

        # Update with UAV1 model
        if any(anomscore1):
            b_pred, P_pred = kalman_fuse_predict(b_fuse, P_fuse, Q)
            # b_pred, P_pred, K = kalman_fuse_update(b_pred, P_pred, bhat1, P1)

            b_pred, P_pred, K, log_det_P = kalman_fuse_update_adaptive(b_pred, P_pred, bhat1, P1, gain_norms)
            log_det_P_trace.append(log_det_P)

            gain_norm = np.linalg.norm(K, ord='fro')
            Q = gain_norm * np.eye(P_pred.shape[0])
            gain_norms.append(gain_norm)

            # Store determinant of covariance
            #det_P_trace.append(np.linalg.det(P_pred))
            
            b_fuse, P_fuse = b_pred, P_pred

            # Update UAVs
            bhat1 = np.copy(b_fuse) 
            bhat2 = np.copy(b_fuse)
            P1 = np.copy(P_fuse)
            P2 = np.copy(P_fuse)


        # Update with UAV2 model
        if any(anomscore2):
            b_pred, P_pred = kalman_fuse_predict(b_fuse, P_fuse, Q)
            #b_pred, P_pred, K  = kalman_fuse_update(b_pred, P_pred, bhat2, P2)

            b_pred, P_pred, K, log_det_P = kalman_fuse_update_adaptive(b_pred, P_pred, bhat2, P2, gain_norms)
            log_det_P_trace.append(log_det_P)

            gain_norm = np.linalg.norm(K, ord='fro')
            Q = gain_norm * np.eye(P_pred.shape[0])
            gain_norms.append(gain_norm)

            # Store determinant of covariance
            #det_P_trace.append(np.linalg.det(P_pred))

            # Now b_pred, P_pred is your fused model
            b_fuse, P_fuse = b_pred, P_pred

            # Update UAVs
            bhat1 = np.copy(b_fuse) 
            bhat2 = np.copy(b_fuse)
            P1 = np.copy(P_fuse)
            P2 = np.copy(P_fuse)

        models['base'].append(b_fuse)

        # Predict after merge
        pred1a, astmp = predict_model(bhat1, x_test_flat, R, y_R,n)
        pred2a, astmp = predict_model(bhat2, x_test_flat, R, y_R,n)
        acc1a, f1_1a = compute_metrics(y_test_flat, pred1a, valid_idx)
        acc2a, f1_2a = compute_metrics(y_test_flat, pred2a, valid_idx)
        metrics['uav1_after']['acc'].append(acc1a)
        metrics['uav1_after']['f1'].append(f1_1a)
        metrics['uav2_after']['acc'].append(acc2a)
        metrics['uav2_after']['f1'].append(f1_2a)
        
  

        # Base station metrics
        pred_base,astmp = predict_model(b_fuse, x_test_flat, R, y_R,n)
        test_results['base'].append(pred_base.reshape((H_test,W_test)))
        acc_base, f1_base = compute_metrics(y_test_flat, pred_base, valid_idx)
        metrics['base']['acc'].append(acc_base)
        metrics['base']['f1'].append(f1_base)

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['uav1_before']['acc'], label='UAV1 without base')
    plt.plot(metrics['uav2_before']['acc'], label='UAV2 without base')
    plt.plot(metrics['uav1_after']['acc'], label='UAV1 with base', linestyle='--')
    plt.plot(metrics['uav2_after']['acc'], label='UAV2 with base', linestyle='--')
    plt.plot(metrics['base']['acc'], label='Base station', linestyle='-.')
    plt.xlabel('Training row')
    plt.ylabel('Accuracy on x_test')
    plt.legend()
    plt.title('Accuracy over updates')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

    # Plot F1-score
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['uav1_before']['f1'], label='UAV1 without base')
    plt.plot(metrics['uav2_before']['f1'], label='UAV2 without base')
    plt.plot(metrics['uav1_after']['f1'], label='UAV1 with base', linestyle='--')
    plt.plot(metrics['uav2_after']['f1'], label='UAV2 with base', linestyle='--')
    plt.plot(metrics['base']['f1'], label='Base station', linestyle='-.')
    plt.xlabel('Training row')
    plt.ylabel('F1-score on x_test')
    plt.legend()
    plt.title('F1-score over updates')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gain norm of Kalman filter
    plt.plot(gain_norms)
    plt.title("Kalman gain norm over updates")
    plt.xlabel("Update step")
    plt.ylabel("Gain norm")

    # Determinant of Kalman covariance
    plt.figure(figsize=(10, 4))
    #plt.plot(det_P_trace)
    plt.title("Determinant of Kalman covariance matrix P over time")
    plt.xlabel("Update step")
    plt.ylabel("det(P)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return metrics, test_results, train_res, models

if __name__ == "__main__":
    x_test = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_x_test.npy')
    x_train = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_x_train.npy')
    y_test = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_y_test.npy')
    y_train = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_y_train.npy')


    ##from scipy.io import loadmat
    ##X = loadmat('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\Pavia.mat')
    ##Y = loadmat('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\Pavia_gt.mat')
    ##X = np.double(np.array(X['paviaU']))
    ##Y = np.double(np.array(Y['paviaU_gt']))

    ##x_test = X[::2,:,:]
    ##x_train = X[1::2,:,:]
    ##y_test = Y[::2,:]
    ##y_train = Y[1::2,:]

    labels = np.unique(y_train)


    idx=0
    for l in labels:
        y_train[y_train==l]=idx
        y_test[y_test==l]=idx
        idx +=1

    R, y_R = select_R(x_train, y_train, n_per_class=5, random_seed=10)



    uav1_rows = x_train[:,0:int(x_train.shape[1]/2),:]
    uav2_rows = x_train[:,int(x_train.shape[1]/2)::,]
    metrics, test_results, train_res, models = run_simulation(x_train, y_train, x_test, y_test, R, y_R, uav1_rows, uav2_rows, 0.94736842, 1.94736842)
    plt.figure(figsize=(20,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(test_results['base'][i])
        plt.title('After row '+str(i))
    plt.show()

    plt.figure(figsize=(20,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(test_results['base'][i]-test_results['base'][i+1])
        plt.title('Difference between '+str(i) + 'and '+str(i+1))
    plt.show()

    plt.figure(figsize=(20,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow((test_results['base'][i]-y_test)*(y_test>0))
        plt.title('After row '+str(i))
    plt.show()

    plt.figure(figsize=(20,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(test_results['base'][i]*(y_test>0))
        plt.title('After row '+str(i))
    plt.show()


    plt.imshow(train_res)
    plt.show()

    plt.imshow((train_res*(y_train>0)-y_train)>0)
    plt.show()

    mallit = np.zeros((30,30,42))
    for i in range(42):
        mallit[:,:,i] = models['base'][i]

    for i in range(30):
        for j in range(30):
            plt.plot(mallit[i,j,0::-1]-mallit[i,j,1::],alpha=0.1)
            

    plt.show()