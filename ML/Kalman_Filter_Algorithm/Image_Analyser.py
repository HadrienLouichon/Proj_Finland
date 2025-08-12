import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances 
from sklearn.metrics import accuracy_score, f1_score
import paho.mqtt.client as mqtt
import base64
import time
import uuid
import json
from io import BytesIO
import threading

MQTT_BROKER = "localhost" ##<-- TO CHANGE
MQTT_PORT = 1883
USE_TLS = False

TOPIC_Data = "pi/data"
TOPIC_ACK = "pc/ack"

CHUNK_SIZE = 10_000_000  # 10 Mo

ack_recu = False
message_id_en_cours = ""
feedback_vars = {}
lock = threading.Lock()

#------MQTT-------#

def on_message(client, userdata, msg):
    global ack_recu, feedback_vars
    ack = json.loads(msg.payload.decode('utf-8'))
    if ack.get("id") == message_id_en_cours:
        with lock:
            ack_recu = True
            print(f"[RECEIVER] '{ack.get('status')}' → variables bien reçues.")
            if "extra_data" in ack:
                decoded_fb = {}
                for k, v in ack["extra_data"].items():
                    # On détecte si c'est un tableau NumPy
                    try:
                        decoded_fb[k] = decode_variable(v)
                    except Exception:
                        decoded_fb[k] = decode_variable(v)
                feedback_vars = decoded_fb  # Remplace entièrement

def encode_variable(var):
    buf = BytesIO()
    np.save(buf, var)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def decode_variable(b64_str):
    buf = BytesIO(base64.b64decode(b64_str))
    return np.load(buf, allow_pickle=True)

def send_variables(variables, client):
    global message_id_en_cours, ack_recu
    with lock:
        ack_recu = False
    message_id = str(uuid.uuid4())
    message_id_en_cours = message_id

    vars_payload = {f"var{i+1}": encode_variable(v) for i, v in enumerate(variables)}
    data_bytes = json.dumps(vars_payload).encode("utf-8")
    total_chunks = (len(data_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"\n📤 Envoi de {len(variables)} variables ({len(data_bytes)} octets) en {total_chunks} chunk(s)...")

    for i in range(total_chunks):
        chunk = data_bytes[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
        payload = {
            "message_id": message_id,
            "chunk_id": i,
            "total_chunks": total_chunks,
            "data": base64.b64encode(chunk).decode("utf-8")
        }
        client.publish(TOPIC_Data, json.dumps(payload))
        time.sleep(0.05)

    # Attendre l'ACK complet
    print("[TRANSMITTER] En attente de l'ACK complet...")
    timeout = time.time() + 5  # 5 secondes max
    while True:
        with lock:
            if ack_recu:
                print("[TRANSMITTER] ✅ ACK reçu.")
                return True
        if time.time() > timeout:
            print("[TRANSMITTER] ❌ Timeout en attente de l'ACK.")
            return False
        time.sleep(0.1)

def attendre_feedback(timeout=5):
    """Attend que ack_recu soit True et que feedback_vars contienne des données."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with lock:
            if ack_recu and feedback_vars:
                return dict(feedback_vars)  # retourne une copie
        time.sleep(0.05)
    return None

#------IMAGE-------#

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

def compute_metrics(y_true, y_pred, valid_idx):
    acc = accuracy_score(y_true[valid_idx], y_pred[valid_idx])
    f1 = f1_score(y_true[valid_idx], y_pred[valid_idx], average='weighted', zero_division=0)
    return acc, f1

def run_simulation(x_train, y_train, x_test, y_test, R, y_R, uav1_rows, uav2_rows,anom_threshold_b,anom_threshold_u, client):
    
    global feedback_vars
    
    H_train, W_train, bands = x_train.shape
    H_test, W_test, _ = x_test.shape
    n = 5 #
    # anom_threshold_b = 0.7
    # anom_threshold_u = 1.5

    gain_norms = []
    log_det_P_trace = []
    det_P_trace = []


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




        #'HERE'
        #'Send data to Model_Upgrade : anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms'
        #'Receive : log_det_P, gain_norm, bhat1, bhat2, P1, P2'
        #to_send = [anomscore1.copy(), anomscore2.copy(), b_fuse.copy(), P_fuse.copy(), Q.copy(), bhat1.copy(), bhat2.copy(), P1.copy(), P2.copy(), gain_norms.copy(), log_det_P_trace.copy()]
        to_send = [anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace]
        Send = send_variables(to_send, client)
        if Send:
            fb = attendre_feedback(timeout=5)
            if fb: 
                log_det_P_trace = list(fb.get("log_det_P_trace", log_det_P_trace))
                gain_norms = list(fb.get("gain_norms", gain_norms))
                bhat1 = np.array(fb.get("bhat1", bhat1))
                bhat2 = np.array(fb.get("bhat2", bhat2))
                P1 = np.array(fb.get("P1", P1))
                P2 = np.array(fb.get("P2", P2))
                b_fuse = np.array(fb.get("b_fuse", b_fuse))
                P_fuse = np.array(fb.get("P_fuse", P_fuse))
                Q = np.array(fb.get("Q", Q))
                det_P_trace = list(fb.get("det_P_trace", det_P_trace))
        else:
            print("[MAIN] Aucun feedback reçu, on continue avec les anciens modèles.")
        
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
    plt.plot(det_P_trace)
    plt.title("Determinant of Kalman covariance matrix P over time")
    plt.xlabel("Update step")
    plt.ylabel("det(P)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(det_P_trace)
    return metrics, test_results, train_res, models

def main():

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_ACK)
    client.loop_start()

    x_test = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_x_test.npy')
    x_train = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_x_train.npy')
    y_test = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_y_test.npy')
    y_train = np.load('C:\\Users\\hadri\\Documents\\Projets\\Projet Finlande - Université Jyväskylä\\Proj_Finland\\ML\\Kalman_Filter_Algorithm\\data\\salinas_y_train.npy')

    labels = np.unique(y_train)


    idx=0
    for l in labels:
        y_train[y_train==l]=idx
        y_test[y_test==l]=idx
        idx +=1

    R, y_R = select_R(x_train, y_train, n_per_class=5, random_seed=10)

    uav1_rows = x_train[:,0:int(x_train.shape[1]/2),:]
    uav2_rows = x_train[:,int(x_train.shape[1]/2)::,]

    # Run simulation
    metrics, test_results, train_res, models= run_simulation(x_train, y_train, x_test, y_test, R, y_R, uav1_rows, uav2_rows, 0.94736842, 1.94736842, client)

    ### NOT NECESSARY -- GRAPHS
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
    time.sleep(2)
    client.loop_stop()
    client.disconnect()
    print("[TRANSMITTER] Fin de transmission.")

if __name__ == "__main__":
    main()