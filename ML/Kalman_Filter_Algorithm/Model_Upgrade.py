import numpy as np
from numpy.linalg import pinv
import paho.mqtt.client as mqtt
import base64
import json
import os
from io import BytesIO

MQTT_BROKER = "localhost"  # Adresse du broker
MQTT_PORT = 1883
USE_TLS = False

TOPIC_Data = "pi/data"
TOPIC_ACK = "pc/ack"     # Topic pour envoyer le feedback

buffers = {}  # Buffer pour reconstruire les chunks
received_vars = {}

#------MQTT-------#

def decode_variable(b64_str, as_numpy=False):
    raw = base64.b64decode(b64_str)
    buf = BytesIO(raw)
    if as_numpy:
        return np.load(buf)
    else:
        return json.loads(raw.decode("utf-8"))

def encode_variable(var):
    buf = BytesIO()
    if isinstance(var, np.ndarray):
        np.save(buf, var)
    else:
        buf.write(json.dumps(var).encode("utf-8"))
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def on_connect(client, userdata, flags, rc):
    print("[RECEIVER] Connected, waiting for data ...")
    client.subscribe(TOPIC_Data)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        message_id = payload["message_id"]
        chunk_id = payload["chunk_id"]
        total_chunks = payload["total_chunks"]
        data = base64.b64decode(payload["data"])

        if message_id not in buffers:
            buffers[message_id] = {
                "chunks": {},
                "total": total_chunks
            }

        buffers[message_id]["chunks"][chunk_id] = data
        print(f"[RECEIVER] Chunk {chunk_id+1}/{total_chunks} received.")

        # Reconstruction complète
        if len(buffers[message_id]["chunks"]) == total_chunks:
            full_data = b"".join(buffers[message_id]["chunks"][i] for i in range(total_chunks))
            del buffers[message_id]  # Nettoyage

            # Décodage des variables
            vars_payload = json.loads(full_data.decode("utf-8"))
            for k, v in vars_payload.items():
                if k.startswith("np_") or k.startswith("var") and "array" in str(type(v)).lower():
                    received_vars[k] = decode_variable(v, as_numpy=True)
                else:
                    received_vars[k] = decode_variable(v, as_numpy=False)

            for k, v in received_vars.items():
                anomscore1 = v[0]
                anomscore2 = v[1]
                b_fuse = v[2]
                P_fuse = v[3]
                Q = v[4]
                bhat1 = v[5]
                bhat2 = v[6]
                P1 = v[7]
                P2 = v[8]
                gain_norms = v[9]

            # Send data to Model_Upgrade
            log_det_P, gain_norm, bhat1, bhat2, P1, P2 = upgrading_model(anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms)

            ##DONNEES A FEEDBACK
            feedback_data = { 
                "log_det_P": encode_variable(log_det_P),
                "gain_norm": encode_variable(gain_norm),
                "bhat1": encode_variable(bhat1),
                "bhat2": encode_variable(bhat2),
                "P1": encode_variable(P1),
                "P2": encode_variable(P2),
            }
            ack_payload = {
                "id": message_id,
                "status": "Reception ✅ + Variables stored",
                "extra_data": feedback_data
            }
            client.publish(TOPIC_ACK, json.dumps(ack_payload))
            print("[RECEIVER] 📤 Feedback sent with extra variables.")

    except Exception as e:
        print("[RECEIVER] ❌ Error:", str(e))
        client.publish(TOPIC_ACK, json.dumps({
            "id": payload.get("message_id", "unknown"),
            "status": f"Error ❌ : {str(e)}"
        }))
#------IMAGE-------#

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

def upgrading_model(anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms):

    'HERE'
    'Send data to Model_Upgrade : anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms'
    'Receive : log_det_P, gain_norm, bhat1, bhat2, P1, P2'
    
    # Update with UAV1 model
    if any(anomscore1):
        b_pred, P_pred = kalman_fuse_predict(b_fuse, P_fuse, Q)
        # b_pred, P_pred, K = kalman_fuse_update(b_pred, P_pred, bhat1, P1)

        b_pred, P_pred, K, log_det_P = kalman_fuse_update_adaptive(b_pred, P_pred, bhat1, P1, gain_norms)
        ###log_det_P_trace.append(log_det_P)

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
        ###log_det_P_trace.append(log_det_P)

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

    return log_det_P, gain_norm, bhat1, bhat2, P1, P2

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()