import numpy as np
from numpy.linalg import pinv
import paho.mqtt.client as mqtt
import base64
import json
import os
from io import BytesIO

### Broker MQTT Configuration, change the following variable with the IP Adress of the Raspberry / PC containing the Machine Learning Algorithm.
MQTT_BROKER = "localhost"  # <--- TO CHANGE
MQTT_PORT = 1883
USE_TLS = False

TOPIC_Data = "pi/data"
TOPIC_ACK = "pc/ack"

buffers = {}
received_vars = {}

#------MQTT Functions-------#

def decode_variable(b64_str):
    '''
    This function decode all type of variable of base64 type.
    Input : Variable(s) to decode.
    Output : Decoded variable(s).
    '''
    buf = BytesIO(base64.b64decode(b64_str))
    return np.load(buf, allow_pickle=True)

def encode_variable(var):
    '''
    This function encode all type of variable in base64.
    Input : Variable(s) to encode.
    Output : Encoded variable(s) in base64.
    '''
    buf = BytesIO()
    np.save(buf, var)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def on_connect(client, userdata, flags, rc):
    '''
    This function subscribe to the main topic to receive the data from the transmitter.
    '''
    print("[RECEIVER] Connected, waiting for data ...")
    client.subscribe(TOPIC_Data)

def on_message(client, userdata, msg):
    '''
    This function is called when a message is received from the MQTT broker. It decodes the message and reconstruct any potential chunks,
    and sends a feedback to the sender.
    Input : MQTT Client, Userdata, Message.
    Output : None.
    '''
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        message_id = payload["message_id"]
        chunk_id = payload["chunk_id"]
        total_chunks = payload["total_chunks"]
        data = base64.b64decode(payload["data"])

        if message_id not in buffers:
            buffers[message_id] = {"chunks": {}, "total": total_chunks}

        buffers[message_id]["chunks"][chunk_id] = data

        # Reconstruction of the chunks.
        if len(buffers[message_id]["chunks"]) == total_chunks:
            full_data = b"".join(buffers[message_id]["chunks"][i] for i in range(total_chunks))
            del buffers[message_id]
            vars_payload = json.loads(full_data.decode("utf-8"))

            ## Received data from the transmitter : anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace.
            anomscore1 = np.array(decode_variable(vars_payload["var1"]))
            anomscore2 = np.array(decode_variable(vars_payload["var2"]))
            b_fuse     = np.array(decode_variable(vars_payload["var3"]))
            P_fuse     = np.array(decode_variable(vars_payload["var4"]))
            Q          = np.array(decode_variable(vars_payload["var5"]))
            bhat1      = np.array(decode_variable(vars_payload["var6"]))
            bhat2      = np.array(decode_variable(vars_payload["var7"]))
            P1         = np.array(decode_variable(vars_payload["var8"]))
            P2         = np.array(decode_variable(vars_payload["var9"]))
            gain_norms = list(decode_variable(vars_payload["var10"]))
            log_det_P_trace = list(decode_variable(vars_payload["var11"]))
            det_P_trace = list(decode_variable(vars_payload["var12"]))
            
            ## Upgrade the model using the Kalman Filter with the received variables.
            log_det_P_trace, gain_norms, bhat1, bhat2, P1, P2, b_fuse, P_fuse, Q, det_P_trace = upgrading_model(
                anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace
            )

            ## Send feedback with the updated variables : log_det_P, gain_norm, bhat1, bhat2, P1, P2, b_fuse, P_fuse, Q, det_P_trace.
            feedback_data = {
                "log_det_P_trace": encode_variable(log_det_P_trace),
                "gain_norms": encode_variable(gain_norms),
                "bhat1": encode_variable(bhat1),
                "bhat2": encode_variable(bhat2),
                "P1": encode_variable(P1),
                "P2": encode_variable(P2),
                "b_fuse": encode_variable(b_fuse),
                "P_fuse": encode_variable(P_fuse),
                "Q": encode_variable(Q),
                "det_P_trace": encode_variable(det_P_trace)
            }
            ack_payload = {
                "id": message_id,
                "status": "Latest variables well received and stored.",
                "extra_data": feedback_data
            }
            client.publish(TOPIC_ACK, json.dumps(ack_payload))

    except Exception as e:
        print("Error:", str(e))
        client.publish(TOPIC_ACK, json.dumps({
            "id": payload.get("message_id", "unknown"),
            "status": f"Error : {str(e)}"
        }))

#------Kalman Filter Algorithm-------#

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

def upgrading_model(anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace):
    '''
    This function updates the model using the updated data from the MQTT Broker, and returns the updated variables.
    Input : anomscore1 (array), anomscore2 (array), b_fuse (array), P_fuse (array), Q (array), bhat1 (array), bhat2 (array), P1 (array), P2 (array), gain_norms (list), log_det_P_trace (list), det_P_trace (list)
    Output : log_det_P_trace (list), gain_norms (list), bhat1 (array), bhat2 (array), P1 (array), P2 (array), b_fuse (array), P_fuse (array), Q (array), det_P_trace (list)
    '''
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
        det_P_trace.append(np.linalg.det(P_pred))
        
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
        det_P_trace.append(np.linalg.det(P_pred))

        # Now b_pred, P_pred is your fused model
        b_fuse, P_fuse = b_pred, P_pred

        # Update UAVs
        bhat1 = np.copy(b_fuse) 
        bhat2 = np.copy(b_fuse)
        P1 = np.copy(P_fuse)
        P2 = np.copy(P_fuse)

    return log_det_P_trace, gain_norms, bhat1, bhat2, P1, P2, b_fuse, P_fuse, Q, det_P_trace

### MQTT Client Setup

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()