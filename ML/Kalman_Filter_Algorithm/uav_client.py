import paho.mqtt.client as mqtt
import numpy as np
import base64, json, os, uuid
from io import BytesIO
from datetime import datetime
from mlm_functions import init_model, predict_model, recursive_ls, select_R

# ===================== CONFIG ===================== #
MQTT_BROKER = "IP_PC"
MQTT_PORT = 1883
TOPIC_SEND = "mlm/uav/uav1/model/update"
TOPIC_RECV = "mlm/base/model/fused"
TOPIC_ACK = "pc/ack"

UAV_ID = "uav1"
VERSION = 0

# ===================== LOAD DATA ===================== #
# Exemple : chargement Salinas depuis .mat (à adapter à ton code)
# Ici on suppose X_train, Y_train sont prêts
# from scipy.io import loadmat
# salinas = loadmat("SalinasA_corrected.mat")["salinasA_corrected"]
# gt = loadmat("SalinasA_gt.mat")["salinasA_gt"]

# Pour simplifier, supposons R et y_R choisis :
# R, y_R = select_R(X_train, Y_train, n_per_class=5)
# bhat_local, P_local = init_model(R, y_R, R, y_R)

# ===================== MQTT CLIENT ===================== #
client = mqtt.Client()

def encode_np_array(arr):
    buf = BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def decode_np_array(b64_str):
    return np.load(BytesIO(base64.b64decode(b64_str)))

def send_model():
    global VERSION
    payload = {
        "uav_id": UAV_ID,
        "ts": datetime.utcnow().isoformat(),
        "version": VERSION,
        "bhat_b64": encode_np_array(bhat_local),
        "P_b64": encode_np_array(P_local)
    }
    msg_id = str(uuid.uuid4())
    payload["message_id"] = msg_id
    # Envoi direct (non chunké) si message < 256kB, sinon adapter avec ton transmitter.py
    client.publish(TOPIC_SEND, json.dumps(payload))
    print(f"[UAV] Model sent (version {VERSION})")
    VERSION += 1

def on_connect(client, userdata, flags, rc):
    print("[UAV] Connected to broker")
    client.subscribe(TOPIC_RECV)

def on_message(client, userdata, msg):
    global bhat_local, P_local
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        bhat_local = decode_np_array(payload["bhat_b64"])
        P_local = decode_np_array(payload["P_b64"])
        print(f"[UAV] Received fused model version {payload['version']}")
    except Exception as e:
        print("[UAV] Error decoding model:", e)

client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ===================== MAIN LOOP ===================== #
client.loop_start()
for batch_x, batch_y in data_stream():  # data_stream() = ta boucle existante sur l'image
    y_pred, anom_score = predict_model(bhat_local, batch_x, R, y_R, neighbours=0)
    if np.any(anom_score > THRESHOLD):  # utilise ta logique actuelle
        bhat_local, P_local = recursive_ls(P_local, x_di, y_di, bhat_local)
        send_model()
