import paho.mqtt.client as mqtt
import numpy as np
import base64, json
from io import BytesIO
from datetime import datetime
from mlm_functions import kalman_fuse_predict, kalman_fuse_update_adaptive, init_model

# ===================== CONFIG ===================== #
MQTT_BROKER = "IP_PC"
MQTT_PORT = 1883
TOPIC_RECV = "mlm/uav/+/model/update"
TOPIC_SEND = "mlm/base/model/fused"

VERSION = 0
gain_norm_history = []
Q = 0.01  # Bruit processus

# ===================== INIT MODEL ===================== #
# R, y_R = select_R(X_train, Y_train, n_per_class=5)
# b_fuse, P_fuse = init_model(R, y_R, R, y_R)

def encode_np_array(arr):
    buf = BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def decode_np_array(b64_str):
    return np.load(BytesIO(base64.b64decode(b64_str)))

# ===================== MQTT CALLBACKS ===================== #
def send_fused_model():
    global VERSION
    payload = {
        "bhat_b64": encode_np_array(b_fuse),
        "P_b64": encode_np_array(P_fuse),
        "version": VERSION,
        "ts": datetime.utcnow().isoformat()
    }
    client.publish(TOPIC_SEND, json.dumps(payload))
    print(f"[BASE] Sent fused model version {VERSION}")
    VERSION += 1

def on_connect(client, userdata, flags, rc):
    print("[BASE] Connected to broker")
    client.subscribe(TOPIC_RECV)

def on_message(client, userdata, msg):
    global b_fuse, P_fuse
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        bhat_local = decode_np_array(payload["bhat_b64"])
        P_local = decode_np_array(payload["P_b64"])
        b_pred, P_pred = kalman_fuse_predict(b_fuse, P_fuse, Q)
        b_fuse, P_fuse, gain_norm, logdet = kalman_fuse_update_adaptive(
            b_pred, P_pred, bhat_local, P_local, gain_norm_history)
        send_fused_model()
    except Exception as e:
        print("[BASE] Error handling model update:", e)

# ===================== RUN ===================== #
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
