import paho.mqtt.client as mqtt
import base64
import time
import uuid
import json
import ssl

# === Configuration du broker ===
MQTT_BROKER = "localhost"  # ou par ex. "example.cloudmqtt.com"
MQTT_PORT = 1883           # 8883 pour TLS, 1883 sans TLS
USE_TLS = False            # ← change à True si tu veux activer TLS
MQTT_USERNAME = ""         # à remplir si broker distant
MQTT_PASSWORD = ""

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

ack_recu = False
message_id = str(uuid.uuid4())

def on_message(client, userdata, msg):
    global ack_recu
    ack = json.loads(msg.payload.decode('utf-8'))
    if ack.get("id") == message_id:
        ack_recu = True
        print(f"[SIMULATEUR] Accusé de réception : {ack.get('status')}")
    else:
        print("[SIMULATEUR] ACK reçu avec mauvais ID")

client = mqtt.Client()
client.on_message = on_message

if USE_TLS:
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

if MQTT_USERNAME:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(TOPIC_ACK)

# Lire et encoder l'image
with open("files_to_send/image.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

message_payload = {
    "id": message_id,
    "image": encoded
}

client.loop_start()

max_essais = 3
for tentative in range(1, max_essais + 1):
    print(f"[SIMULATEUR] Envoi tentative {tentative}")
    client.publish(TOPIC_IMAGE, json.dumps(message_payload))

    for _ in range(10):  # timeout 5 secondes
        if ack_recu:
            break
        time.sleep(0.5)

    if ack_recu:
        break
    else:
        print("[SIMULATEUR] Pas d'ACK, nouvelle tentative…")

client.loop_stop()
client.disconnect()
print("[SIMULATEUR] Fin du programme.")
