import paho.mqtt.client as mqtt
import base64
import time
import uuid
import json
import ssl
import os

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
USE_TLS = False
MQTT_USERNAME = ""
MQTT_PASSWORD = ""

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

ack_recu = False
message_id_en_cours = ""

def on_message(client, userdata, msg):
    global ack_recu
    ack = json.loads(msg.payload.decode('utf-8'))
    if ack.get("id") == message_id_en_cours:
        ack_recu = True
        print(f"[SIMULATEUR] Accusé de réception : {ack.get('status')}")

client = mqtt.Client()
client.on_message = on_message

if USE_TLS:
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

if MQTT_USERNAME:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(TOPIC_ACK)
client.loop_start()

# === Liste des fichiers à envoyer
fichiers = ["files_to_send/wordfile.docx", "files_to_send/pdffile.pdf", "files_to_send/scriptfile.py"]

for fichier in fichiers:
    ack_recu = False
    message_id_en_cours = str(uuid.uuid4())

    with open(fichier, "rb") as f:
        contenu_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "id": message_id_en_cours,
        "filename": os.path.basename(fichier),
        "data": contenu_base64
    }

    max_essais = 3
    for tentative in range(1, max_essais + 1):
        print(f"[SIMULATEUR] Envoi de {fichier} - tentative {tentative}")
        client.publish(TOPIC_IMAGE, json.dumps(payload))

        for _ in range(10):  # timeout 5 sec
            if ack_recu:
                break
            time.sleep(0.5)

        if ack_recu:
            print(f"[SIMULATEUR] {fichier} → envoyé avec succès ✅")
            break
        else:
            print(f"[SIMULATEUR] Pas d’ACK pour {fichier}, nouvelle tentative...")

client.loop_stop()
client.disconnect()
print("[SIMULATEUR] Fin du programme.")
