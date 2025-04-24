import paho.mqtt.client as mqtt
import base64
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

def on_connect(client, userdata, flags, rc):
    print("[RECEPTEUR] Connecté, en attente de messages...")
    client.subscribe(TOPIC_IMAGE)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        fichier_nom = data.get("filename", f"recu_{data['id']}")
        contenu = base64.b64decode(data["data"])

        dossier = "files_received"
        os.makedirs(dossier, exist_ok=True)
        chemin = os.path.join(dossier, fichier_nom)

        with open(chemin, "wb") as f:
            f.write(contenu)

        print(f"[RECEPTEUR] Fichier reçu et sauvegardé : {chemin}")

        ack_payload = {
            "id": data["id"],
            "status": "Réception OK ✅"
        }
        client.publish(TOPIC_ACK, json.dumps(ack_payload))

    except Exception as e:
        print("[RECEPTEUR] Erreur :", str(e))
        if "data" in locals():
            client.publish(TOPIC_ACK, json.dumps({
                "id": data.get("id"),
                "status": f"Erreur ❌ : {str(e)}"
            }))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

if USE_TLS:
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

if MQTT_USERNAME:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
