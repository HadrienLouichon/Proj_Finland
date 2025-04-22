import paho.mqtt.client as mqtt
import base64
import json
import ssl

MQTT_BROKER = "localhost"  # ou par ex. "example.cloudmqtt.com"
MQTT_PORT = 1883
USE_TLS = False
MQTT_USERNAME = ""
MQTT_PASSWORD = ""

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

def on_connect(client, userdata, flags, rc):
    print("[RECEPTEUR] Connecté, abonnement au topic…")
    client.subscribe(TOPIC_IMAGE)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        message_id = data.get("id")
        image_data = data.get("image")

        with open("files_received/image_recue.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))
        print("[RECEPTEUR] Image sauvegardée.")

        ack_payload = {
            "id": message_id,
            "status": "Image bien reçue ✅"
        }
        client.publish(TOPIC_ACK, json.dumps(ack_payload))

    except Exception as e:
        print("[RECEPTEUR] Erreur :", str(e))
        client.publish(TOPIC_ACK, json.dumps({
            "id": data.get("id") if "data" in locals() else None,
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
