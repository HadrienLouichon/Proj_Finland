import paho.mqtt.client as mqtt
import base64
import json
import ssl
import os

MQTT_BROKER = "localhost" #Central HUB that connects devices altogether.
MQTT_PORT = 1883 # Port for MQTT communication.
USE_TLS = False # Port 8883 is used for secure communication (TLS/SSL).

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

#Connection to the MQTT broker and subscription to the main topic (TOPIC_IMAGE).
# The receiver will listen for incoming messages on the specified topic and process them accordingly.
def on_connect(client, userdata, flags, rc):
    print("[RECEIVER] Connected, waiting for data ...")
    client.subscribe(TOPIC_IMAGE)

def on_message(client, userdata, msg):
    try:
        ## Get & Decode the incoming message
        data = json.loads(msg.payload.decode("utf-8"))
        data_name = data.get("filename", f"recu_{data['id']}")
        data_content = base64.b64decode(data["data"])

        repository = "files_received"
        os.makedirs(repository, exist_ok=True)
        complete_path = os.path.join(repository, data_name)

        with open(complete_path, "wb") as f:
            f.write(data_content)

        print(f"[RECEIVER] Data succesfully received and saved here : {complete_path} ✅")

        ## Send feedback back to the sender (receiver.py)
        ack_payload = {
            "id": data["id"],
            "status": "Reception OK ✅"
        }
        client.publish(TOPIC_ACK, json.dumps(ack_payload))

    #except json.JSONDecodeError:
    except Exception as e:
        print("[RECEIVER] ❌ Error :", str(e))
        if "data" in locals():
            client.publish(TOPIC_ACK, json.dumps({
                "id": data.get("id"),
                "status": f"Error ❌ : {str(e)}"
            }))

#Create a new MQTT client instance and set the connection and message handling callbacks.
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
