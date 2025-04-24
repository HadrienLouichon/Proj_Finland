import paho.mqtt.client as mqtt
import base64
import json
import os

MQTT_BROKER = "192.168.151.25" #Online broker for communication with different network connections.
MQTT_PORT = 1883 # Port for MQTT communication.
USE_TLS = False # Port 8883 is used for secure communication (TLS/SSL).

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

buffers = {}

# Function which subscribes to the main topic
def on_connect(client, userdata, flags, rc): 
    print("[RECEIVER] Connected, waiting for data ...")
    client.subscribe(TOPIC_IMAGE)

# Callback function that handles incoming messages (data) and reconstruct to have the final data.
def on_message(client, userdata, msg):
    try:
        # Get & Decode the incoming message
        payload = json.loads(msg.payload.decode("utf-8"))
        message_id = payload["message_id"]
        filename = payload["filename"]
        chunk_id = payload["chunk_id"]
        total_chunks = payload["total_chunks"]
        data = base64.b64decode(payload["data"])

        print(f"Chunk {chunk_id+1}/{total_chunks} of the file '{filename}' received.")

        if message_id not in buffers:
            buffers[message_id] = {
                "filename": filename,
                "chunks": {},
                "total": total_chunks
            }

        buffers[message_id]["chunks"][chunk_id] = data # Store the current chunk in the buffer

        repository = "files_received"
        os.makedirs(repository, exist_ok=True)
        complete_path = os.path.join(repository, filename)

        if len(buffers[message_id]["chunks"]) == total_chunks: # Reconstruct the file if all chunks have been received
            with open(complete_path, "wb") as f:
                for i in range(total_chunks):
                    f.write(buffers[message_id]["chunks"][i])
            print(f"✅ The data was completely reconstructed here : {complete_path}")
            del buffers[message_id]

        # Send feedback back to the sender (transmitter.py)
        ack_payload = {
            "id": payload["message_id"],
            "status": "Reception ✅"
        }
        client.publish(TOPIC_ACK, json.dumps(ack_payload))

    # except json.JSONDecodeError:
    except Exception as e:
        print("[RECEIVER] ❌ Error :", str(e))
        if "data" in locals():
            client.publish(TOPIC_ACK, json.dumps({
                "message_id": payload.get("message_id"),
                "status": f"Error ❌ : {str(e)}"
            }))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT,60)
client.loop_forever()