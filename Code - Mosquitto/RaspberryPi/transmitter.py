import paho.mqtt.client as mqtt
import base64
import time
import uuid
import json
import os

MQTT_BROKER = "192.168.151.25" #Online broker for communication with the same network connexion.
MQTT_PORT = 1883 # Port for MQTT communication.
USE_TLS = False # Port 8883 is used for secure communication (TLS/SSL).

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

CHUNK_SIZE = 10_000_000  # 10 Mo = Max size of each chunk of the data

ack_recu = False # Variable to know if the feedback has been received.
current_message_id = "" # Variable to store the ID of the current message.

# Main function that connects to the MQTT broker, sends files through the main topic and also subscribe to the feedback topic.
def main():
    client = mqtt.Client()
    client.on_message = on_message #Callback function to handle incoming messages (feedback).

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_ACK) # Subscribe to the feedback topic.
    client.loop_start()

    repository = "files_to_send"
    filesnames = next(os.walk(repository), (None, None, []))[2] # [] if no file
    
    for file in filesnames:
        path_file = os.path.join(repository, file)
        if os.path.exists(path_file):
            send_file(path_file, client)
        else:
            print(f"❌ This data has not been found : {file}")

    time.sleep(2) # Wait some time between each file to be sent to avoid too much traffic.
    client.loop_stop()
    client.disconnect()
    print("[TRANSMITTER] End of the transmission.")

# Callback function to handle incoming messages (feedback).
def on_message(client, userdata, msg):
        global ack_recu
        ack = json.loads(msg.payload.decode('utf-8'))
        if ack.get("id") == message_id_en_cours:
            ack_recu = True
            print(f"[RECEIVER] '{ack.get('status')}' : the data was succesfully received by the receiver !")

# Function that sends each file to the receiver in chunks.
def send_file(file, client):

    global message_id_en_cours, ack_recu
    ack_recu = False
    message_id = str(uuid.uuid4())
    message_id_en_cours = message_id

    with open(file, "rb") as f:
        file_data = f.read()

    total_chunks = (len(file_data) + CHUNK_SIZE - 1) // CHUNK_SIZE # Calculate the number of chunks needed to send the file.
    filename = os.path.basename(file)

    print(f"\n📤 Sending '{filename}' ({len(file_data)} octets) through {total_chunks} chunks...")

    for i in range(total_chunks):
        chunk = file_data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        payload = {
            "message_id": message_id,
            "filename": filename,
            "chunk_id": i,
            "total_chunks": total_chunks,
            "data": base64.b64encode(chunk).decode("utf-8")
        }
        max_try = 3 # Maximum number of tries to send the chunk (in case of a ponctual error).
        for Try in range(1, max_try + 1):

            print(f"[TRANSMITTER] Sending {filename} - Chunk {i+1}/{total_chunks} - Try n°{Try}")
            client.publish(TOPIC_IMAGE, json.dumps(payload))
            time.sleep(0.1)

            for _ in range(10):  # timeout 5 sec
                if ack_recu:
                    break
                time.sleep(0.5)

            if ack_recu: ## If the feedback is received, we can stop sending the chunk.
                print(f"[RECEIVER] {filename} → was successfully sent.")
                break
            else:
                print(f"[RECEIVER] No feedback for {filename}, new try...")

    print(f"File '{filename}' send with success.")
    time.sleep(1.5)

if __name__ == "__main__":
    main()