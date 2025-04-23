import paho.mqtt.client as mqtt
import base64
import time
import uuid
import json
import os

MQTT_BROKER = "192.168.151.25" # Put here the IP address of the MQTT broker (device that will receive datas).
MQTT_PORT = 1883 # Port for MQTT communication.
USE_TLS = False # Port 8883 is used for secure communication (TLS/SSL).

TOPIC_IMAGE = "pi/photo"
TOPIC_ACK = "pc/ack"

ack_recu = False # Variable to know if the feedback has been received.
current_message_id = "" # Variable to store the ID of the current message.

def on_message(client, userdata, msg): # Callback function to handle incoming feedback from the receiver.
    global ack_recu
    ack = json.loads(msg.payload.decode('utf-8'))

    if ack.get("id") == current_message_id:
        ack_recu = True
        print(f"[TRANSMITTER] Confirmation of receipt : {ack.get('status')}")

#Create a new MQTT client instance and set the connection.
client = mqtt.Client()
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(TOPIC_ACK)
client.loop_start()

# List of files to send.
files =  []
repository = "files_to_send"
os.makedirs(repository, exist_ok=True)

files_name = ["wordfile.docx", "pdffile.pdf", "scriptfile.py"] #Names of files to send.
for file_name in files_name:
    file = os.path.join(repository, file_name)
    files.append(file)
print(files)

# Loop to send each file to the receiver through the topic (TOPIC_IMAGE).
for file in files:
    ack_recu = False
    current_message_id = str(uuid.uuid4())

    with open(file, "rb") as f: # Read the file in binary mode & Encode it in base64.
        contenu_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "id": current_message_id,
        "filename": os.path.basename(file),
        "data": contenu_base64
    }

    max_try = 3
    for Try in range(1, max_try + 1):
        print(f"[TRANSMITTER] Transmission of {file} - try n°{Try}")
        client.publish(TOPIC_IMAGE, json.dumps(payload)) # Send the file to the receiver.

        for _ in range(10):  # timeout 5 sec
            if ack_recu:
                break
            time.sleep(0.5)

        if ack_recu: # If the feedback is received, break the loop and move to the next file.
            print(f"[TRANSMITTER] {file} was send with success.")
            break
        else:
            print(f"[TRANSMITTER] No feedback for {file}, try again")

client.loop_stop()
client.disconnect()
print("[TRANSMITTER] End of transmission.")
