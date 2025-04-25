#!/bin/bash
# Mosquitto MQTT Broker Setup Script

CONFIG_PATH="/home/sapejura/Documents/Communication_RaspberryPi/Proj_Finland/Setup/mosquitto_custom.conf"

echo "[INFO] Launching Mosquitto with file $CONFIG_PATH..."
mosquitto -v -c "$CONFIG_PATH"