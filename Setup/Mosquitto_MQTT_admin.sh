#!/bin/bash
# Mosquitto MQTT Broker Setup Script with admin rights

CONFIG_PATH="/home/sapejura/Documents/Communication_RaspberryPi/Proj_Finland/Setup/mosquitto_custom.conf"

if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] This script must be executed as root (sudo)."
  exit 1
fi

echo "[INFO] Launching Mosquitto with file $CONFIG_PATH..."
sudo mosquitto -v -c "$CONFIG_PATH"