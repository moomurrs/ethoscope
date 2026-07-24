#!/bin/bash

# Check if the script is running as root
if [ "$EUID" -ne 0 ]; then
  echo "Error: This script requires root access."
  echo "Please run it using sudo (e.g., sudo ./switch_network.sh)"
  exit 1
fi

echo "Stopping and disabling wpa_supplicant on wlan0..."
systemctl stop wpa_supplicant@wlan0.service
systemctl disable wpa_supplicant@wlan0.service

echo "Stopping and disabling systemd-networkd..."
systemctl stop systemd-networkd
systemctl disable systemd-networkd

echo "Enabling and starting NetworkManager..."
systemctl enable --now NetworkManager

echo "NetworkManager is now managing your connections. Ready for online operations"
