#!/bin/bash

# Check if the script is running as root
if [ "$EUID" -ne 0 ]; then
  echo "Error: This script requires root access."
  echo "Please run it using sudo (e.g., sudo ./revert_network.sh)"
  exit 1
fi

echo "Disabling and stopping NetworkManager..."
systemctl disable --now NetworkManager

echo "Enabling and starting systemd-networkd..."
systemctl enable systemd-networkd
systemctl start systemd-networkd

echo "Enabling and starting wpa_supplicant on wlan0..."
systemctl enable wpa_supplicant@wlan0.service
systemctl start wpa_supplicant@wlan0.service

echo "systemd-networkd is now managing your connections. Ready for Tracking."
