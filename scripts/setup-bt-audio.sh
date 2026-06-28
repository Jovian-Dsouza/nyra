#!/usr/bin/env bash
set -euo pipefail

BT_MAC="34:AF:B3:8C:F0:26"
BT_CARD="bluez_card.34_AF_B3_8C_F0_26"
TIMEOUT=30

log() { echo "[setup-bt-audio] $*"; }

wait_for() {
  local desc="$1" cmd="$2"
  local elapsed=0
  while ! eval "$cmd" >/dev/null 2>&1; do
    if (( elapsed >= TIMEOUT )); then
      log "ERROR: timed out waiting for $desc"
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
}

log "Waiting for PipeWire..."
wait_for "PipeWire" "pactl info"

log "Waiting for Bluetooth..."
wait_for "Bluetooth" "bluetoothctl show | grep -q 'Powered: yes'"

if ! bluetoothctl info "$BT_MAC" 2>/dev/null | grep -q "Connected: yes"; then
  log "Connecting to Echo Dot ($BT_MAC)..."
  bluetoothctl connect "$BT_MAC" || true
  sleep 3
fi

if pactl list cards short 2>/dev/null | grep -q "$BT_CARD"; then
  log "Setting A2DP sink profile..."
  pactl set-card-profile "$BT_CARD" a2dp-sink 2>/dev/null || true
fi

log "Waiting for Bluetooth audio sink..."
wait_for "bluez_output sink" "pactl list short sinks | grep -q bluez_output"

SINK=$(pactl list short sinks | grep bluez_output | awk '{print $2}' | head -1)
log "Setting default sink to $SINK"
pactl set-default-sink "$SINK"

log "Bluetooth audio ready (default sink: $SINK)"
