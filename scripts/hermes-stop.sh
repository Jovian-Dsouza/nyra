#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="hermes-dashboard.service"

log() { echo "[hermes-stop] $*"; }

if [[ ! -f "/etc/systemd/system/${SERVICE_NAME}" ]]; then
  log "ERROR: ${SERVICE_NAME} is not installed"
  log "Run: scripts/hermes-start.sh"
  exit 1
fi

log "Stopping ${SERVICE_NAME}..."
sudo systemctl stop "${SERVICE_NAME}"
log "Hermes dashboard stopped."
