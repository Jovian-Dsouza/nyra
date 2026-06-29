#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="hermes-dashboard.service"
UNIT_SRC="${ROOT_DIR}/scripts/${SERVICE_NAME}"
UNIT_DST="/etc/systemd/system/${SERVICE_NAME}"

log() { echo "[hermes-start] $*"; }

if [[ ! -f "${UNIT_SRC}" ]]; then
  log "ERROR: missing unit file at ${UNIT_SRC}"
  exit 1
fi

if [[ ! -f "${UNIT_DST}" ]] || ! cmp -s "${UNIT_SRC}" "${UNIT_DST}"; then
  log "Installing ${SERVICE_NAME}..."
  sudo cp "${UNIT_SRC}" "${UNIT_DST}"
  sudo systemctl daemon-reload
fi

log "Starting ${SERVICE_NAME}..."
sudo systemctl start "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager
