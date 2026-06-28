#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] project root: ${ROOT_DIR}"
echo "[setup] creating virtualenv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[setup] installing python dependencies"
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"

if [[ "${NYRA_INSTALL_VOICE_DEPS:-0}" == "1" ]]; then
  echo "[setup] installing voice dependencies (openwakeword/sounddevice)"
  "${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements-voice.txt"
fi

if [[ "${NYRA_INSTALL_STT_DEPS:-0}" == "1" ]]; then
  echo "[setup] installing stt dependencies (vosk)"
  "${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements-stt.txt"
fi

echo "[setup] checking audio capture devices"
if ! arecord -l; then
  echo "[setup] WARNING: arecord failed. Configure I2S capture before running nyra service."
fi

echo "[setup] done"
echo "[setup] run: ${VENV_DIR}/bin/python -m orchestrator.main --dev-console"
