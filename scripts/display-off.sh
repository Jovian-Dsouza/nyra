#!/usr/bin/env bash
set -euo pipefail

# Waveshare 3.5" RPi LCD (F) — backlight on GPIO 18 (LCD_BL)
# https://www.waveshare.com/wiki/3.5inch_RPi_LCD_(F)

BACKLIGHT_GPIO=12

log() { echo "[display-off] $*"; }

if ! command -v pinctrl >/dev/null 2>&1; then
  log "ERROR: pinctrl not found (install raspberrypi-utils)"
  exit 1
fi

log "Turning off display backlight (GPIO ${BACKLIGHT_GPIO})..."
sudo pinctrl set "${BACKLIGHT_GPIO}" op dl
log "Display off."
