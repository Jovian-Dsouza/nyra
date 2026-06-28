from __future__ import annotations

import logging
import subprocess


log = logging.getLogger(__name__)


def verify_capture_device(required: bool = True) -> bool:
    """Return True when ALSA reports at least one capture device."""
    result = subprocess.run(["arecord", "-l"], check=False, capture_output=True, text=True)
    has_device = "card " in result.stdout.lower()
    if has_device:
        return True
    message = (
        "No capture device detected (`arecord -l`). Configure I2S overlay in "
        "/boot/firmware/config.txt before running Nyra."
    )
    if required:
        raise RuntimeError(message)
    log.warning(message)
    return False

