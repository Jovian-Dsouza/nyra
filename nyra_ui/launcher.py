"""Spawns the pygame status window as its own OS process.

A separate process (not a background thread in the agent process) is
required on macOS: pygame/SDL needs the true OS main thread, which the
agent's asyncio loop already owns. Running the UI in its own process gives
it an unencumbered main thread and works the same way on every platform.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _ui_enabled() -> bool:
    return _strip_env(os.environ.get("NYRA_UI_ENABLED", "true")).lower() in (
        "1",
        "true",
        "yes",
    )


def start_ui_process() -> subprocess.Popen | None:
    if not _ui_enabled():
        logger.info("[ui] disabled via NYRA_UI_ENABLED")
        return None

    host = _strip_env(os.environ.get("NYRA_UI_HOST", "127.0.0.1"))
    port = _strip_env(os.environ.get("NYRA_UI_PORT", "8790"))

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "nyra_ui", "--host", host, "--port", port],
            env=os.environ.copy(),
        )
    except OSError:
        logger.warning("[ui] failed to launch status window", exc_info=True)
        return None

    logger.info("[ui] status window started (pid %s)", proc.pid)
    return proc


def stop_ui_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
