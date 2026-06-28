from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from orchestrator.config import Settings


log = logging.getLogger(__name__)


class StreamingSTT:
    """Streaming STT using Vosk + arecord with partial/final callbacks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._on_partial: Callable[[str], Awaitable[None]] | None = None
        self._on_final: Callable[[str], Awaitable[None]] | None = None
        self._running = False
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._last_partial = ""
        self._model = None

    async def prepare(self) -> None:
        async with self._lock:
            if self._model is not None:
                return
        model = await asyncio.to_thread(_load_vosk_model, self._settings)
        async with self._lock:
            if self._model is None:
                self._model = model
        if self._model is not None:
            log.info("stt model preloaded")

    async def start(
        self,
        on_partial: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ) -> None:
        async with self._lock:
            if self._running:
                return
            self._on_partial = on_partial
            self._on_final = on_final
            self._loop = asyncio.get_running_loop()
            self._running = True
            self._last_partial = ""
            self._worker = threading.Thread(target=self._capture_loop, name="stt-loop", daemon=True)
            self._worker.start()
        log.info("stt started")

    async def stop(self) -> None:
        async with self._lock:
            if not self._running:
                return
            self._running = False
            proc = self._proc
            self._proc = None
            worker = self._worker
            self._worker = None
            self._on_partial = None
            self._on_final = None
        if proc and proc.poll() is None:
            proc.terminate()
        if worker and worker.is_alive():
            await asyncio.to_thread(worker.join, 2.0)
        log.info("stt stopped")

    async def submit_partial(self, text: str) -> None:
        if self._running and self._on_partial:
            await self._on_partial(text)

    async def submit_final(self, text: str) -> None:
        if self._running and self._on_final:
            await self._on_final(text)

    def _capture_loop(self) -> None:
        try:
            recognizer = _create_recognizer(self._settings, self._model)
            if recognizer is None:
                log.warning("stt backend unavailable; install Vosk model/deps")
                return
            for device in _arecord_device_candidates(self._settings.stt_arecord_device):
                while self._running:
                    cmd = [
                        "arecord",
                        "-q",
                        "-f",
                        "S16_LE",
                        "-c",
                        "1",
                        "-r",
                        str(self._settings.stt_sample_rate),
                        "-t",
                        "raw",
                    ]
                    if device:
                        cmd.extend(["-D", device])
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self._proc = proc
                    if not proc.stdout:
                        raise RuntimeError("arecord started without stdout")
                    log.info("stt listening via arecord%s", f" ({device})" if device else "")
                    exited = False
                    while self._running:
                        data = proc.stdout.read(self._settings.stt_chunk_bytes)
                        if not data:
                            if proc.poll() is not None:
                                exited = True
                                break
                            continue
                        if recognizer.AcceptWaveform(data):
                            text = _extract_text(recognizer.Result(), "text")
                            if text:
                                self._emit_final(text)
                                self._last_partial = ""
                        else:
                            partial = _extract_text(recognizer.PartialResult(), "partial")
                            if partial and partial != self._last_partial:
                                self._last_partial = partial
                                self._emit_partial(partial)
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=2)
                    if not self._running:
                        return
                    if exited:
                        err = b""
                        if proc.stderr:
                            err = proc.stderr.read()
                        log.warning(
                            "stt arecord exited (device=%s, code=%s): %s",
                            device or "default",
                            proc.returncode,
                            err.decode("utf-8", errors="ignore").strip(),
                        )
                        break
        except Exception:  # pragma: no cover - defensive runtime path
            log.exception("stt loop failed")
        finally:
            proc = self._proc
            self._proc = None
            if proc and proc.poll() is None:
                proc.terminate()

    def _emit_partial(self, text: str) -> None:
        if not self._loop or not self._on_partial:
            return
        self._loop.call_soon_threadsafe(asyncio.create_task, self._on_partial(text))

    def _emit_final(self, text: str) -> None:
        if not self._loop or not self._on_final:
            return
        self._loop.call_soon_threadsafe(asyncio.create_task, self._on_final(text))


def _create_recognizer(settings: Settings, model):
    try:
        from vosk import KaldiRecognizer
    except Exception as exc:
        log.warning("vosk import failed: %s", exc)
        return None
    if model is None:
        model = _load_vosk_model(settings)
    if model is None:
        return None
    recognizer = KaldiRecognizer(model, float(settings.stt_sample_rate))
    recognizer.SetWords(True)
    return recognizer


def _load_vosk_model(settings: Settings):
    try:
        from vosk import Model
    except Exception as exc:
        log.warning("vosk import failed: %s", exc)
        return None
    model_path = _resolve_model_path(settings)
    if not model_path.exists():
        log.warning("stt model path not found: %s", model_path)
        return None
    return Model(str(model_path))


def _extract_text(payload: str, key: str) -> str:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    value = parsed.get(key, "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _arecord_device_candidates(device: str | None) -> list[str | None]:
    if not device:
        return [None]
    if device.startswith("hw:"):
        candidates = [device.replace("hw:", "plughw:", 1), device]
    else:
        candidates = [device]
    candidates.append(None)
    seen: set[str | None] = set()
    ordered: list[str | None] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_model_path(settings: Settings) -> Path:
    if _is_valid_vosk_model_dir(settings.stt_model_path):
        return settings.stt_model_path
    if settings.stt_model_path.exists():
        log.warning("existing STT model directory is invalid, reinstalling: %s", settings.stt_model_path)
        shutil.rmtree(settings.stt_model_path, ignore_errors=True)

    target_dir = settings.stt_model_path
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    archive_path = parent_dir / "vosk-model.zip"
    extract_dir = parent_dir / ".extracting"

    try:
        log.info("downloading STT model from %s", settings.stt_model_url)
        with urllib.request.urlopen(settings.stt_model_url) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

        discovered = _find_vosk_model_dir(extract_dir)
        if discovered is None:
            log.warning("downloaded STT archive does not contain a Vosk model directory")
            return settings.stt_model_path

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(discovered), str(target_dir))
        if not _is_valid_vosk_model_dir(target_dir):
            log.warning("downloaded STT model is incomplete after install: %s", target_dir)
            return settings.stt_model_path
        log.info("installed STT model to %s", target_dir)
        return target_dir
    except Exception as exc:
        log.warning("failed to download/install STT model: %s", exc)
        return settings.stt_model_path
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


def _find_vosk_model_dir(root: Path) -> Path | None:
    if _is_valid_vosk_model_dir(root):
        return root
    for candidate in sorted(p for p in root.rglob("*") if p.is_dir()):
        if _is_valid_vosk_model_dir(candidate):
            return candidate
    return None


def _is_valid_vosk_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    required = [
        path / "am" / "final.mdl",
        path / "conf" / "mfcc.conf",
        path / "conf" / "model.conf",
    ]
    return all(p.exists() for p in required)
