from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from pathlib import Path
from collections.abc import Awaitable, Callable

from orchestrator.config import Settings

log = logging.getLogger(__name__)


class WakeWordEngine:
    """Wake-word engine with openWakeWord backend and dev fallback trigger."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._on_wake: Callable[[], Awaitable[None]] | None = None
        self._running = False
        self._backend_enabled = False
        self._queue: asyncio.Queue[None] = asyncio.Queue()
        self._dispatch_task: asyncio.Task[None] | None = None
        self._backend_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self, on_wake: Callable[[], Awaitable[None]]) -> None:
        self._on_wake = on_wake
        self._running = True
        self._backend_enabled = True
        self._loop = asyncio.get_running_loop()
        self._dispatch_task = asyncio.create_task(self._run_dispatch(), name="wakeword-dispatch")
        self._start_backend_thread()
        log.info("wake-word engine started (model=%s)", self._settings.wakeword_model_path)

    async def stop(self) -> None:
        self._running = False
        self._backend_enabled = False
        if self._dispatch_task:
            self._dispatch_task.cancel()
            await asyncio.gather(self._dispatch_task, return_exceptions=True)
        await self._join_backend_thread()
        self._backend_thread = None
        self._dispatch_task = None
        log.info("wake-word engine stopped")

    async def pause_detection(self) -> None:
        if not self._running:
            return
        self._backend_enabled = False
        await self._join_backend_thread()
        self._backend_thread = None
        log.info("wake-word detection paused")

    async def resume_detection(self) -> None:
        if not self._running or self._backend_enabled:
            return
        self._backend_enabled = True
        self._start_backend_thread()
        log.info("wake-word detection resumed")

    async def trigger_now(self) -> None:
        """Force a wake event (used by dev console/tests)."""
        await self._queue.put(None)

    async def _run_dispatch(self) -> None:
        while self._running:
            await self._queue.get()
            if not self._on_wake:
                continue
            try:
                await self._on_wake()
            except Exception:  # pragma: no cover - defensive logging path
                log.exception("wake callback failed")

    def _microphone_loop(self) -> None:
        try:
            detector = _OpenWakeWordDetector.try_create(self._settings)
            if detector is None:
                log.warning("openWakeWord backend unavailable; wake detection is disabled")
                while self._running and self._backend_enabled:
                    time.sleep(0.2)
                return
            detector.run(lambda: self._running and self._backend_enabled, self._emit_wake_from_thread)
        except Exception:  # pragma: no cover - defensive logging path
            log.exception("wake-word backend failed")

    def _emit_wake_from_thread(self) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def _start_backend_thread(self) -> None:
        if self._backend_thread and self._backend_thread.is_alive():
            return
        self._backend_thread = threading.Thread(
            target=self._microphone_loop, name="wakeword-backend", daemon=True
        )
        self._backend_thread.start()

    async def _join_backend_thread(self) -> None:
        if self._backend_thread and self._backend_thread.is_alive():
            log.debug("wake-word backend thread will exit asynchronously")


class _OpenWakeWordDetector:
    def __init__(
        self,
        *,
        model,
        np_module,
        sd_module,
        model_key: str | None,
        threshold: float,
        sample_rate: int,
        frame_samples: int,
        cooldown_s: float,
        input_device: int | None,
        arecord_device: str | None,
    ) -> None:
        self._model = model
        self._np = np_module
        self._sd = sd_module
        self._model_key = model_key
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._cooldown_s = cooldown_s
        self._input_device = input_device
        self._arecord_device = arecord_device
        self._last_detection = 0.0
        self._key_warning_emitted = False

    @classmethod
    def try_create(cls, settings: Settings) -> "_OpenWakeWordDetector | None":
        try:
            import numpy as np
            import openwakeword
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except Exception as exc:
            log.warning("wake-word imports unavailable: %s", exc)
            return None
        try:
            import sounddevice as sd
        except Exception as exc:
            sd = None
            log.warning("sounddevice unavailable (%s); falling back to arecord capture", exc)

        model_key = settings.wakeword_model_key
        model_path = settings.wakeword_model_path
        model_path = _resolve_or_download_model_path(
            settings=settings,
            model_path=model_path,
            model_key=model_key,
            openwakeword_module=openwakeword,
            download_models_fn=download_models,
        )

        if model_path and model_path.exists():
            try:
                feature_models = _feature_model_paths(model_path.parent)
                model = Model(
                    wakeword_models=[str(model_path)],
                    inference_framework="onnx",
                    melspec_model_path=str(feature_models["melspectrogram"]),
                    embedding_model_path=str(feature_models["embedding"]),
                )
                model_key = model_key or model_path.stem
                log.info("using wake-word model file: %s", model_path)
            except Exception as exc:
                log.warning(
                    "failed to load wake-word model '%s': %s",
                    model_path,
                    exc,
                )
                return None
        else:
            log.warning("no usable wake-word model available")
            return None

        return cls(
            model=model,
            np_module=np,
            sd_module=sd,
            model_key=model_key,
            threshold=settings.wakeword_threshold,
            sample_rate=settings.wakeword_sample_rate,
            frame_samples=settings.wakeword_frame_samples,
            cooldown_s=settings.wakeword_cooldown_s,
            input_device=settings.wakeword_input_device,
            arecord_device=settings.wakeword_arecord_device,
        )

    def run(self, is_running: Callable[[], bool], on_wake: Callable[[], None]) -> None:
        if self._sd is not None:
            self._run_sounddevice(is_running, on_wake)
            return
        self._run_arecord(is_running, on_wake)

    def _run_sounddevice(self, is_running: Callable[[], bool], on_wake: Callable[[], None]) -> None:
        with self._sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_samples,
            device=self._input_device,
        ) as stream:
            log.info("openWakeWord listening via sounddevice (device=%s)", self._input_device)
            while is_running():
                audio_bytes, overflowed = stream.read(self._frame_samples)
                if overflowed:
                    log.debug("wake-word stream overflow")
                self._process_frame(audio_bytes, on_wake)

    def _run_arecord(self, is_running: Callable[[], bool], on_wake: Callable[[], None]) -> None:
        frame_bytes = self._frame_samples * 2
        for device in _arecord_device_candidates(self._arecord_device):
            while is_running():
                cmd = [
                    "arecord",
                    "-q",
                    "-f",
                    "S16_LE",
                    "-c",
                    "1",
                    "-r",
                    str(self._sample_rate),
                    "-t",
                    "raw",
                ]
                if device:
                    cmd.extend(["-D", device])
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self._proc = proc
                if not proc.stdout:
                    raise RuntimeError("arecord started without stdout pipe")
                log.info(
                    "openWakeWord listening via arecord%s",
                    f" ({device})" if device else "",
                )
                exited = False
                try:
                    while is_running():
                        audio_bytes = proc.stdout.read(frame_bytes)
                        if len(audio_bytes) != frame_bytes:
                            if proc.poll() is not None:
                                exited = True
                                break
                            continue
                        self._process_frame(audio_bytes, on_wake)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=2)
                if not is_running():
                    return
                if exited:
                    err = b""
                    if proc.stderr:
                        err = proc.stderr.read()
                    log.warning(
                        "wake arecord exited (device=%s, code=%s): %s",
                        device or "default",
                        proc.returncode,
                        err.decode("utf-8", errors="ignore").strip(),
                    )
                    break

    def _process_frame(self, audio_bytes: bytes, on_wake: Callable[[], None]) -> None:
        audio_frame = self._np.frombuffer(audio_bytes, dtype=self._np.int16)
        scores = self._model.predict(audio_frame)
        score = self._select_score(scores)
        now = time.monotonic()
        if score >= self._threshold and (now - self._last_detection) >= self._cooldown_s:
            self._last_detection = now
            on_wake()

    def _select_score(self, scores: dict[str, float]) -> float:
        if not scores:
            return 0.0
        if self._model_key and self._model_key in scores:
            return float(scores[self._model_key])
        if self._model_key and not self._key_warning_emitted:
            self._key_warning_emitted = True
            log.warning(
                "configured wake-word key '%s' not found in model outputs %s; falling back to max score",
                self._model_key,
                sorted(scores.keys()),
            )
        return float(max(scores.values()))


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


def _resolve_or_download_model_path(
    *,
    settings: Settings,
    model_path: Path,
    model_key: str | None,
    openwakeword_module,
    download_models_fn,
) -> Path | None:
    resolved_key = model_key or "alexa"
    models_dir = model_path.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        _ensure_support_models(download_models_fn=download_models_fn, models_dir=models_dir, model_key=resolved_key)
        return model_path

    if resolved_key in getattr(openwakeword_module, "MODELS", {}):
        filename = Path(openwakeword_module.MODELS[resolved_key]["model_path"]).name.replace(
            ".tflite", ".onnx"
        )
        candidate = models_dir / filename
        if candidate.exists():
            _ensure_support_models(
                download_models_fn=download_models_fn, models_dir=models_dir, model_key=resolved_key
            )
            return candidate
        try:
            log.info(
                "downloading openWakeWord model '%s' into %s (one-time)",
                resolved_key,
                models_dir,
            )
            download_models_fn([resolved_key], str(models_dir))
            if candidate.exists():
                return candidate
        except Exception as exc:
            log.warning("failed to download wake-word model '%s': %s", resolved_key, exc)
        return None

    log.warning(
        "unknown NYRA_WAKEWORD_MODEL_KEY='%s'; expected one of %s",
        resolved_key,
        sorted(getattr(openwakeword_module, "MODELS", {}).keys()),
    )
    return None


def _ensure_support_models(*, download_models_fn, models_dir: Path, model_key: str) -> None:
    required = ["melspectrogram.onnx", "embedding_model.onnx"]
    missing = [name for name in required if not (models_dir / name).exists()]
    if not missing:
        return
    log.info("downloading missing wake-word support models into %s", models_dir)
    download_models_fn([model_key], str(models_dir))


def _feature_model_paths(models_dir: Path) -> dict[str, Path]:
    return {
        "melspectrogram": models_dir / "melspectrogram.onnx",
        "embedding": models_dir / "embedding_model.onnx",
    }
