from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from livekit import rtc

if TYPE_CHECKING:
    from openwakeword.model import Model

    from nyra_wakeword.settings import WakeWordSettings

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def _resample_to_16k(frame: rtc.AudioFrame) -> np.ndarray:
    samples = np.frombuffer(frame.data.tobytes(), dtype=np.int16)
    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
    if frame.sample_rate == TARGET_SAMPLE_RATE:
        return samples
    if len(samples) == 0:
        return samples
    num_out = max(1, int(len(samples) * TARGET_SAMPLE_RATE / frame.sample_rate))
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num_out, endpoint=False)
    return np.interp(x_new, x_old, samples.astype(np.float32)).astype(np.int16)


class WakeWordDetector:
    """Runs openWakeWord inference on LiveKit audio frames."""

    def __init__(self, settings: WakeWordSettings) -> None:
        self._settings = settings
        self._model: Model | None = None
        self._model_names: list[str] = []
        self._last_detection_at: float | None = None

    @property
    def model_names(self) -> list[str]:
        return list(self._model_names)

    def load(self) -> None:
        if self._model is not None:
            return

        from openwakeword.model import Model
        from openwakeword.utils import download_models

        from nyra_wakeword.settings import resolve_model_spec

        models_dir = Path(self._settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        spec = resolve_model_spec(self._settings)
        model_name = (
            self._settings.placeholder_model_name
            if isinstance(spec, str) and not spec.endswith((".onnx", ".tflite"))
            else Path(spec).stem
            if isinstance(spec, str) and spec.endswith((".onnx", ".tflite"))
            else self._settings.placeholder_model_name
        )
        download_models([model_name], target_directory=str(models_dir))

        if isinstance(spec, str) and not spec.endswith((".onnx", ".tflite")):
            resolved = models_dir / f"{model_name}.onnx"
            if resolved.exists():
                spec = str(resolved)
            else:
                spec = model_name

        feature_kwargs = {
            "inference_framework": "onnx",
            "melspec_model_path": str(models_dir / "melspectrogram.onnx"),
            "embedding_model_path": str(models_dir / "embedding_model.onnx"),
        }

        try:
            self._model = Model(wakeword_models=[spec], **feature_kwargs)
        except ValueError:
            fallback = str(models_dir / f"{self._settings.placeholder_model_name}.onnx")
            if not Path(fallback).exists():
                fallback = self._settings.placeholder_model_name
            self._model = Model(wakeword_models=[fallback], **feature_kwargs)

        self._model_names = list(self._model.models.keys())
        logger.info("[wakeword] loaded models: %s", ", ".join(self._model_names))

    def _predict_sync(self, audio: np.ndarray) -> dict[str, float]:
        if self._model is None or len(audio) == 0:
            return {}
        scores = self._model.predict(audio)
        if isinstance(scores, tuple):
            scores = scores[0]
        return {str(key): float(value) for key, value in scores.items()}

    def _best_score(self, scores: dict[str, float]) -> tuple[str, float]:
        if not scores:
            return "", 0.0
        name = max(scores, key=scores.get)
        return name, scores[name]

    def _debounced(self) -> bool:
        now = time.monotonic()
        if self._last_detection_at is None:
            return True
        return (now - self._last_detection_at) >= self._settings.debounce_seconds

    async def process_frame(self, frame: rtc.AudioFrame) -> tuple[str, float] | None:
        """Return (model_name, score) when the wake word is detected.

        openWakeWord's `Model.predict()` is a streaming API: it keeps its own
        rolling buffer internally and expects only the *new* chunk on each
        call. Passing an ever-growing buffer here would re-feed already-seen
        audio back into that internal buffer every frame, corrupting the
        rolling window and making detection unreliable.
        """
        if self._model is None:
            self.load()

        chunk = _resample_to_16k(frame)
        if len(chunk) == 0:
            return None

        scores = await asyncio.to_thread(self._predict_sync, chunk)
        name, score = self._best_score(scores)
        if score < self._settings.threshold or not self._debounced():
            return None

        self._last_detection_at = time.monotonic()
        if self._model is not None:
            self._model.reset()
        logger.info("[wakeword] detected %s (%.2f)", name, score)
        return name, score
