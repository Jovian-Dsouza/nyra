from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_hermes_command() -> str:
    local_cli = Path.home() / ".local/bin/hermes"
    if local_cli.exists():
        return f"{shlex.quote(str(local_cli))} acp"

    hermes_on_path = shutil.which("hermes")
    if hermes_on_path:
        return f"{shlex.quote(hermes_on_path)} acp"

    hermes_venv_cli = Path.home() / ".hermes/hermes-agent/venv/bin/hermes"
    if hermes_venv_cli.exists():
        return f"{shlex.quote(str(hermes_venv_cli))} acp"

    return "python -m acp_adapter.entry"


def _default_piper_command() -> str:
    local_piper = Path(sys.executable).with_name("piper")
    if local_piper.exists():
        return str(local_piper)
    piper_on_path = shutil.which("piper")
    if piper_on_path:
        return piper_on_path
    return "piper"


def _default_piper_voice_path() -> Path:
    return _resolve_piper_voice_path(Path("models/tts/voice.onnx"))


def _resolve_piper_voice_path(configured_path: Path) -> Path:
    if configured_path.exists():
        return configured_path
    default_path = Path("models/tts/voice.onnx")
    if configured_path != default_path:
        return configured_path
    candidates = sorted(
        path for path in default_path.parent.glob("*.onnx") if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    return configured_path


@dataclass(frozen=True)
class Settings:
    hermes_command: str = field(default_factory=_default_hermes_command)
    piper_command: str = field(default_factory=_default_piper_command)
    piper_voice_path: Path = field(default_factory=_default_piper_voice_path)
    tts_play_command: str | None = None
    wakeword_model_path: Path = Path("models/wakeword/hey_nyra.onnx")
    wakeword_backend: str = "arecord"
    wakeword_model_key: str | None = None
    wakeword_threshold: float = 0.5
    wakeword_sample_rate: int = 16000
    wakeword_frame_samples: int = 1280
    wakeword_cooldown_s: float = 1.2
    wakeword_input_device: int | None = None
    wakeword_arecord_device: str | None = None
    stt_model_path: Path = Path("models/stt/vosk-model-small-en-us-0.15")
    stt_model_url: str = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    stt_sample_rate: int = 16000
    stt_chunk_bytes: int = 4000
    stt_arecord_device: str | None = None
    hermes_session_idle_timeout_s: float = 180.0
    require_audio_device: bool = True
    log_level: str = "INFO"
    dev_console_mode: bool = False

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            hermes_command=os.getenv("NYRA_HERMES_COMMAND", _default_hermes_command()),
            piper_command=os.getenv("NYRA_PIPER_COMMAND", _default_piper_command()),
            piper_voice_path=_resolve_piper_voice_path(
                Path(os.getenv("NYRA_PIPER_VOICE", "models/tts/voice.onnx"))
            ),
            tts_play_command=_parse_optional_str(os.getenv("NYRA_TTS_PLAY_COMMAND")),
            wakeword_model_path=Path(
                os.getenv("NYRA_WAKEWORD_MODEL", "models/wakeword/hey_nyra.onnx")
            ),
            wakeword_backend=os.getenv("NYRA_WAKEWORD_BACKEND", "arecord").strip().lower(),
            wakeword_model_key=_parse_optional_str(os.getenv("NYRA_WAKEWORD_MODEL_KEY")),
            wakeword_threshold=float(os.getenv("NYRA_WAKEWORD_THRESHOLD", "0.5")),
            wakeword_sample_rate=int(os.getenv("NYRA_WAKEWORD_SAMPLE_RATE", "16000")),
            wakeword_frame_samples=int(os.getenv("NYRA_WAKEWORD_FRAME_SAMPLES", "1280")),
            wakeword_cooldown_s=float(os.getenv("NYRA_WAKEWORD_COOLDOWN_S", "1.2")),
            wakeword_input_device=_parse_optional_int(os.getenv("NYRA_WAKEWORD_INPUT_DEVICE")),
            wakeword_arecord_device=_parse_optional_str(os.getenv("NYRA_WAKEWORD_ARECORD_DEVICE")),
            stt_model_path=Path(
                os.getenv("NYRA_STT_MODEL", "models/stt/vosk-model-small-en-us-0.15")
            ),
            stt_model_url=os.getenv(
                "NYRA_STT_MODEL_URL",
                "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            ),
            stt_sample_rate=int(os.getenv("NYRA_STT_SAMPLE_RATE", "16000")),
            stt_chunk_bytes=int(os.getenv("NYRA_STT_CHUNK_BYTES", "4000")),
            stt_arecord_device=_parse_optional_str(os.getenv("NYRA_STT_ARECORD_DEVICE")),
            hermes_session_idle_timeout_s=float(
                os.getenv("NYRA_HERMES_SESSION_IDLE_TIMEOUT_S", "180")
            ),
            require_audio_device=_env_bool("NYRA_REQUIRE_AUDIO", True),
            log_level=os.getenv("NYRA_LOG_LEVEL", "INFO"),
            dev_console_mode=_env_bool("NYRA_DEV_CONSOLE", False),
        )


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None
