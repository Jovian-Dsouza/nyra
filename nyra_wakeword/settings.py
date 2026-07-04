import os
from dataclasses import dataclass
from pathlib import Path


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


@dataclass(frozen=True)
class WakeWordSettings:
    enabled: bool
    model_path: str
    models_dir: str
    placeholder_model_name: str
    threshold: float
    debounce_seconds: float
    ack_on_activate: bool


def load_wakeword_settings() -> WakeWordSettings:
    enabled = _strip_env(os.environ.get("NYRA_WAKEWORD_ENABLED", "true")).lower() in (
        "1",
        "true",
        "yes",
    )
    model_path = _strip_env(os.environ.get("NYRA_WAKEWORD_MODEL_PATH", ""))
    models_dir = _strip_env(os.environ.get("NYRA_WAKEWORD_MODELS_DIR", "models"))
    placeholder = _strip_env(os.environ.get("NYRA_WAKEWORD_PLACEHOLDER_MODEL", "alexa"))
    threshold = float(os.environ.get("NYRA_WAKEWORD_THRESHOLD", "0.5"))
    debounce_seconds = float(os.environ.get("NYRA_WAKEWORD_DEBOUNCE_SECONDS", "2.0"))
    ack_on_activate = _strip_env(os.environ.get("NYRA_WAKEWORD_ACK_ON_ACTIVATE", "true")).lower() in (
        "1",
        "true",
        "yes",
    )
    return WakeWordSettings(
        enabled=enabled,
        model_path=model_path,
        models_dir=models_dir,
        placeholder_model_name=placeholder,
        threshold=threshold,
        debounce_seconds=debounce_seconds,
        ack_on_activate=ack_on_activate,
    )


def resolve_model_spec(settings: WakeWordSettings) -> str | list[str]:
    if settings.model_path:
        path = Path(settings.model_path)
        if path.exists():
            return str(path)
        return settings.model_path
    return settings.placeholder_model_name
