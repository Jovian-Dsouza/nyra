import os
from dataclasses import dataclass


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


@dataclass(frozen=True)
class HermesSettings:
    api_url: str
    api_key: str
    session_key_prefix: str
    max_concurrent: int
    poll_interval: float
    result_max_chars: int
    connect_timeout: float
    request_timeout: float
    summarize_model: str
    openai_api_key: str
    standby_after_seconds: float

    def session_key_for_room(self, room_name: str) -> str:
        return f"{self.session_key_prefix}:{room_name}"


def load_hermes_settings() -> HermesSettings:
    api_url = _strip_env(os.environ.get("HERMES_API_URL", "http://127.0.0.1:8642")).rstrip("/")
    api_key = _strip_env(os.environ.get("HERMES_API_KEY", ""))
    return HermesSettings(
        api_url=api_url,
        api_key=api_key,
        session_key_prefix=_strip_env(os.environ.get("HERMES_SESSION_KEY", "nyra")),
        max_concurrent=int(os.environ.get("HERMES_MAX_CONCURRENT", "3")),
        poll_interval=float(os.environ.get("HERMES_POLL_INTERVAL", "2.0")),
        result_max_chars=int(os.environ.get("HERMES_RESULT_MAX_CHARS", "1200")),
        connect_timeout=float(os.environ.get("HERMES_CONNECT_TIMEOUT", "3.0")),
        request_timeout=float(os.environ.get("HERMES_REQUEST_TIMEOUT", "15.0")),
        summarize_model=os.environ.get("LLM_CHOICE", "gpt-4.1-mini"),
        openai_api_key=_strip_env(os.environ.get("OPENAI_API_KEY", "")),
        standby_after_seconds=float(os.environ.get("HERMES_STANDBY_AFTER_SECONDS", "45")),
    )


def is_configured(settings: HermesSettings) -> bool:
    return bool(settings.api_key)
