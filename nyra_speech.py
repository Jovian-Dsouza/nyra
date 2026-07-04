import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import AgentSession

logger = logging.getLogger(__name__)

GREETING_TEXT = (
    "Hi, I'm Nyra, your personal assistant. How can I help you today?"
)

WAITING_PHRASES = [
    "Give me just a moment — I'm pulling that together.",
    "Still on it — won't be long.",
    "One sec, I'm thinking this through.",
    "Hang tight, I'm almost there.",
    "Just a moment while I work on that.",
    "Bear with me — I'm on it.",
    "Let me take a quick look at that for you.",
    "Working on your request now.",
    "Almost got it — just a second.",
    "I'm digging into that for you.",
    "Hold on, I'm putting the pieces together.",
    "Still working — thanks for your patience.",
    "Give me a beat, I'm nearly there.",
    "Let me sort that out for you real quick.",
    "I'm on the case — won't be a moment.",
    "Just lining things up for you.",
]


DEFAULT_TTS_VOICE = "nova"
DEFAULT_TTS_SPEED = 1.05
DEFAULT_MAX_RESPONSE_WORDS = 50


@dataclass(frozen=True)
class TtsSettings:
    voice: str
    speed: float


def load_tts_settings() -> TtsSettings:
    voice = os.environ.get("NYRA_TTS_VOICE", DEFAULT_TTS_VOICE).strip().lower()
    speed = float(os.environ.get("NYRA_TTS_SPEED", str(DEFAULT_TTS_SPEED)))
    return TtsSettings(voice=voice, speed=speed)


def load_filler_settings() -> tuple[float, float]:
    delay = float(os.environ.get("NYRA_FILLER_DELAY_SECONDS", "2.5"))
    min_interval = float(os.environ.get("NYRA_FILLER_MIN_INTERVAL", "8.0"))
    return delay, min_interval


def load_min_interruption_words() -> int:
    return int(os.environ.get("NYRA_MIN_INTERRUPTION_WORDS", "2"))


def load_max_response_words() -> int:
    """Maximum words Nyra may speak per reply. Set to 0 to disable the limit."""
    return int(os.environ.get("NYRA_MAX_RESPONSE_WORDS", str(DEFAULT_MAX_RESPONSE_WORDS)))


def build_brevity_instructions(max_words: int) -> str:
    if max_words <= 0:
        return ""
    return (
        f"Keep every spoken reply to at most {max_words} words — summary style, not a lecture. "
        f"Answer the core question directly, then stop. "
        f"Do not read long lists aloud; give the top one or two points only. "
        f"If more detail exists, say the user can ask a follow-up."
    )


def count_words(text: str) -> int:
    return len(text.split())


def truncate_to_word_limit(text: str, max_words: int) -> str:
    """Trim text to max_words, preferring to end on a sentence boundary."""
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    for sep in (". ", "! ", "? "):
        idx = truncated.rfind(sep)
        if idx >= len(truncated) // 2:
            return truncated[: idx + 1].strip()

    trimmed = truncated.rstrip(",;:")
    return f"{trimmed}." if trimmed else truncated


class WaitingSpeechController:
    """Speaks a rotating filler phrase after a delay during long waits."""

    def __init__(
        self,
        session: "AgentSession",
        *,
        delay: float = 2.5,
        min_interval: float = 8.0,
    ) -> None:
        self._session = session
        self._delay = delay
        self._min_interval = min_interval
        self._task: asyncio.Task[None] | None = None
        self._handle = None
        self._last_spoken_at: float | None = None
        self._generation = 0

    def _next_phrase(self) -> str:
        return random.choice(WAITING_PHRASES)

    def start(self) -> None:
        self._cancel_pending()
        self._generation += 1
        generation = self._generation
        self._task = asyncio.create_task(
            self._run(generation),
            name="WaitingSpeechController._run",
        )

    def stop(self) -> None:
        self._cancel_pending()
        if self._handle is not None:
            try:
                self._handle.interrupt()
            except RuntimeError:
                # Filler was created without allow_interruptions — already finished.
                pass
            except Exception:
                logger.debug("[speech] filler interrupt failed", exc_info=True)
            self._handle = None

    def _cancel_pending(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self, generation: int) -> None:
        try:
            await asyncio.sleep(self._delay)
            if generation != self._generation:
                return

            now = time.monotonic()
            if (
                self._last_spoken_at is not None
                and now - self._last_spoken_at < self._min_interval
            ):
                return

            phrase = self._next_phrase()
            logger.info("[speech] waiting filler: %s", phrase)
            self._handle = self._session.say(
                phrase,
                add_to_chat_ctx=False,
                allow_interruptions=True,
            )
            self._last_spoken_at = time.monotonic()
        except asyncio.CancelledError:
            return
