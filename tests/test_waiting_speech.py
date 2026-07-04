import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nyra_speech import (
    DEFAULT_TTS_SPEED,
    DEFAULT_TTS_VOICE,
    GREETING_TEXT,
    WAITING_PHRASES,
    WaitingSpeechController,
    load_filler_settings,
    load_min_interruption_words,
    load_tts_settings,
)


def test_greeting_text():
    assert GREETING_TEXT == (
        "Hi, I'm Nyra, your personal assistant. How can I help you today?"
    )


def test_waiting_phrases_non_empty():
    assert len(WAITING_PHRASES) >= 12
    assert all(phrase.strip() for phrase in WAITING_PHRASES)


def test_load_filler_settings_defaults(monkeypatch):
    monkeypatch.delenv("NYRA_FILLER_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("NYRA_FILLER_MIN_INTERVAL", raising=False)
    delay, min_interval = load_filler_settings()
    assert delay == 2.5
    assert min_interval == 8.0


def test_load_filler_settings_from_env(monkeypatch):
    monkeypatch.setenv("NYRA_FILLER_DELAY_SECONDS", "3.0")
    monkeypatch.setenv("NYRA_FILLER_MIN_INTERVAL", "10.0")
    delay, min_interval = load_filler_settings()
    assert delay == 3.0
    assert min_interval == 10.0


def test_load_min_interruption_words_defaults(monkeypatch):
    monkeypatch.delenv("NYRA_MIN_INTERRUPTION_WORDS", raising=False)
    assert load_min_interruption_words() == 2


def test_load_min_interruption_words_from_env(monkeypatch):
    monkeypatch.setenv("NYRA_MIN_INTERRUPTION_WORDS", "3")
    assert load_min_interruption_words() == 3


def test_load_tts_settings_defaults(monkeypatch):
    monkeypatch.delenv("NYRA_TTS_VOICE", raising=False)
    monkeypatch.delenv("NYRA_TTS_SPEED", raising=False)
    settings = load_tts_settings()
    assert settings.voice == DEFAULT_TTS_VOICE
    assert settings.speed == DEFAULT_TTS_SPEED


def test_load_tts_settings_from_env(monkeypatch):
    monkeypatch.setenv("NYRA_TTS_VOICE", "shimmer")
    monkeypatch.setenv("NYRA_TTS_SPEED", "1.0")
    settings = load_tts_settings()
    assert settings.voice == "shimmer"
    assert settings.speed == 1.0


def test_next_phrase_from_pool():
    session = MagicMock()
    controller = WaitingSpeechController(session, delay=0.01, min_interval=0.0)
    phrase = controller._next_phrase()
    assert phrase in WAITING_PHRASES


@pytest.mark.asyncio
async def test_start_speaks_after_delay():
    session = MagicMock()
    handle = MagicMock()
    session.say.return_value = handle

    controller = WaitingSpeechController(session, delay=0.05, min_interval=0.0)
    with patch.object(controller, "_next_phrase", return_value="Still on it — won't be long."):
        controller.start()
        await asyncio.sleep(0.1)

    session.say.assert_called_once_with(
        "Still on it — won't be long.",
        add_to_chat_ctx=False,
        allow_interruptions=True,
    )


@pytest.mark.asyncio
async def test_stop_cancels_before_say():
    session = MagicMock()
    controller = WaitingSpeechController(session, delay=1.0, min_interval=0.0)

    controller.start()
    await asyncio.sleep(0.01)
    controller.stop()
    await asyncio.sleep(0.05)

    session.say.assert_not_called()


@pytest.mark.asyncio
async def test_min_interval_blocks_rapid_fillers():
    session = MagicMock()
    session.say.return_value = MagicMock()
    controller = WaitingSpeechController(session, delay=0.01, min_interval=60.0)

    with patch.object(controller, "_next_phrase", return_value="Hang tight, I'm almost there."):
        controller.start()
        await asyncio.sleep(0.05)
        controller.stop()

        controller.start()
        await asyncio.sleep(0.05)

    session.say.assert_called_once()


@pytest.mark.asyncio
async def test_stop_interrupts_active_handle():
    session = MagicMock()
    handle = MagicMock()
    session.say.return_value = handle

    controller = WaitingSpeechController(session, delay=0.01, min_interval=0.0)
    with patch.object(controller, "_next_phrase", return_value="Working on your request now."):
        controller.start()
        await asyncio.sleep(0.05)
        controller.stop()

    handle.interrupt.assert_called_once()
