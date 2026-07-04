import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from livekit import rtc

from nyra_wakeword.controller import InteractionMode, WakeWordController
from nyra_wakeword.detector import WakeWordDetector, _resample_to_16k
from nyra_wakeword.gate import WakeWordGateInput
from nyra_wakeword.settings import WakeWordSettings, load_wakeword_settings


def _settings(**overrides) -> WakeWordSettings:
    base = {
        "enabled": True,
        "model_path": "",
        "models_dir": "models",
        "placeholder_model_name": "alexa",
        "threshold": 0.5,
        "debounce_seconds": 2.0,
        "ack_on_activate": True,
    }
    base.update(overrides)
    return WakeWordSettings(**base)


def _frame(sample_rate: int = 24000, samples: int = 2400) -> rtc.AudioFrame:
    data = np.zeros(samples, dtype=np.int16).tobytes()
    return rtc.AudioFrame(data, sample_rate=sample_rate, num_channels=1, samples_per_channel=samples)


def test_load_wakeword_settings_defaults(monkeypatch):
    monkeypatch.delenv("NYRA_WAKEWORD_ENABLED", raising=False)
    monkeypatch.delenv("NYRA_WAKEWORD_THRESHOLD", raising=False)
    monkeypatch.delenv("NYRA_WAKEWORD_PLACEHOLDER_MODEL", raising=False)
    settings = load_wakeword_settings()
    assert settings.enabled is True
    assert settings.threshold == 0.5
    assert settings.placeholder_model_name == "alexa"


def test_resample_to_16k():
    frame = _frame(sample_rate=24000, samples=2400)
    out = _resample_to_16k(frame)
    assert len(out) == 1600


@pytest.mark.asyncio
async def test_gate_blocks_when_passive():
    source = MagicMock()
    frames = [_frame(), _frame()]
    source.__anext__ = AsyncMock(side_effect=frames)

    detector = MagicMock()
    detector.process_frame = AsyncMock(return_value=None)

    controller = MagicMock()
    controller.on_wake_detected = AsyncMock()

    gate = WakeWordGateInput(source, detector, controller)
    gate.set_forward_enabled(False)

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(gate.__anext__(), timeout=0.2)

    assert detector.process_frame.await_count == 2


@pytest.mark.asyncio
async def test_gate_yields_when_active():
    frame = _frame()
    source = MagicMock()
    source.__anext__ = AsyncMock(return_value=frame)

    detector = MagicMock()
    detector.process_frame = AsyncMock(return_value=None)

    controller = MagicMock()
    controller.on_wake_detected = AsyncMock()

    gate = WakeWordGateInput(source, detector, controller)
    gate.set_forward_enabled(True)

    result = await gate.__anext__()
    assert result is frame


@pytest.mark.asyncio
async def test_detection_triggers_enter_active():
    session = MagicMock()
    session.input.set_audio_enabled = MagicMock()
    handle = asyncio.get_running_loop().create_future()
    handle.set_result(None)
    session.generate_reply = MagicMock(return_value=handle)
    session.clear_user_turn = MagicMock()

    ui = MagicMock()
    controller = WakeWordController(_settings(), ui_client=ui)
    controller.attach_session(session)
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    await controller.on_wake_detected("alexa", 0.9)

    assert controller.is_active
    controller._gate.set_forward_enabled.assert_called_with(True)
    ui.publish_phase.assert_called_with("listening")
    session.generate_reply.assert_called_once()


@pytest.mark.asyncio
async def test_enter_passive_publishes_standby():
    session = MagicMock()
    session.input.set_audio_enabled = MagicMock()
    session.clear_user_turn = MagicMock()

    ui = MagicMock()
    controller = WakeWordController(_settings(), ui_client=ui)
    controller.attach_session(session)
    controller._mode = InteractionMode.ACTIVE
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    await controller.enter_passive()

    assert controller.is_passive
    controller._gate.set_forward_enabled.assert_called_with(False)
    ui.publish_standby.assert_called_once()


@pytest.mark.asyncio
async def test_agent_speech_blocks_stt_forwarding():
    session = MagicMock()
    session.clear_user_turn = MagicMock()

    controller = WakeWordController(_settings(), echo_tail_seconds=0.0)
    controller.attach_session(session)
    controller._mode = InteractionMode.ACTIVE
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    controller.on_agent_state_changed("speaking", "listening")
    controller._gate.set_forward_enabled.assert_called_with(False)
    session.clear_user_turn.assert_called_once()

    controller.on_agent_state_changed("listening", "speaking")
    controller._gate.set_forward_enabled.assert_called_with(True)


@pytest.mark.asyncio
async def test_echo_gate_when_wakeword_disabled():
    session = MagicMock()
    session.clear_user_turn = MagicMock()

    controller = WakeWordController(_settings(enabled=False), echo_tail_seconds=0.0)
    controller.attach_session(session)
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    controller.on_agent_state_changed("speaking", "listening")
    controller._gate.set_forward_enabled.assert_called_with(False)

    controller.on_agent_state_changed("listening", "speaking")
    controller._gate.set_forward_enabled.assert_called_with(True)


@pytest.mark.asyncio
async def test_enter_passive_deferred_while_speaking():
    session = MagicMock()
    session.input.set_audio_enabled = MagicMock()
    session.clear_user_turn = MagicMock()

    ui = MagicMock()
    controller = WakeWordController(_settings(), ui_client=ui)
    controller.attach_session(session)
    controller._mode = InteractionMode.ACTIVE
    controller._agent_state = "speaking"
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    await controller.enter_passive()

    assert controller.is_active
    assert controller._pending_passive
    ui.publish_standby.assert_not_called()


@pytest.mark.asyncio
async def test_enter_passive_runs_after_speaking_ends():
    session = MagicMock()
    session.input.set_audio_enabled = MagicMock()
    session.clear_user_turn = MagicMock()

    ui = MagicMock()
    controller = WakeWordController(_settings(), ui_client=ui, echo_tail_seconds=0.0)
    controller.attach_session(session)
    controller._mode = InteractionMode.ACTIVE
    controller._agent_state = "speaking"
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    await controller.enter_passive()
    assert controller._pending_passive

    controller.on_agent_state_changed("listening", "speaking")
    await asyncio.sleep(0)

    assert controller.is_passive
    ui.publish_standby.assert_called_once()


@pytest.mark.asyncio
async def test_enter_active_cancels_deferred_passive():
    session = MagicMock()
    session.input.set_audio_enabled = MagicMock()
    session.generate_reply = MagicMock(
        return_value=asyncio.get_running_loop().create_future()
    )
    session.generate_reply.return_value.set_result(None)
    session.clear_user_turn = MagicMock()

    controller = WakeWordController(_settings(), ui_client=MagicMock())
    controller.attach_session(session)
    controller._mode = InteractionMode.PASSIVE
    controller._agent_state = "speaking"
    controller._gate = MagicMock()
    controller._gate.set_forward_enabled = MagicMock()

    await controller.enter_passive()
    assert controller._pending_passive

    await controller.enter_active(reason="wake_word", model_name="alexa", score=0.9)

    assert not controller._pending_passive
    assert controller.is_active


@pytest.mark.asyncio
async def test_disabled_controller_stays_active():
    session = MagicMock()
    ui = MagicMock()
    controller = WakeWordController(_settings(enabled=False), ui_client=ui)
    controller.attach_session(session)

    await controller.enter_passive()
    await controller.on_wake_detected("alexa", 0.9)

    assert controller.is_active
    ui.publish_standby.assert_not_called()


@pytest.mark.asyncio
async def test_detector_process_frame_with_mock_model():
    detector = WakeWordDetector(_settings(debounce_seconds=0.0))
    mock_model = MagicMock()
    mock_model.predict.return_value = {"alexa": 0.9}
    mock_model.reset = MagicMock()
    mock_model.models = {"alexa_v0.1": MagicMock()}
    detector._model = mock_model
    detector._model_names = ["alexa_v0.1"]

    result = await detector.process_frame(_frame())
    assert result == ("alexa", 0.9)


@pytest.mark.asyncio
async def test_assistant_enter_standby_tool():
    from nyra_agent import Assistant

    wakeword = AsyncMock()
    wakeword.enabled = True
    wakeword.enter_passive = AsyncMock()

    assistant = Assistant(memory_settings=MagicMock(), wakeword=wakeword)
    assistant._ui = None

    context = MagicMock()
    context.wait_for_playout = AsyncMock()

    result = await assistant.enter_standby(context)

    assert "standby" in result.lower()
    context.wait_for_playout.assert_awaited_once()
    wakeword.enter_passive.assert_awaited_once()


@pytest.mark.asyncio
async def test_assistant_enter_standby_blocked_with_active_hermes():
    from nyra_agent import Assistant

    hermes = MagicMock()
    hermes.has_active_tasks = MagicMock(return_value=True)
    wakeword = AsyncMock()
    wakeword.enabled = True
    wakeword.enter_passive = AsyncMock()

    assistant = Assistant(memory_settings=MagicMock(), hermes=hermes, wakeword=wakeword)
    result = await assistant.enter_standby(MagicMock())

    assert "background tasks" in result.lower()
    wakeword.enter_passive.assert_not_awaited()
