import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_bridge.delivery import HermesResultAnnouncer, SOFT_TIMEOUT_SECONDS


def _session_with_awaitable_say():
    session = MagicMock()

    async def _say_impl(*_args, **_kwargs):
        return None

    session.say = MagicMock(side_effect=lambda *_a, **_k: _say_impl())
    return session


@pytest.mark.asyncio
async def test_announcer_waits_for_idle_state():
    session = _session_with_awaitable_say()

    task_manager = MagicMock()
    task_manager.announce_queue = asyncio.Queue()
    task_manager.mark_announced = MagicMock()

    announcer = HermesResultAnnouncer(session, task_manager)
    announcer.start()

    await task_manager.announce_queue.put("Task one finished successfully.")
    await asyncio.sleep(0.1)

    session.say.assert_not_called()

    announcer.on_state_changed("listening")
    await asyncio.sleep(2.0)

    session.say.assert_called_once()
    args, kwargs = session.say.call_args
    assert "Task one finished successfully" in args[0]
    assert kwargs.get("add_to_chat_ctx") is True
    task_manager.mark_announced.assert_called_once()

    announcer.stop()


@pytest.mark.asyncio
async def test_announcer_coalesces_multiple_results():
    session = _session_with_awaitable_say()

    task_manager = MagicMock()
    task_manager.announce_queue = asyncio.Queue()
    task_manager.mark_announced = MagicMock()

    announcer = HermesResultAnnouncer(session, task_manager)
    announcer.on_state_changed("idle")
    announcer.start()

    await task_manager.announce_queue.put("First result.")
    await task_manager.announce_queue.put("Second result.")
    await asyncio.sleep(2.5)

    session.say.assert_called_once()
    spoken = session.say.call_args[0][0]
    assert "2 background tasks finished" in spoken
    assert task_manager.mark_announced.call_count == 2

    announcer.stop()


@pytest.mark.asyncio
async def test_announcer_force_delivers_after_soft_timeout():
    session = _session_with_awaitable_say()

    task_manager = MagicMock()
    task_manager.announce_queue = asyncio.Queue()
    task_manager.mark_announced = MagicMock()

    announcer = HermesResultAnnouncer(session, task_manager)
    announcer.on_state_changed("speaking")
    announcer.start()

    await task_manager.announce_queue.put("Delayed result.")
    announcer._pending_since = __import__("time").monotonic() - SOFT_TIMEOUT_SECONDS - 1

    await asyncio.sleep(2.5)

    session.say.assert_called_once()
    spoken = session.say.call_args[0][0]
    assert "Quick update when you have a second" in spoken

    announcer.stop()
