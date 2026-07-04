import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_bridge.client import SubmitRunResult
from hermes_bridge.settings import HermesSettings
from hermes_bridge.tasks import HermesTaskManager
from nyra_agent import Assistant


def _settings() -> HermesSettings:
    return HermesSettings(
        api_url="http://127.0.0.1:8642",
        api_key="test-key",
        session_key_prefix="nyra",
        max_concurrent=3,
        poll_interval=0.01,
        result_max_chars=1200,
        connect_timeout=1.0,
        request_timeout=5.0,
        summarize_model="gpt-4.1-mini",
        openai_api_key="",
    )


@pytest.mark.asyncio
async def test_delegate_tool_returns_without_awaiting_watcher():
    manager = HermesTaskManager(_settings(), room_name="room")
    manager._healthy = True

    submit_started = time.monotonic()
    submit_mock = AsyncMock(
        return_value=SubmitRunResult(run_id="run_fast", status="started")
    )
    manager._client.submit_run = submit_mock

    async def slow_watch(*_args, **_kwargs):
        await __import__("asyncio").sleep(5)

    assistant = Assistant(memory_settings=MagicMock(), hermes=manager)

    with patch.object(manager, "_spawn") as spawn_mock:
        spawn_mock.side_effect = lambda coro: coro.close()
        result = await assistant.delegate_to_hermes(
            MagicMock(),
            task="compile a report",
        )

    elapsed = time.monotonic() - submit_started
    assert elapsed < 1.0
    assert result.startswith("Queued as task-1:")
    submit_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_hermes_tasks_without_manager():
    assistant = Assistant(memory_settings=MagicMock(), hermes=None)
    result = await assistant.list_hermes_tasks(MagicMock())
    assert "background task" in result.lower()


@pytest.mark.asyncio
async def test_schedule_hermes_task():
    manager = HermesTaskManager(_settings(), room_name="room")
    manager._healthy = True
    manager._client.create_scheduled_job = AsyncMock(
        return_value=MagicMock(job_id="job_9", name="nyra-task-1", schedule="0 9 * * *")
    )

    assistant = Assistant(memory_settings=MagicMock(), hermes=manager)
    result = await assistant.schedule_hermes_task(
        MagicMock(),
        task="send briefing",
        schedule="0 9 * * *",
    )
    assert "Scheduled as task-1" in result
