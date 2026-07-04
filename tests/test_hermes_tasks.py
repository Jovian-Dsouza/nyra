import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_bridge.client import HermesClientError, RunStatus, SubmitRunResult
from hermes_bridge.settings import HermesSettings, is_configured, load_hermes_settings
from hermes_bridge.tasks import HermesTaskManager, _derive_label_from_prompt, _map_status, _short_description


def _settings(**overrides) -> HermesSettings:
    base = dict(
        api_url="http://127.0.0.1:8642",
        api_key="test-key",
        session_key_prefix="nyra",
        max_concurrent=2,
        poll_interval=0.01,
        result_max_chars=100,
        connect_timeout=1.0,
        request_timeout=5.0,
        summarize_model="gpt-4.1-mini",
        openai_api_key="",
        standby_after_seconds=45.0,
    )
    base.update(overrides)
    return HermesSettings(**base)


def test_is_configured():
    assert is_configured(_settings()) is True
    assert is_configured(_settings(api_key="")) is False


def test_load_hermes_settings_defaults(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "secret")
    monkeypatch.setenv("HERMES_API_URL", "http://localhost:8642")
    settings = load_hermes_settings()
    assert settings.api_key == "secret"
    assert settings.api_url == "http://localhost:8642"
    assert settings.session_key_for_room("living-room") == "nyra:living-room"


def test_map_status():
    assert _map_status("completed") == "completed"
    assert _map_status("running") == "running"
    assert _map_status("waiting_approval") == "waiting_approval"
    assert _map_status("canceled") == "cancelled"


def test_short_description_truncates():
    long_text = "a" * 100
    result = _short_description(long_text, max_len=20)
    assert len(result) <= 20
    assert result.endswith("…")


def test_derive_label_from_prompt():
    assert _derive_label_from_prompt("research the weather in Tokyo") == "Research The Weather In Tokyo"
    assert _derive_label_from_prompt("please find my latest invoices") == "Find My Latest Invoices"


@pytest.mark.asyncio
async def test_delegate_submits_and_returns_immediately():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    manager._healthy = True

    submit_mock = AsyncMock(
        return_value=SubmitRunResult(run_id="run_1", status="started")
    )
    manager._client.submit_run = submit_mock

    with patch.object(manager, "_spawn"):
        result = await manager.delegate("research the weather in Tokyo")

    assert result.startswith("Queued as Research The Weather In Tokyo:")
    assert "run_1" in manager._tasks
    submit_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_delegate_queues_when_at_capacity():
    manager = HermesTaskManager(_settings(max_concurrent=1), room_name="test-room")
    manager._healthy = True
    manager._tasks["run_existing"] = MagicMock(
        status="running",
        submitted_at=0,
        completed_at=None,
        voice_summary=None,
        announced=False,
        prompt="existing",
        label="task-0",
        run_id="run_existing",
        is_scheduled=False,
        raw_output=None,
        error_message=None,
    )

    result = await manager.delegate("another task")
    assert "queued" in result.lower()
    assert len(manager._local_queue) == 1


@pytest.mark.asyncio
async def test_delegate_unavailable_when_healthy_false():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    manager._healthy = False
    result = await manager.delegate("do something")
    assert "can't reach" in result.lower()


@pytest.mark.asyncio
async def test_watch_run_completes_and_queues_announcement():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    manager._tasks["run_1"] = MagicMock(
        run_id="run_1",
        label="task-1",
        prompt="find files",
        status="queued",
        submitted_at=0,
        completed_at=None,
        raw_output=None,
        voice_summary=None,
        error_message=None,
        announced=False,
        is_scheduled=False,
    )

    get_run_mock = AsyncMock(
        side_effect=[
            RunStatus(run_id="run_1", status="running", output=None, error=None),
            RunStatus(run_id="run_1", status="completed", output="Found 3 files.", error=None),
        ]
    )
    manager._client.get_run = get_run_mock
    manager._summarize_for_voice = AsyncMock(return_value="Found three files for you.")

    with patch.object(manager, "_spawn"):
        await manager._watch_run("run_1")

    assert manager._tasks["run_1"].status == "completed"
    assert manager._tasks["run_1"].voice_summary == "Found three files for you."
    assert not manager._announce_queue.empty()


@pytest.mark.asyncio
async def test_cancel_active_task():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    from hermes_bridge.tasks import HermesTask

    task = HermesTask(run_id="run_1", label="Slow Job", prompt="slow job", status="running")
    manager._register_task(task)
    manager._client.stop_run = AsyncMock()

    result = await manager.cancel("Slow Job")
    assert "Cancelled Slow Job" in result
    assert manager._tasks["run_1"].status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_removes_local_queue_entry():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    manager._local_queue.append(("Queued Prompt", "queued prompt"))
    result = await manager.cancel("Queued Prompt")
    assert "Removed Queued Prompt" in result
    assert manager._local_queue == []


def test_has_active_tasks():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    assert manager.has_active_tasks() is False

    from hermes_bridge.tasks import HermesTask

    task = HermesTask(run_id="run_1", label="task-1", prompt="research", status="running")
    manager._register_task(task)
    assert manager.has_active_tasks() is True

    task.status = "completed"
    manager._publish_snapshot()
    assert manager.has_active_tasks() is False


def test_get_results_context_includes_completed_summaries():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    from hermes_bridge.tasks import HermesTask

    task = HermesTask(
        run_id="run_1",
        label="Tokyo Weather",
        prompt="research",
        status="completed",
        voice_summary="Tokyo is sunny.",
        completed_at=1.0,
    )
    manager._register_task(task)
    context = manager.get_results_context()
    assert context is not None
    assert "Tokyo Weather" in context
    assert "Tokyo is sunny" in context


@pytest.mark.asyncio
async def test_delegate_handles_429_from_client():
    manager = HermesTaskManager(_settings(), room_name="test-room")
    manager._healthy = True
    manager._client.submit_run = AsyncMock(
        side_effect=HermesClientError("busy", status_code=429)
    )

    with patch.object(manager, "_spawn"):
        result = await manager.delegate("heavy task")

    assert "queued" in result.lower()
    assert len(manager._local_queue) == 1
