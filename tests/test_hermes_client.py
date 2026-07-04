import pytest
import httpx

from hermes_bridge.client import HermesClient, HermesClientError
from hermes_bridge.settings import HermesSettings


def _settings() -> HermesSettings:
    return HermesSettings(
        api_url="http://127.0.0.1:8642",
        api_key="test-key",
        session_key_prefix="nyra",
        max_concurrent=3,
        poll_interval=0.1,
        result_max_chars=1200,
        connect_timeout=1.0,
        request_timeout=5.0,
        summarize_model="gpt-4.1-mini",
        openai_api_key="sk-test",
    )


@pytest.mark.asyncio
async def test_health_check_ok(httpx_mock):
    httpx_mock.add_response(url="http://127.0.0.1:8642/health", json={"status": "ok"})
    client = HermesClient(_settings(), session_key="nyra:room-1")
    assert await client.health_check() is True
    await client.close()


@pytest.mark.asyncio
async def test_submit_run_returns_run_id(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8642/v1/runs",
        method="POST",
        json={"run_id": "run_abc123", "status": "started"},
    )
    client = HermesClient(_settings(), session_key="nyra:room-1")
    result = await client.submit_run("research weather in Tokyo")
    assert result.run_id == "run_abc123"
    assert result.status == "started"

    request = httpx_mock.get_request()
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Hermes-Session-Key"] == "nyra:room-1"
    body = httpx_mock.get_request().read().decode()
    assert "research weather in Tokyo" in body
    await client.close()


@pytest.mark.asyncio
async def test_submit_run_raises_on_429(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8642/v1/runs",
        method="POST",
        status_code=429,
        json={"error": "too many requests"},
    )
    client = HermesClient(_settings(), session_key="nyra:room-1")
    with pytest.raises(HermesClientError) as exc_info:
        await client.submit_run("slow task")
    assert exc_info.value.status_code == 429
    await client.close()


@pytest.mark.asyncio
async def test_get_run_completed(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8642/v1/runs/run_abc123",
        json={
            "run_id": "run_abc123",
            "status": "completed",
            "output": "Tokyo is sunny today.",
        },
    )
    client = HermesClient(_settings(), session_key="nyra:room-1")
    status = await client.get_run("run_abc123")
    assert status.status == "completed"
    assert status.output == "Tokyo is sunny today."
    await client.close()


@pytest.mark.asyncio
async def test_stop_run(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8642/v1/runs/run_abc123/stop",
        method="POST",
        json={"status": "stopping"},
    )
    client = HermesClient(_settings(), session_key="nyra:room-1")
    await client.stop_run("run_abc123")
    await client.close()


@pytest.mark.asyncio
async def test_create_scheduled_job(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8642/api/jobs",
        method="POST",
        json={"job": {"id": "job_1", "name": "nyra-task-1", "schedule": "0 9 * * *"}},
    )
    client = HermesClient(_settings(), session_key="nyra:room-1")
    result = await client.create_scheduled_job(
        name="nyra-task-1",
        schedule="0 9 * * *",
        prompt="Send daily briefing",
    )
    assert result.job_id == "job_1"
    assert result.schedule == "0 9 * * *"
    await client.close()
