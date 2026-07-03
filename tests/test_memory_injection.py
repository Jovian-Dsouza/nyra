import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from livekit.agents import llm

from cognee_memory import MemorySettings, load_memory_settings, recall_for_transcript
from cognee import SearchType
from nyra_agent import Assistant


@pytest.fixture
def memory_settings():
    return MemorySettings(
        openai_api_key="test-key",
        cognee_llm_model="gpt-4o-mini",
        cognee_root_dir=".cognee_system",
        sessions_dataset="nyra_sessions",
        recall_type=SearchType.CHUNKS,
    )


def test_load_memory_settings_defaults(monkeypatch):
    monkeypatch.delenv("COGNEE_BASE_URL", raising=False)
    monkeypatch.delenv("COGNEE_SERVICE_URL", raising=False)
    monkeypatch.delenv("COGNEE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SESSIONS_DATASET", "my_dataset")
    monkeypatch.setenv("COGNEE_RECALL_TYPE", "CHUNKS")
    settings = load_memory_settings()
    assert settings.sessions_dataset == "my_dataset"
    assert settings.recall_type == SearchType.CHUNKS
    assert settings.use_cloud is False


def test_load_memory_settings_cloud(monkeypatch):
    monkeypatch.setenv("COGNEE_BASE_URL", "https://tenant.example.aws.cognee.ai")
    monkeypatch.setenv("COGNEE_API_KEY", "test-cognee-key")
    settings = load_memory_settings()
    assert settings.use_cloud is True
    assert settings.cognee_base_url == "https://tenant.example.aws.cognee.ai"
    assert settings.recall_timeout == 8.0
    assert settings.turn_recall_timeout == 8.0


def test_append_final_segment_dedupes_and_accumulates(memory_settings):
    agent = Assistant(memory_settings=memory_settings)
    agent.append_final_segment("What was")
    agent.append_final_segment("What was")
    agent.append_final_segment("my trip")
    assert agent._turn_transcript == "What was my trip"


@pytest.mark.asyncio
async def test_recall_for_transcript_empty_query(memory_settings):
    assert await recall_for_transcript("", memory_settings) == ""
    assert await recall_for_transcript("   ", memory_settings) == ""


@pytest.mark.asyncio
async def test_on_user_turn_completed_injects_memory(memory_settings):
    agent = Assistant(memory_settings=memory_settings)
    agent._turn_transcript = "Who is Jovan?"

    turn_ctx = llm.ChatContext()
    new_message = llm.ChatMessage(role="user", content=["Who is Jovan?"])

    with patch(
        "nyra_agent.recall_for_transcript",
        new=AsyncMock(return_value="Jovan is 30 and lives in NYC."),
    ):
        with patch.object(agent, "update_chat_ctx", new=AsyncMock()) as update_mock:
            await agent.on_user_turn_completed(turn_ctx, new_message)

    memory_msg = turn_ctx.get_by_id("nyra_memory_context")
    assert memory_msg is not None
    assert "Jovan is 30" in memory_msg.text_content
    update_mock.assert_called_once()
    assert agent._last_injected_memory == "Jovan is 30 and lives in NYC."
