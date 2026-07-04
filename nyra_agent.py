import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
)
from livekit.agents import llm
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from cognee_memory import (
    MemorySettings,
    init_cognee,
    load_memory_settings,
    recall_for_transcript,
    shutdown_cognee,
)
from nyra_speech import (
    GREETING_TEXT,
    WaitingSpeechController,
    load_filler_settings,
    load_min_interruption_words,
)

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_CONTEXT_ID = "nyra_memory_context"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


def _upsert_memory_message(chat_ctx: llm.ChatContext, memory_text: str) -> None:
    content = f"[Relevant memory from past conversations]\n{memory_text}"
    existing = chat_ctx.get_by_id(MEMORY_CONTEXT_ID)
    if existing is not None and existing.type == "message":
        idx = chat_ctx.index_by_id(MEMORY_CONTEXT_ID)
        if idx is not None:
            chat_ctx.items[idx] = llm.ChatMessage(
                role="system",
                content=[content],
                id=MEMORY_CONTEXT_ID,
            )
            return

    chat_ctx.add_message(role="system", content=content, id=MEMORY_CONTEXT_ID)


class Assistant(Agent):
    """Voice assistant with Cognee memory recall before each LLM reply."""

    def __init__(self, memory_settings: MemorySettings):
        super().__init__(
            instructions="""You are a helpful and friendly voice AI assistant named Nyra.
            You speak clearly and naturally, as if having a phone conversation.
            Be concise but warm in your responses.
            When relevant memory context is provided in the conversation, use it naturally.
            If you don't know something and no memory context applies, be honest about it."""
        )
        self._memory_settings = memory_settings
        self._last_injected_memory = ""

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Recall Cognee memory before the LLM generates a reply."""
        user_text = new_message.text_content
        if not user_text:
            return

        memory_text = await recall_for_transcript(user_text, self._memory_settings)
        if not memory_text or memory_text == self._last_injected_memory:
            return

        _upsert_memory_message(turn_ctx, memory_text)
        self._last_injected_memory = memory_text
        logger.info("[memory] injected context before LLM reply")

    async def on_enter(self):
        logger.info("Agent session started")
        handle = self.session.say(GREETING_TEXT, add_to_chat_ctx=False)
        await handle

    async def on_exit(self):
        logger.info("Agent session ended")


async def entrypoint(ctx: agents.JobContext):
    logger.info("Agent started in room: %s", ctx.room.name)

    memory_settings = load_memory_settings()
    await init_cognee(memory_settings)

    async def _shutdown() -> None:
        await shutdown_cognee(memory_settings)

    ctx.add_shutdown_callback(_shutdown)

    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    filler_delay, filler_min_interval = load_filler_settings()
    min_interruption_words = load_min_interruption_words()

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en",
        ),
        llm=openai.LLM(
            model=os.getenv("LLM_CHOICE", "gpt-4.1-mini"),
            temperature=0.7,
        ),
        tts=openai.TTS(
            voice="echo",
            speed=1.0,
        ),
        vad=vad,
        turn_detection=MultilingualModel(),
        min_interruption_words=min_interruption_words,
    )

    waiting_speech = WaitingSpeechController(
        session,
        delay=filler_delay,
        min_interval=filler_min_interval,
    )
    agent = Assistant(memory_settings=memory_settings)

    @session.on("agent_state_changed")
    def on_state_changed(ev):
        logger.info("State: %s -> %s", ev.old_state, ev.new_state)
        if ev.new_state == "thinking":
            waiting_speech.start()
        elif ev.old_state == "thinking":
            waiting_speech.stop()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
