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

# Load environment variables before cognee reads env at import time in worker processes
load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_CONTEXT_ID = "nyra_memory_context"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class Assistant(Agent):
    """Voice assistant with streaming Cognee memory injection on stable STT segments."""

    def __init__(self, memory_settings: MemorySettings):
        super().__init__(
            instructions="""You are a helpful and friendly voice AI assistant named Nyra.
            You speak clearly and naturally, as if having a phone conversation.
            Be concise but warm in your responses.
            When relevant memory context is provided in the conversation, use it naturally.
            If you don't know something and no memory context applies, be honest about it."""
        )
        self._memory_settings = memory_settings
        self._turn_transcript = ""
        self._last_final_segment = ""
        self._last_injected_memory = ""

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"

    def reset_turn_transcript(self) -> None:
        self._turn_transcript = ""
        self._last_final_segment = ""

    def append_final_segment(self, segment: str) -> None:
        """Accumulate a stable STT segment for turn-end recall."""
        text = segment.strip()
        if not text:
            return
        if text == self._last_final_segment:
            logger.debug("[STT final] duplicate segment skipped: %s", text)
            return

        self._last_final_segment = text
        if self._turn_transcript:
            self._turn_transcript = f"{self._turn_transcript} {text}".strip()
        else:
            self._turn_transcript = text

        logger.info("[STT final] accumulated transcript: %s", self._turn_transcript)

    async def _inject_memory_context(self, memory_text: str, *, chat_ctx: llm.ChatContext | None = None) -> None:
        ctx = chat_ctx.copy() if chat_ctx is not None else self.chat_ctx.copy()
        content = f"[Relevant memory from past conversations]\n{memory_text}"
        existing = ctx.get_by_id(MEMORY_CONTEXT_ID)
        if existing is not None and existing.type == "message":
            idx = ctx.index_by_id(MEMORY_CONTEXT_ID)
            if idx is not None:
                ctx.items[idx] = llm.ChatMessage(
                    role="system",
                    content=[content],
                    id=MEMORY_CONTEXT_ID,
                )
        else:
            ctx.add_message(role="system", content=content, id=MEMORY_CONTEXT_ID)

        await self.update_chat_ctx(ctx)

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Recall Cognee memory before the LLM generates a reply."""
        user_text = new_message.text_content or self._turn_transcript
        if not user_text:
            return

        memory_text = await recall_for_transcript(
            user_text,
            self._memory_settings,
            timeout=self._memory_settings.turn_recall_timeout,
        )
        if not memory_text or memory_text == self._last_injected_memory:
            return

        content = f"[Relevant memory from past conversations]\n{memory_text}"
        existing = turn_ctx.get_by_id(MEMORY_CONTEXT_ID)
        if existing is not None and existing.type == "message":
            idx = turn_ctx.index_by_id(MEMORY_CONTEXT_ID)
            if idx is not None:
                turn_ctx.items[idx] = llm.ChatMessage(
                    role="system",
                    content=[content],
                    id=MEMORY_CONTEXT_ID,
                )
        else:
            turn_ctx.add_message(role="system", content=content, id=MEMORY_CONTEXT_ID)

        await self.update_chat_ctx(turn_ctx)
        self._last_injected_memory = memory_text
        logger.info("[memory] injected context before LLM reply")

    async def on_enter(self):
        logger.info("Agent session started")
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )

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
    agent = Assistant(memory_settings=memory_settings)

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
    )

    @session.on("agent_state_changed")
    def on_state_changed(ev):
        logger.info("State: %s -> %s", ev.old_state, ev.new_state)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        if not ev.is_final:
            logger.debug("[STT interim] %s", ev.transcript)
            return
        logger.info("[STT final] %s", ev.transcript)
        agent.append_final_segment(ev.transcript)

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        if ev.new_state == "speaking":
            agent.reset_turn_transcript()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
