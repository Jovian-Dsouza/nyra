import asyncio
import logging
import os
from collections.abc import AsyncIterable
from datetime import datetime

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobProcess,
    ModelSettings,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    llm,
)
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
from hermes_bridge import (
    HermesResultAnnouncer,
    HermesTaskManager,
    load_hermes_settings,
)
from hermes_bridge.tasks import upsert_hermes_results_message
from nyra_speech import (
    GREETING_TEXT,
    WaitingSpeechController,
    load_filler_settings,
    load_min_interruption_words,
)
from nyra_ui.bridge import get_ui_client
from nyra_ui.launcher import start_ui_process, stop_ui_process
from nyra_wakeword import WakeWordController, load_wakeword_settings

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_CONTEXT_ID = "nyra_memory_context"

HERMES_INSTRUCTIONS = """
You are Nyra, a real-time voice assistant. Keep conversation flowing naturally.

Wake word / standby:
- The spoken wake word activates you from passive listening. This is separate from
  background Hermes task delegation (delegate_to_hermes).
- When the conversation is clearly finished — goodbye, "that's all", task fully resolved
  with no follow-up expected — call enter_standby before your closing remark.
- Do not call enter_standby mid-task or while the user may still respond.
- After standby, the user re-engages by saying the wake word again.

Routing:
- Handle inline: greetings, chitchat, datetime, quick facts, clarifying questions.
- Delegate to Hermes (delegate_to_hermes): research, web browsing, file or terminal work,
  multi-step projects, anything that may take more than ten seconds.
- Schedule with Hermes (schedule_hermes_task): requests with a future time like
  "tomorrow at 9am" or "every Monday".

When delegating:
1. Confirm what was queued using the task label returned by the tool.
2. Keep talking — never go silent waiting for Hermes.
3. Do not call enter_standby after delegating or while Hermes tasks are still running.
4. If the user asks about progress, use list_hermes_tasks.
5. If the user wants to cancel, use cancel_hermes_task with the task label.
6. Before re-submitting a similar request, check list_hermes_tasks for duplicates.

When background task results appear in context or are announced, reference them naturally.
"""


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
    """Voice assistant with Cognee memory recall and async Hermes delegation."""

    def __init__(
        self,
        memory_settings: MemorySettings,
        hermes: HermesTaskManager | None = None,
        ui_client=None,
        wakeword: WakeWordController | None = None,
    ):
        super().__init__(
            instructions=f"""You are a helpful and friendly voice AI assistant named Nyra.
            You speak clearly and naturally, as if having a phone conversation.
            Be concise but warm in your responses.
            When relevant memory context is provided in the conversation, use it naturally.
            If you don't know something and no memory context applies, be honest about it.
            {HERMES_INSTRUCTIONS}"""
        )
        self._memory_settings = memory_settings
        self._last_injected_memory = ""
        self._hermes = hermes
        self._ui = ui_client
        self._wakeword = wakeword

    @function_tool
    async def enter_standby(self, context: RunContext) -> str:
        """Return to passive wake-word listening when the conversation is clearly finished.

        Call after the user says goodbye, "that's all", or when their request is fully
        resolved and no follow-up is expected. Do not call mid-task or while Hermes
        background tasks are still running.
        """
        if self._wakeword is None or not self._wakeword.enabled:
            return "Standby mode is not enabled in this session."
        if self._hermes is not None and self._hermes.has_active_tasks():
            return (
                "Background tasks are still running — staying active so the user can "
                "see the confirmation and task status."
            )
        await context.wait_for_playout()
        await self._wakeword.enter_passive()
        return "Returning to standby. Say the wake word when you need me again."

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"

    @function_tool
    async def delegate_to_hermes(self, context: RunContext, task: str, context_note: str = "") -> str:
        """Send a task to Hermes for background processing. Returns immediately with a task label.
        Use for research, web browsing, file work, terminal commands, or any slow multi-step work.
        Do not use for quick questions you can answer directly.

        Args:
            task: What Hermes should do.
            context_note: Optional extra context for Hermes.
        """
        if self._hermes is None:
            return "Background task delegation is not available in this session."
        return await self._hermes.delegate(task, context=context_note)

    @function_tool
    async def schedule_hermes_task(self, context: RunContext, task: str, schedule: str) -> str:
        """Schedule a Hermes task to run later on a cron schedule.

        Args:
            task: What Hermes should do when the schedule fires.
            schedule: Cron-style schedule (e.g. '0 9 * * *' for daily at 9am, or natural language
                schedule string accepted by Hermes).
        """
        if self._hermes is None:
            return "Scheduled task delegation is not available in this session."
        return await self._hermes.schedule(task, schedule)

    @function_tool
    async def list_hermes_tasks(self, context: RunContext) -> str:
        """List pending and recently completed background Hermes tasks."""
        if self._hermes is None:
            return "No background task tracking in this session."
        return self._hermes.list_tasks()

    @function_tool
    async def cancel_hermes_task(self, context: RunContext, label: str) -> str:
        """Cancel a background Hermes task by its name (e.g. Weather in Tokyo).

        Args:
            label: The task name returned when the task was queued.
        """
        if self._hermes is None:
            return "Background task cancellation is not available in this session."
        return await self._hermes.cancel(label)

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Recall Cognee memory and Hermes results before the LLM generates a reply."""
        user_text = new_message.text_content
        if user_text:
            if self._ui is not None:
                self._ui.publish_memory_status("recalling")
            memory_text = await recall_for_transcript(user_text, self._memory_settings)
            if self._ui is not None:
                match_count = len([line for line in memory_text.splitlines() if line.strip()])
                self._ui.publish_memory_status("done", match_count=match_count)
            if memory_text and memory_text != self._last_injected_memory:
                _upsert_memory_message(turn_ctx, memory_text)
                self._last_injected_memory = memory_text
                logger.info("[memory] injected context before LLM reply")

        if self._hermes is not None:
            results_text = self._hermes.get_results_context()
            if results_text:
                upsert_hermes_results_message(turn_ctx, results_text)

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        """Mirror the agent's live spoken text to the UI as it's generated.

        This is the same text stream the room transcription is built from
        (RoomOutputOptions(transcription_enabled=True)), so it fires for
        every utterance the agent speaks — say(), generate_reply(), fillers,
        the greeting — regardless of whether that turn was added to chat
        history. Deltas pass through unchanged; only the accumulated text is
        mirrored to the UI, live, so it reflects what's being spoken *now*.
        """
        buffer = ""
        mirror_ui = (
            self._ui is not None
            and (self._wakeword is None or not self._wakeword.enabled or self._wakeword.is_active)
        )
        async for delta in text:
            buffer += delta
            if mirror_ui:
                self._ui.publish_llm(buffer, is_final=False)
            yield delta
        if mirror_ui and buffer:
            self._ui.publish_llm(buffer, is_final=True)

    async def on_enter(self):
        logger.info("Agent session started")
        if self._wakeword is not None and self._wakeword.defer_greeting:
            logger.info("[wakeword] deferring greeting until wake word activation")
            return
        handle = self.session.say(GREETING_TEXT, add_to_chat_ctx=False)
        await handle

    async def on_exit(self):
        logger.info("Agent session ended")


async def entrypoint(ctx: agents.JobContext):
    logger.info("Agent started in room: %s", ctx.room.name)

    memory_settings = load_memory_settings()
    await init_cognee(memory_settings)

    ui = get_ui_client()
    ui.publish_hello(ctx.room.name)

    wakeword_settings = load_wakeword_settings()
    wakeword = WakeWordController(wakeword_settings, ui_client=ui)

    hermes_settings = load_hermes_settings()
    hermes_manager = HermesTaskManager(
        hermes_settings,
        room_name=ctx.room.name,
        ui_client=ui,
        on_long_running=wakeword.enter_passive if wakeword.enabled else None,
    )
    await hermes_manager.startup()

    async def _shutdown() -> None:
        hermes_announcer.stop()
        await wakeword.shutdown()
        await hermes_manager.shutdown()
        await shutdown_cognee(memory_settings)
        await ui.aclose()

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
    hermes_announcer = HermesResultAnnouncer(session, hermes_manager, wakeword=wakeword)
    hermes_announcer.start()
    wakeword.start(session)
    if wakeword.enabled:
        ui.publish_standby()

    agent = Assistant(
        memory_settings=memory_settings,
        hermes=hermes_manager,
        ui_client=ui,
        wakeword=wakeword,
    )

    @session.on("agent_state_changed")
    def on_state_changed(ev):
        logger.info("State: %s -> %s", ev.old_state, ev.new_state)
        hermes_announcer.on_state_changed(ev.new_state)
        wakeword.on_agent_state_changed(ev.new_state, ev.old_state)
        if wakeword.is_active or not wakeword.enabled:
            ui.publish_phase(ev.new_state)
        if ev.new_state == "thinking":
            waiting_speech.start()
        elif ev.old_state == "thinking":
            waiting_speech.stop()

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        if (
            wakeword.enabled
            and wakeword.is_active
            and ev.new_state == "away"
            and ev.old_state != "away"
            and not hermes_manager.has_active_tasks()
        ):
            asyncio.create_task(wakeword.enter_passive())

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        if wakeword.is_active or not wakeword.enabled:
            ui.publish_stt(ev.transcript, ev.is_final)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    ui_process = start_ui_process()
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, initialize_process_timeout=30.0))
    finally:
        stop_ui_process(ui_process)
