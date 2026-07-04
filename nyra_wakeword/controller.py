from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any

from nyra_wakeword.detector import WakeWordDetector
from nyra_wakeword.gate import WakeWordGateInput
from nyra_wakeword.settings import WakeWordSettings, load_wakeword_settings

if TYPE_CHECKING:
    from livekit.agents import AgentSession

logger = logging.getLogger(__name__)

FIRST_ACTIVATION_INSTRUCTIONS = (
    "The user just said the wake word to activate you. "
    "Give a brief, warm greeting as Nyra and ask how you can help."
)
REACTIVATION_INSTRUCTIONS = (
    "The user said the wake word again. Acknowledge briefly and ask what they need."
)


class InteractionMode(str, Enum):
    PASSIVE = "passive"
    ACTIVE = "active"


class WakeWordController:
    """Passive/active state machine for wake-word gated voice interaction."""

    def __init__(
        self,
        settings: WakeWordSettings | None = None,
        *,
        ui_client: Any = None,
        echo_tail_seconds: float | None = None,
    ) -> None:
        self._settings = settings or load_wakeword_settings()
        self._ui = ui_client
        self._detector = WakeWordDetector(self._settings)
        self._gate: WakeWordGateInput | None = None
        self._session: AgentSession | None = None
        self._mode = InteractionMode.ACTIVE if not self._settings.enabled else InteractionMode.PASSIVE
        self._activated_once = False
        self._activating = asyncio.Lock()
        self._install_task: asyncio.Task[None] | None = None
        self._suppress_stt = False
        self._stt_resume_task: asyncio.Task[None] | None = None
        if echo_tail_seconds is None:
            echo_tail_seconds = float(
                os.environ.get("NYRA_ECHO_TAIL_SECONDS", "0.35")
            )
        self._echo_tail_seconds = max(0.0, echo_tail_seconds)

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def is_passive(self) -> bool:
        return self.enabled and self._mode is InteractionMode.PASSIVE

    @property
    def is_active(self) -> bool:
        return not self.enabled or self._mode is InteractionMode.ACTIVE

    @property
    def defer_greeting(self) -> bool:
        return self.enabled and not self._activated_once

    def attach_session(self, session: AgentSession) -> None:
        self._session = session

    def start(self, session: AgentSession) -> None:
        self._session = session
        if self.enabled:
            self._detector.load()
        self._install_task = asyncio.create_task(
            self._install_gate_when_ready(session),
            name="WakeWordController._install_gate_when_ready",
        )

    def on_agent_state_changed(self, new_state: str, old_state: str) -> None:
        """Block mic→STT while the agent speaks to prevent speaker echo loops."""
        if new_state in ("speaking", "thinking"):
            if self._stt_resume_task is not None:
                self._stt_resume_task.cancel()
                self._stt_resume_task = None
            self._suppress_stt = True
            self._sync_stt_forward()
            if self._session is not None:
                try:
                    self._session.clear_user_turn()
                except RuntimeError:
                    pass
            return

        if old_state in ("speaking", "thinking") and new_state in ("listening", "idle"):
            if self._echo_tail_seconds <= 0:
                self._suppress_stt = False
                self._sync_stt_forward()
                return

            if self._stt_resume_task is not None:
                self._stt_resume_task.cancel()

            async def _resume() -> None:
                try:
                    await asyncio.sleep(self._echo_tail_seconds)
                    self._suppress_stt = False
                    self._sync_stt_forward()
                except asyncio.CancelledError:
                    pass

            self._stt_resume_task = asyncio.create_task(
                _resume(),
                name="WakeWordController._resume_stt_after_echo_tail",
            )

    def _sync_stt_forward(self) -> None:
        if self._gate is None:
            return
        allow = (not self.enabled or self.is_active) and not self._suppress_stt
        self._gate.set_forward_enabled(allow)

    async def _install_gate_when_ready(self, session: AgentSession) -> None:
        while session.input.audio is None:
            await asyncio.sleep(0.01)

        current = session.input.audio
        if isinstance(current, WakeWordGateInput):
            self._gate = current
        else:
            detector = self._detector if self.enabled else None
            self._gate = WakeWordGateInput(current, detector, self if self.enabled else None)
            session.input.audio = self._gate

        if self.enabled:
            await self.enter_passive(publish_ui=True)
            logger.info("[wakeword] gate installed; starting in passive mode")
        else:
            self._sync_stt_forward()
            logger.info("[audio] echo gate installed for STT input")

    async def on_wake_detected(self, model_name: str, score: float) -> None:
        if not self.enabled or self._mode is InteractionMode.ACTIVE:
            return
        await self.enter_active(reason="wake_word", model_name=model_name, score=score)

    async def enter_active(
        self,
        *,
        reason: str = "wake_word",
        model_name: str = "",
        score: float = 0.0,
    ) -> None:
        if not self.enabled:
            return

        async with self._activating:
            if self._mode is InteractionMode.ACTIVE:
                return

            self._mode = InteractionMode.ACTIVE
            self._sync_stt_forward()
            if self._session is not None:
                self._session.input.set_audio_enabled(True)

            if self._ui is not None:
                self._ui.publish_phase("listening")

            if self._session is None or not self._settings.ack_on_activate:
                self._activated_once = True
                return

            if reason == "wake_word":
                logger.info(
                    "[wakeword] activated by %s (%.2f)",
                    model_name or "wake_word",
                    score,
                )
            else:
                logger.info("[wakeword] activated for %s", reason)

            instructions = (
                FIRST_ACTIVATION_INSTRUCTIONS
                if not self._activated_once
                else REACTIVATION_INSTRUCTIONS
            )
            self._activated_once = True
            handle = self._session.generate_reply(instructions=instructions)
            await handle

    async def enter_passive(self, *, publish_ui: bool = True) -> None:
        if not self.enabled:
            return

        self._mode = InteractionMode.PASSIVE
        self._suppress_stt = False
        if self._stt_resume_task is not None:
            self._stt_resume_task.cancel()
            self._stt_resume_task = None
        self._sync_stt_forward()
        if self._session is not None:
            self._session.input.set_audio_enabled(True)
            try:
                self._session.clear_user_turn()
            except RuntimeError:
                pass

        if publish_ui and self._ui is not None:
            self._ui.publish_standby()

        logger.info("[wakeword] returned to wake-word clock screen")

    async def shutdown(self) -> None:
        if self._stt_resume_task is not None:
            self._stt_resume_task.cancel()
            try:
                await self._stt_resume_task
            except asyncio.CancelledError:
                pass
            self._stt_resume_task = None
        if self._install_task is not None:
            self._install_task.cancel()
            try:
                await self._install_task
            except asyncio.CancelledError:
                pass
            self._install_task = None
