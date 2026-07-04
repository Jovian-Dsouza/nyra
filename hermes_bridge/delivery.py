import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from hermes_bridge.tasks import HermesTaskManager

logger = logging.getLogger(__name__)

DELIVERABLE_STATES = frozenset({"idle", "listening"})
SOFT_TIMEOUT_SECONDS = 90.0
COALESCE_WINDOW_SECONDS = 1.5


class HermesResultAnnouncer:
    """Speaks completed Hermes results at natural pauses without interrupting conversation."""

    def __init__(
        self,
        session: "AgentSession",
        task_manager: "HermesTaskManager",
        *,
        wakeword: Any | None = None,
    ) -> None:
        self._session = session
        self._task_manager = task_manager
        self._wakeword = wakeword
        self._agent_state = "initializing"
        self._pending: list[str] = []
        self._pending_since: float | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._delivering = False

    @property
    def agent_state(self) -> str:
        return self._agent_state

    def start(self) -> None:
        if self._loop_task is not None:
            return
        self._loop_task = asyncio.create_task(
            self._delivery_loop(),
            name="HermesResultAnnouncer._delivery_loop",
        )

    def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            self._loop_task = None

    def on_state_changed(self, new_state: str) -> None:
        self._agent_state = new_state

    async def _delivery_loop(self) -> None:
        queue = self._task_manager.announce_queue
        try:
            while True:
                try:
                    summary = await asyncio.wait_for(queue.get(), timeout=0.5)
                    self._pending.append(summary)
                    if self._pending_since is None:
                        self._pending_since = time.monotonic()
                except asyncio.TimeoutError:
                    pass

                if not self._pending or self._delivering:
                    continue

                if not self._can_deliver_now():
                    if self._should_force_deliver():
                        await self._deliver(force_prefix=True)
                    continue

                await asyncio.sleep(COALESCE_WINDOW_SECONDS)
                while not queue.empty():
                    try:
                        self._pending.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                await self._deliver(force_prefix=False)
        except asyncio.CancelledError:
            return

    def _can_deliver_now(self) -> bool:
        return self._agent_state in DELIVERABLE_STATES and not self._delivering

    def _should_force_deliver(self) -> bool:
        if self._pending_since is None:
            return False
        return (time.monotonic() - self._pending_since) >= SOFT_TIMEOUT_SECONDS

    async def _deliver(self, *, force_prefix: bool) -> None:
        if not self._pending:
            return

        self._delivering = True
        try:
            if self._wakeword is not None and self._wakeword.is_passive:
                await self._wakeword.enter_active(reason="hermes_result")

            if len(self._pending) == 1:
                message = self._pending[0]
            else:
                parts = []
                for i, summary in enumerate(self._pending, start=1):
                    parts.append(f"Task {i}: {summary}")
                message = (
                    f"{len(self._pending)} background tasks finished. "
                    + " ".join(parts)
                )

            if force_prefix:
                message = f"Quick update when you have a second — {message}"

            logger.info("[hermes] announcing result")
            handle = self._session.say(message, add_to_chat_ctx=True)
            await handle

            for summary in self._pending:
                self._task_manager.mark_announced(summary)

            self._pending.clear()
            self._pending_since = None
        except Exception:
            logger.warning("[hermes] announcement failed", exc_info=True)
        finally:
            self._delivering = False
