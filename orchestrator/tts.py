from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from collections.abc import Awaitable, Callable


log = logging.getLogger(__name__)

SpeakFn = Callable[[str], Awaitable[None]]


async def default_speak(text: str) -> None:
    """Fallback speech path.

    If `spd-say` is available it speaks locally; otherwise this simulates
    synthesis/playback latency so orchestration behavior stays testable.
    """

    if shutil.which("spd-say"):
        proc = await asyncio.create_subprocess_exec("spd-say", text)
        await proc.wait()
        return
    await asyncio.sleep(min(0.2 + len(text) * 0.01, 2.0))


class TTSQueue:
    def __init__(self, speak_fn: SpeakFn | None = None) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._speak_fn = speak_fn or default_speak
        self._worker_task: asyncio.Task[None] | None = None
        self._active_speak_task: asyncio.Task[None] | None = None
        self._interrupt_event = asyncio.Event()
        self._stopped = False

    async def start(self) -> None:
        if self._worker_task:
            return
        self._worker_task = asyncio.create_task(self._worker(), name="tts-worker")
        log.info("tts queue started")

    async def stop(self) -> None:
        self._stopped = True
        await self.interrupt()
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
            self._worker_task = None
        log.info("tts queue stopped")

    async def enqueue(self, text: str) -> None:
        await self._queue.put(text)

    async def interrupt(self) -> None:
        self._interrupt_event.set()
        self._drain_queue_now()
        if self._active_speak_task:
            self._active_speak_task.cancel()
            await asyncio.gather(self._active_speak_task, return_exceptions=True)
            self._active_speak_task = None
        self._interrupt_event.clear()

    async def drain(self, timeout_s: float = 15.0) -> None:
        await asyncio.wait_for(self._queue.join(), timeout=timeout_s)

    async def _worker(self) -> None:
        while not self._stopped:
            text = await self._queue.get()
            try:
                if self._interrupt_event.is_set():
                    continue
                self._active_speak_task = asyncio.create_task(self._speak_fn(text))
                await self._active_speak_task
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive path
                log.exception("tts playback failed")
            finally:
                self._active_speak_task = None
                self._queue.task_done()

    def _drain_queue_now(self) -> None:
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self._queue.task_done()

