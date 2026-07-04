"""Thread/process-safe hand-off between the agent and the UI window.

Two roles live in this module:

- `UIStateStore` + `UIStateServer`: run inside the UI process. The server
  accepts the (at most one, in practice) agent connection on a background
  thread running its own asyncio loop, decodes newline-delimited JSON events,
  and folds them into a lock-protected `UIState`. The pygame render loop
  (the process's true main thread) calls `store.snapshot()` once per frame.
  The store always holds only the single most recent state, so a slow or
  absent renderer never blocks the agent.

- `UIClient` + `NullUIClient`: run inside the agent's job subprocess. `publish_*`
  methods are synchronous, non-blocking, and never raise — they just enqueue an
  event for a background writer task that owns the actual socket and retries
  with backoff. If the UI process isn't running, events are silently dropped.
"""

import asyncio
import logging
import os
import threading
from typing import Protocol

from nyra_ui.protocol import (
    apply_event,
    encode_event,
    goodbye_event,
    hello_event,
    hermes_tasks_event,
    llm_event,
    memory_event,
    phase_event,
    stt_event,
)
from nyra_ui.state import UIState

logger = logging.getLogger(__name__)


class UIStateStore:
    """Lock-protected latest-state holder, shared between the server thread
    and the pygame render loop's main thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = UIState.initial()

    def apply(self, event: dict) -> None:
        with self._lock:
            self._state = apply_event(self._state, event)

    def snapshot(self) -> UIState:
        with self._lock:
            return self._state


class UIStateServer:
    """Accepts agent connections on a background thread and feeds a UIStateStore."""

    def __init__(self, store: UIStateStore, *, host: str, port: int) -> None:
        self._store = store
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="nyra-ui-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            logger.warning("[ui] server loop crashed", exc_info=True)
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        server = await asyncio.start_server(self._handle_client, self._host, self._port)
        logger.info("[ui] listening on %s:%s", self._host, self._port)
        async with server:
            serve_task = asyncio.create_task(server.serve_forever())
            await self._stop_event.wait()
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                self._apply_line(line)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            self._store.apply(goodbye_event())
            writer.close()

    def _apply_line(self, line: bytes) -> None:
        import json

        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("[ui] dropped malformed event line", exc_info=True)
            return
        self._store.apply(event)


class UIClientProtocol(Protocol):
    def publish_hello(self, room_name: str | None) -> None: ...
    def publish_phase(self, agent_state: str) -> None: ...
    def publish_stt(self, text: str, is_final: bool) -> None: ...
    def publish_llm(self, text: str, is_final: bool = True) -> None: ...
    def publish_memory_status(self, status: str, match_count: int | None = None) -> None: ...
    def publish_hermes_tasks(self, tasks: list[dict]) -> None: ...
    async def aclose(self) -> None: ...


class NullUIClient:
    """No-op stand-in used when the UI is disabled — callers never need None-checks."""

    def publish_hello(self, room_name: str | None) -> None:
        return None

    def publish_phase(self, agent_state: str) -> None:
        return None

    def publish_stt(self, text: str, is_final: bool) -> None:
        return None

    def publish_llm(self, text: str, is_final: bool = True) -> None:
        return None

    def publish_memory_status(self, status: str, match_count: int | None = None) -> None:
        return None

    def publish_hermes_tasks(self, tasks: list[dict]) -> None:
        return None

    async def aclose(self) -> None:
        return None


class UIClient:
    """Agent-side, non-blocking publisher. Must be constructed from a running
    asyncio loop (i.e. from inside `entrypoint()`)."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        max_backoff: float = 10.0,
        queue_size: int = 200,
    ) -> None:
        self._host = host
        self._port = port
        self._max_backoff = max_backoff
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self._task = asyncio.create_task(self._writer_loop(), name="nyra-ui-client")

    def _enqueue(self, event: dict) -> None:
        try:
            self._queue.put_nowait(encode_event(event))
        except asyncio.QueueFull:
            logger.debug("[ui] dropped event, queue full: %s", event.get("type"))

    def publish_hello(self, room_name: str | None) -> None:
        self._enqueue(hello_event(room_name))

    def publish_phase(self, agent_state: str) -> None:
        self._enqueue(phase_event(agent_state))

    def publish_stt(self, text: str, is_final: bool) -> None:
        self._enqueue(stt_event(text, is_final))

    def publish_llm(self, text: str, is_final: bool = True) -> None:
        self._enqueue(llm_event(text, is_final))

    def publish_memory_status(self, status: str, match_count: int | None = None) -> None:
        self._enqueue(memory_event(status, match_count))

    def publish_hermes_tasks(self, tasks: list[dict]) -> None:
        self._enqueue(hermes_tasks_event(tasks))

    async def aclose(self) -> None:
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass

    async def _writer_loop(self) -> None:
        backoff = 0.5
        while True:
            try:
                reader, writer = await asyncio.open_connection(self._host, self._port)
            except OSError:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue

            backoff = 0.5
            try:
                while True:
                    data = await self._queue.get()
                    writer.write(data)
                    await writer.drain()
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("[ui] connection dropped, will retry", exc_info=True)
            except asyncio.CancelledError:
                writer.close()
                raise
            finally:
                writer.close()


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


def get_ui_client() -> "UIClient | NullUIClient":
    enabled = _strip_env(os.environ.get("NYRA_UI_ENABLED", "true")).lower() in (
        "1",
        "true",
        "yes",
    )
    if not enabled:
        return NullUIClient()

    host = _strip_env(os.environ.get("NYRA_UI_HOST", "127.0.0.1"))
    port = int(_strip_env(os.environ.get("NYRA_UI_PORT", "8790")))
    max_backoff = float(_strip_env(os.environ.get("NYRA_UI_RECONNECT_MAX_BACKOFF", "10.0")))
    return UIClient(host=host, port=port, max_backoff=max_backoff)
