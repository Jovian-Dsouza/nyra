from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections.abc import AsyncIterator
from pathlib import Path


log = logging.getLogger(__name__)


class SentenceAggregator:
    def __init__(self) -> None:
        self._buffer = ""

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        chunks: list[str] = []
        while True:
            idx = self._next_boundary(self._buffer)
            if idx == -1:
                break
            sentence = self._buffer[: idx + 1].strip()
            self._buffer = self._buffer[idx + 1 :].lstrip()
            if sentence:
                chunks.append(sentence)
        return chunks

    def flush_tail(self) -> str | None:
        tail = self._buffer.strip()
        self._buffer = ""
        return tail or None

    @staticmethod
    def _next_boundary(text: str) -> int:
        for i, ch in enumerate(text):
            if ch in ".!?;":
                return i
        return -1


class ACPHermesClient:
    """Streams text deltas from a Hermes ACP subprocess."""

    def __init__(self, hermes_command: str, idle_timeout_s: float = 180.0) -> None:
        self._command = hermes_command
        self._request_id = 0
        self._idle_timeout_s = idle_timeout_s
        self._proc: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._prompt_lock = asyncio.Lock()
        self._idle_close_task: asyncio.Task[None] | None = None
        self._idle_generation = 0

    async def warmup(self) -> None:
        """Start Hermes and create a reusable ACP session before a prompt is ready."""

        async with self._lifecycle_lock:
            await self._ensure_session_locked()
            self._schedule_idle_close_locked()

    async def close(self) -> None:
        async with self._prompt_lock:
            async with self._lifecycle_lock:
                await self._close_locked(reason="shutdown")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with self._prompt_lock:
            async with self._lifecycle_lock:
                proc, session_id = await self._ensure_session_locked()
            try:
                async for delta in self._stream_prompt(proc, session_id, prompt):
                    yield delta
            except Exception:
                async with self._lifecycle_lock:
                    await self._close_locked(reason="error")
                raise
            finally:
                async with self._lifecycle_lock:
                    if self._proc is proc and self._session_id == session_id:
                        self._schedule_idle_close_locked()

    async def _ensure_session_locked(self) -> tuple[asyncio.subprocess.Process, str]:
        self._cancel_idle_close_locked()
        if self._proc and self._session_id and self._proc.returncode is None:
            return self._proc, self._session_id

        if self._proc:
            await self._close_locked(reason="stale-process")

        cmd = shlex.split(self._command)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError("failed to start ACP subprocess with stdio pipes")

        try:
            self._session_id = await self._initialize_session(self._proc)
        except Exception:
            await self._close_locked(reason="error")
            raise
        return self._proc, self._session_id

    async def _stream_prompt(
        self,
        proc: asyncio.subprocess.Process,
        session_id: str,
        prompt: str,
    ) -> AsyncIterator[str]:
        prompt_request_id = await self._send_request(
            proc,
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": prompt}],
            },
        )
        log.info("hermes prompt started (session=%s)", session_id)

        while True:
            line = await proc.stdout.readline() if proc.stdout else b""
            if not line:
                raise RuntimeError("hermes ACP closed during prompt")
            try:
                message = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                continue

            maybe_error = _extract_error(message)
            if maybe_error:
                raise RuntimeError(maybe_error)

            maybe_text = _extract_chunk_text(message)
            if maybe_text:
                yield maybe_text

            if _is_prompt_complete(message, prompt_request_id):
                log.info("hermes prompt completed (session=%s)", session_id)
                return

    def _schedule_idle_close_locked(self) -> None:
        self._cancel_idle_close_locked()
        if self._idle_timeout_s <= 0 or not self._proc:
            return
        self._idle_generation += 1
        self._idle_close_task = asyncio.create_task(
            self._close_after_idle(self._idle_generation),
            name="hermes-idle-close",
        )

    def _cancel_idle_close_locked(self) -> None:
        if self._idle_close_task and not self._idle_close_task.done():
            if self._idle_close_task is not asyncio.current_task():
                self._idle_close_task.cancel()
        self._idle_close_task = None

    async def _close_after_idle(self, generation: int) -> None:
        try:
            await asyncio.sleep(self._idle_timeout_s)
            async with self._prompt_lock:
                async with self._lifecycle_lock:
                    if generation == self._idle_generation:
                        await self._close_locked(reason="idle-timeout")
        except asyncio.CancelledError:
            return

    async def _close_locked(self, reason: str) -> None:
        self._cancel_idle_close_locked()
        proc = self._proc
        session_id = self._session_id
        self._proc = None
        self._session_id = None
        if not proc:
            return
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        stderr = await proc.stderr.read() if proc.stderr else b""
        if reason == "error" and proc.returncode not in (0, None):
            log.warning("hermes acp exited %s: %s", proc.returncode, stderr.decode("utf-8"))
        elif session_id:
            log.info("hermes session closed (%s, reason=%s)", session_id, reason)

    async def _initialize_session(self, proc: asyncio.subprocess.Process) -> str:
        init_request_id = await self._send_request(
            proc,
            "initialize",
            {
                "protocolVersion": 1,
                "clientInfo": {"name": "nyra", "version": "0.1"},
            },
        )
        await self._await_response(proc, init_request_id)

        new_session_request_id = await self._send_request(
            proc,
            "session/new",
            {
                "cwd": str(Path.cwd()),
                "mcpServers": [],
            },
        )
        new_session_result = await self._await_response(proc, new_session_request_id)
        session_id = _extract_session_id(new_session_result)
        if not session_id:
            raise RuntimeError(f"hermes ACP session/new returned no session id: {new_session_result}")
        log.info("hermes session created (%s)", session_id)
        return session_id

    async def _send_request(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        params: dict,
    ) -> int:
        if not proc.stdin:
            raise RuntimeError("ACP subprocess stdin is unavailable")
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        return self._request_id

    async def _await_response(
        self,
        proc: asyncio.subprocess.Process,
        request_id: int,
    ) -> dict:
        if not proc.stdout:
            raise RuntimeError("ACP subprocess stdout is unavailable")
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise RuntimeError(f"hermes ACP closed while waiting for response {request_id}")
            try:
                message = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                continue
            if message.get("id") != request_id:
                continue
            maybe_error = _extract_error(message)
            if maybe_error:
                raise RuntimeError(maybe_error)
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"hermes ACP response {request_id} missing result payload")
            return result


def _extract_chunk_text(message: dict) -> str | None:
    method = message.get("method")
    if method not in {"session/update", "conversation/update"}:
        return None

    params = message.get("params", {})
    update = params.get("update", params)
    update_type = update.get("type") or update.get("sessionUpdate")
    if update_type not in {"agent_message_chunk", "message_chunk", "chunk"}:
        return None

    content = update.get("content")
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text:
            return text

    for key in ("text_delta", "delta", "text", "content"):
        value = update.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_session_id(result: dict) -> str | None:
    for key in ("sessionId", "session_id"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_error(message: dict) -> str | None:
    error = message.get("error")
    if not isinstance(error, dict):
        return None
    details = error.get("data", {}).get("details") if isinstance(error.get("data"), dict) else None
    message_text = error.get("message")
    if isinstance(details, str) and details:
        return f"hermes ACP error: {message_text} ({details})"
    if isinstance(message_text, str) and message_text:
        return f"hermes ACP error: {message_text}"
    return "hermes ACP error"


def _is_prompt_complete(message: dict, request_id: int) -> bool:
    if message.get("id") != request_id:
        return False
    result = message.get("result")
    return isinstance(result, dict) and result.get("stopReason") in {"end_turn", "stop", "max_tokens"}
