from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections.abc import AsyncIterator


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

    def __init__(self, hermes_command: str) -> None:
        self._command = hermes_command
        self._request_id = 0

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        cmd = shlex.split(self._command)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if not proc.stdin or not proc.stdout:
            raise RuntimeError("failed to start ACP subprocess with stdio pipes")

        self._request_id += 1
        start_req = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "session/start",
            "params": {"input": prompt},
        }
        proc.stdin.write((json.dumps(start_req) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    continue

                maybe_text = _extract_chunk_text(message)
                if maybe_text:
                    yield maybe_text

                if _is_terminal_message(message):
                    break
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            stderr = await proc.stderr.read() if proc.stderr else b""
            if proc.returncode not in (0, None):
                log.warning("hermes acp exited %s: %s", proc.returncode, stderr.decode("utf-8"))


def _extract_chunk_text(message: dict) -> str | None:
    method = message.get("method")
    if method not in {"session/update", "conversation/update"}:
        return None

    params = message.get("params", {})
    update = params.get("update", params)
    update_type = update.get("type")
    if update_type not in {"agent_message_chunk", "message_chunk", "chunk"}:
        return None

    for key in ("text_delta", "delta", "text", "content"):
        value = update.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _is_terminal_message(message: dict) -> bool:
    if message.get("method") in {"session/complete", "session/completed", "conversation/completed"}:
        return True
    result = message.get("result")
    if isinstance(result, dict) and result.get("status") in {"completed", "done"}:
        return True
    return False

