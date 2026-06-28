import asyncio
import json
import unittest
from unittest.mock import patch

from orchestrator.hermes_client import (
    ACPHermesClient,
    _extract_chunk_text,
    _extract_error,
    _extract_session_id,
    _is_prompt_complete,
)


class HermesClientHelpersTests(unittest.TestCase):
    def test_extract_session_id_prefers_camel_case(self) -> None:
        self.assertEqual(_extract_session_id({"sessionId": "abc"}), "abc")
        self.assertEqual(_extract_session_id({"session_id": "abc"}), "abc")
        self.assertIsNone(_extract_session_id({}))

    def test_extract_error_includes_details(self) -> None:
        message = {
            "error": {
                "message": "Internal error",
                "data": {"details": "disk full"},
            }
        }
        self.assertEqual(_extract_error(message), "hermes ACP error: Internal error (disk full)")

    def test_extract_chunk_text_supports_session_update(self) -> None:
        message = {
            "method": "session/update",
            "params": {
                "update": {
                    "type": "agent_message_chunk",
                    "content": "Hello",
                }
            },
        }
        self.assertEqual(_extract_chunk_text(message), "Hello")

    def test_extract_chunk_text_supports_acp_aliases(self) -> None:
        message = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Hi"},
                }
            },
        }
        self.assertEqual(_extract_chunk_text(message), "Hi")

    def test_prompt_completion_detects_stop_reason(self) -> None:
        message = {
            "id": 7,
            "result": {"stopReason": "end_turn"},
        }
        self.assertTrue(_is_prompt_complete(message, 7))
        self.assertFalse(_is_prompt_complete(message, 8))


class HermesClientSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_reuses_process_and_session_for_prompts(self) -> None:
        factory = _FakeACPFactory()

        async def fake_create_subprocess_exec(*args, **kwargs):
            return factory.create_process()

        with patch(
            "orchestrator.hermes_client.asyncio.create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ):
            client = ACPHermesClient("hermes acp", idle_timeout_s=60)
            await client.warmup()
            first = await _collect_stream(client, "first")
            second = await _collect_stream(client, "second")
            await client.close()

        self.assertEqual(first, "response.")
        self.assertEqual(second, "response.")
        self.assertEqual(len(factory.processes), 1)
        self.assertEqual(factory.session_new_count, 1)
        self.assertEqual(factory.prompt_session_ids, ["s1", "s1"])

    async def test_idle_timeout_closes_session_and_next_prompt_reopens(self) -> None:
        factory = _FakeACPFactory()

        async def fake_create_subprocess_exec(*args, **kwargs):
            return factory.create_process()

        with patch(
            "orchestrator.hermes_client.asyncio.create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ):
            client = ACPHermesClient("hermes acp", idle_timeout_s=0.01)
            await client.warmup()
            await asyncio.sleep(0.05)
            text = await _collect_stream(client, "after idle")
            await client.close()

        self.assertEqual(text, "response.")
        self.assertEqual(len(factory.processes), 2)
        self.assertEqual(factory.session_new_count, 2)
        self.assertIsNotNone(factory.processes[0].returncode)
        self.assertEqual(factory.prompt_session_ids, ["s2"])


async def _collect_stream(client: ACPHermesClient, prompt: str) -> str:
    chunks: list[str] = []
    async for chunk in client.stream(prompt):
        chunks.append(chunk)
    return "".join(chunks)


class _FakeACPFactory:
    def __init__(self) -> None:
        self.processes: list[_FakeACPProcess] = []
        self.session_new_count = 0
        self.prompt_session_ids: list[str] = []

    def create_process(self) -> "_FakeACPProcess":
        proc = _FakeACPProcess(self)
        self.processes.append(proc)
        return proc

    def next_session_id(self) -> str:
        self.session_new_count += 1
        return f"s{self.session_new_count}"


class _FakeACPProcess:
    def __init__(self, factory: _FakeACPFactory) -> None:
        self.factory = factory
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr()
        self.stdin = _FakeStdin(self)
        self.returncode: int | None = None

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def handle_request(self, request: dict) -> None:
        method = request["method"]
        request_id = request["id"]
        if method == "initialize":
            self.stdout.put({"id": request_id, "result": {}})
            return
        if method == "session/new":
            self.stdout.put(
                {
                    "id": request_id,
                    "result": {"sessionId": self.factory.next_session_id()},
                }
            )
            return
        if method == "session/prompt":
            self.factory.prompt_session_ids.append(request["params"]["sessionId"])
            self.stdout.put(
                {
                    "method": "session/update",
                    "params": {
                        "update": {
                            "type": "agent_message_chunk",
                            "content": {"type": "text", "text": "response."},
                        }
                    },
                }
            )
            self.stdout.put({"id": request_id, "result": {"stopReason": "end_turn"}})
            return
        raise AssertionError(f"unexpected ACP method: {method}")


class _FakeStdin:
    def __init__(self, proc: _FakeACPProcess) -> None:
        self.proc = proc

    def write(self, data: bytes) -> None:
        for raw_line in data.decode("utf-8").splitlines():
            self.proc.handle_request(json.loads(raw_line))

    async def drain(self) -> None:
        await asyncio.sleep(0)


class _FakeStdout:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def put(self, message: dict) -> None:
        self._queue.put_nowait((json.dumps(message) + "\n").encode("utf-8"))

    async def readline(self) -> bytes:
        return await self._queue.get()


class _FakeStderr:
    async def read(self) -> bytes:
        return b""


if __name__ == "__main__":
    unittest.main()
