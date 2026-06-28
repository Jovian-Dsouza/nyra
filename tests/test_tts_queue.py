import asyncio
import unittest
from unittest.mock import patch

from orchestrator.config import Settings
from orchestrator.tts import TTSQueue
from orchestrator.tts import _resolve_play_command


class TTSQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_order(self) -> None:
        spoken: list[str] = []

        async def fake_speak(text: str) -> None:
            await asyncio.sleep(0.01)
            spoken.append(text)

        queue = TTSQueue(speak_fn=fake_speak)
        await queue.start()
        await queue.enqueue("one")
        await queue.enqueue("two")
        await queue.drain()
        await queue.stop()

        self.assertEqual(spoken, ["one", "two"])

    async def test_interrupt_cancels_current_and_pending(self) -> None:
        spoken: list[str] = []
        block = asyncio.Event()

        async def fake_speak(text: str) -> None:
            spoken.append(text)
            await block.wait()

        queue = TTSQueue(speak_fn=fake_speak)
        await queue.start()
        await queue.enqueue("one")
        await queue.enqueue("two")
        await asyncio.sleep(0.02)
        await queue.interrupt()
        block.set()
        await queue.stop()

        self.assertEqual(spoken, ["one"])


class TTSHelpersTests(unittest.TestCase):
    def test_resolve_play_command_prefers_override(self) -> None:
        settings = Settings(tts_play_command="paplay")
        with patch("orchestrator.tts.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
            self.assertEqual(_resolve_play_command(settings), ["paplay"])

    def test_resolve_play_command_defaults_to_aplay(self) -> None:
        settings = Settings()

        def fake_which(name: str) -> str | None:
            if name == "aplay":
                return "/usr/bin/aplay"
            return None

        with patch("orchestrator.tts.shutil.which", side_effect=fake_which):
            self.assertEqual(_resolve_play_command(settings), ["aplay", "-q"])


if __name__ == "__main__":
    unittest.main()
