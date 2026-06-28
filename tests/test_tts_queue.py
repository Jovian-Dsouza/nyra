import asyncio
import unittest

from orchestrator.tts import TTSQueue


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


if __name__ == "__main__":
    unittest.main()

