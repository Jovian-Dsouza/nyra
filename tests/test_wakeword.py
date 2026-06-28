import asyncio
import unittest
from unittest.mock import patch

from orchestrator.config import Settings
from orchestrator.wakeword import WakeWordEngine


class WakeWordEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_trigger_invokes_callback_when_backend_unavailable(self) -> None:
        calls: list[str] = []
        settings = Settings(require_audio_device=False)
        engine = WakeWordEngine(settings=settings)

        async def on_wake() -> None:
            calls.append("wake")

        with patch("orchestrator.wakeword._OpenWakeWordDetector.try_create", return_value=None):
            await engine.start(on_wake)
            await engine.trigger_now()
            await asyncio.sleep(0.05)
            await engine.stop()

        self.assertEqual(calls, ["wake"])


if __name__ == "__main__":
    unittest.main()

