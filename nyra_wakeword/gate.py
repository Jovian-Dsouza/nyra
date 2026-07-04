from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents.voice.io import AudioInput

if TYPE_CHECKING:
    from nyra_wakeword.controller import WakeWordController
    from nyra_wakeword.detector import WakeWordDetector


class WakeWordGateInput(AudioInput):
    """Tee room audio through openWakeWord; gate STT when passive."""

    def __init__(
        self,
        source: AudioInput,
        detector: WakeWordDetector,
        controller: WakeWordController,
    ) -> None:
        # Do not pass source into AudioInput — the base on_attached() recurses.
        super().__init__(label="wakeword_gate", source=None)
        self._source = source
        self._detector = detector
        self._controller = controller
        self._forward_enabled = True
        self._lock = asyncio.Lock()

    def on_attached(self) -> None:
        self._source.on_attached()

    def on_detached(self) -> None:
        self._source.on_detached()

    @property
    def forward_enabled(self) -> bool:
        return self._forward_enabled

    def set_forward_enabled(self, enabled: bool) -> None:
        self._forward_enabled = enabled

    async def __anext__(self) -> rtc.AudioFrame:
        while True:
            frame = await self._source.__anext__()
            detection = await self._detector.process_frame(frame)
            if detection is not None:
                async with self._lock:
                    await self._controller.on_wake_detected(detection[0], detection[1])
            if self._forward_enabled:
                return frame
