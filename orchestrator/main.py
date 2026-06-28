from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from orchestrator.audio import verify_capture_device
from orchestrator.config import Settings
from orchestrator.hermes_client import ACPHermesClient, SentenceAggregator
from orchestrator.state import State, StateMachine
from orchestrator.stt import StreamingSTT
from orchestrator.tts import TTSQueue
from orchestrator.wakeword import WakeWordEngine


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = StateMachine()
        self.wakeword = WakeWordEngine(settings=settings)
        self.stt = StreamingSTT(settings=settings)
        self.hermes = ACPHermesClient(
            settings.hermes_command,
            idle_timeout_s=settings.hermes_session_idle_timeout_s,
        )
        self.tts = TTSQueue(settings=settings)
        self._shutdown = asyncio.Event()
        self._hermes_warmup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        verify_capture_device(required=self.settings.require_audio_device)
        await self.tts.start()
        await self.stt.prepare()
        await self.wakeword.start(self.handle_wake)
        logging.info("orchestrator started in %s", self.state.state.value)

    async def stop(self) -> None:
        await self.wakeword.stop()
        await self.stt.stop()
        await self.tts.stop()
        await self.hermes.close()
        logging.info("orchestrator stopped")

    async def run_forever(self) -> None:
        await self.start()
        await self._shutdown.wait()
        await self.stop()

    async def handle_wake(self) -> None:
        logging.info("wake detected in state=%s", self.state.state.value)
        if self.state.state == State.LISTENING:
            return
        if self.state.state in {State.THINKING, State.SPEAKING}:
            await self.tts.interrupt()
            self.state.set_listening_for_barge_in()
        elif self.state.state == State.IDLE:
            self.state.transition(State.LISTENING, "wake-word")
        self._start_hermes_warmup()
        await self.wakeword.pause_detection()
        await self.stt.start(self._on_partial, self._on_final)

    async def _on_partial(self, text: str) -> None:
        logging.info("stt partial: %s", text)

    async def _on_final(self, text: str) -> None:
        logging.info("stt final: %s", text)
        if not text.strip():
            await self.stt.stop()
            await self.wakeword.resume_detection()
            self.state.transition(State.IDLE, "empty-transcript")
            return
        if self.state.state == State.LISTENING:
            self.state.transition(State.THINKING, "final-transcript")
        await self.stt.stop()
        await self.wakeword.resume_detection()
        try:
            await self._stream_hermes_and_speak(text)
        except Exception:
            logging.exception("hermes/tts pipeline failed")
            if self.state.state != State.LISTENING:
                self.state.transition(State.IDLE, "hermes-error")

    async def _stream_hermes_and_speak(self, prompt: str) -> None:
        logging.info("sending prompt to hermes: %s", prompt)
        aggregator = SentenceAggregator()
        seen_sentence = False
        async for delta in self.hermes.stream(prompt):
            logging.info("hermes delta: %s", delta)
            sentences = aggregator.push(delta)
            for sentence in sentences:
                logging.info("queueing tts sentence: %s", sentence)
                await self.tts.enqueue(sentence)
                if not seen_sentence:
                    self.state.transition(State.SPEAKING, "first-sentence")
                    seen_sentence = True
        tail = aggregator.flush_tail()
        if tail:
            logging.info("queueing tts tail: %s", tail)
            await self.tts.enqueue(tail)
            if not seen_sentence:
                self.state.transition(State.SPEAKING, "tail-sentence")
        await self.tts.drain()
        logging.info("tts queue drained")
        if self.state.state != State.LISTENING:
            self.state.transition(State.IDLE, "response-complete")

    def _start_hermes_warmup(self) -> None:
        if self._hermes_warmup_task and not self._hermes_warmup_task.done():
            return
        self._hermes_warmup_task = asyncio.create_task(
            self._warm_hermes_session(),
            name="hermes-warmup",
        )

    async def _warm_hermes_session(self) -> None:
        try:
            await self.hermes.warmup()
        except Exception:
            logging.exception("hermes warmup failed")

    def request_shutdown(self) -> None:
        self._shutdown.set()


async def _run_console_mode(orchestrator: Orchestrator) -> None:
    print("Dev console mode. Type: wake <prompt>, wake, final <text>, or quit.")
    while True:
        line = await asyncio.to_thread(input, "> ")
        if line.strip() == "quit":
            orchestrator.request_shutdown()
            return
        if line.startswith("wake "):
            await orchestrator.handle_wake()
            await orchestrator.stt.submit_final(line[5:].strip())
            continue
        if line.strip() == "wake":
            await orchestrator.handle_wake()
            continue
        if line.startswith("final "):
            await orchestrator.stt.submit_final(line[6:].strip())


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _main_async(dev_console_override: bool | None = None) -> int:
    settings = Settings.from_env()
    _configure_logging(settings.log_level)
    orchestrator = Orchestrator(settings=settings)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, orchestrator.request_shutdown)

    if dev_console_override is True:
        await orchestrator.start()
        try:
            await _run_console_mode(orchestrator)
        finally:
            await orchestrator.stop()
        return 0

    await orchestrator.run_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Nyra voice orchestrator")
    parser.add_argument("--dev-console", action="store_true", help="manual wake/final transcript testing")
    args = parser.parse_args()
    return asyncio.run(_main_async(dev_console_override=args.dev_console))


if __name__ == "__main__":
    raise SystemExit(main())
