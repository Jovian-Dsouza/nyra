from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import shlex
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from orchestrator.config import Settings


log = logging.getLogger(__name__)

SpeakFn = Callable[[str], Awaitable[None]]


async def default_speak(text: str) -> None:
    """Fallback speech path.

    Prefer a local speech CLI when available; otherwise simulate
    synthesis/playback latency so orchestration behavior stays testable.
    """

    for cmd in (["espeak-ng", text], ["espeak", text], ["spd-say", text]):
        if shutil.which(cmd[0]):
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()
            return
    await asyncio.sleep(min(0.2 + len(text) * 0.01, 2.0))


class PiperSpeakFn:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._piper_cmd = shlex.split(settings.piper_command)
        self._play_cmd = _resolve_play_command(settings)
        self._backend_logged = False
        self._fallback_warned = False

    async def __call__(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if self._can_use_piper():
            if not self._backend_logged:
                self._backend_logged = True
                log.info(
                    "tts backend: piper (voice=%s, playback=%s)",
                    self._settings.piper_voice_path,
                    " ".join(self._play_cmd),
                )
            await self._speak_with_piper(text)
            return

        if not self._fallback_warned:
            self._fallback_warned = True
            log.warning(
                "piper TTS unavailable (command=%s, voice_exists=%s, playback=%s); using fallback speech path",
                self._settings.piper_command,
                self._settings.piper_voice_path.exists(),
                " ".join(self._play_cmd) if self._play_cmd else "none",
            )
        await default_speak(text)

    def _can_use_piper(self) -> bool:
        return bool(self._play_cmd) and _command_exists(self._piper_cmd) and self._settings.piper_voice_path.exists()

    async def _speak_with_piper(self, text: str) -> None:
        fd, wav_path_raw = tempfile.mkstemp(suffix=".wav", prefix="nyra-tts-", dir="/tmp")
        os.close(fd)
        wav_path = Path(wav_path_raw)
        try:
            await _run_subprocess(
                [
                    *self._piper_cmd,
                    "--model",
                    str(self._settings.piper_voice_path),
                    "--output_file",
                    str(wav_path),
                ],
                input_text=text,
                name="piper",
            )
            await _run_subprocess([*self._play_cmd, str(wav_path)], name=self._play_cmd[0])
        finally:
            wav_path.unlink(missing_ok=True)


class SarvamSpeakFn:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._play_cmd = _resolve_play_command(settings)
        self._backend_logged = False
        self._fallback_warned = False

    async def __call__(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if self._can_use_sarvam():
            if not self._backend_logged:
                self._backend_logged = True
                log.info(
                    "tts backend: sarvam (language=%s, speaker=%s, playback=%s)",
                    self._settings.sarvam_tts_language,
                    self._settings.sarvam_tts_speaker,
                    " ".join(self._play_cmd),
                )
            try:
                await self._speak_with_sarvam(text)
                return
            except Exception:
                log.exception("sarvam TTS failed; using fallback speech path")
                await default_speak(text)
                return

        if not self._fallback_warned:
            self._fallback_warned = True
            log.warning(
                "sarvam TTS unavailable (api_key=%s, playback=%s); using fallback speech path",
                "set" if self._settings.sarvam_api_key else "missing",
                " ".join(self._play_cmd) if self._play_cmd else "none",
            )
        await default_speak(text)

    def _can_use_sarvam(self) -> bool:
        return bool(self._play_cmd) and bool(self._settings.sarvam_api_key)

    async def _speak_with_sarvam(self, text: str) -> None:
        wav_bytes = await asyncio.to_thread(self._synthesize, text)
        fd, wav_path_raw = tempfile.mkstemp(suffix=".wav", prefix="nyra-tts-", dir="/tmp")
        os.close(fd)
        wav_path = Path(wav_path_raw)
        try:
            wav_path.write_bytes(wav_bytes)
            await _run_subprocess([*self._play_cmd, str(wav_path)], name=self._play_cmd[0])
        finally:
            wav_path.unlink(missing_ok=True)

    def _synthesize(self, text: str) -> bytes:
        try:
            from sarvamai import SarvamAI
        except ImportError as exc:
            raise RuntimeError("sarvamai package is not installed") from exc

        client = SarvamAI(api_subscription_key=self._settings.sarvam_api_key)
        response = client.text_to_speech.convert(
            text=text,
            target_language_code=self._settings.sarvam_tts_language,
            model="bulbul:v3",
            speaker=self._settings.sarvam_tts_speaker,
        )
        if not response.audios:
            raise RuntimeError("sarvam TTS returned no audio")
        return base64.b64decode(response.audios[0])


def _build_speak_fn(settings: Settings) -> SpeakFn:
    if settings.tts_backend == "sarvam":
        return SarvamSpeakFn(settings)
    return PiperSpeakFn(settings)


class TTSQueue:
    def __init__(self, settings: Settings | None = None, speak_fn: SpeakFn | None = None) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        resolved_settings = settings or Settings()
        self._speak_fn = speak_fn or _build_speak_fn(resolved_settings)
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


def _resolve_play_command(settings: Settings) -> list[str] | None:
    if settings.tts_play_command:
        play_cmd = shlex.split(settings.tts_play_command)
        return play_cmd if _command_exists(play_cmd) else None
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    if shutil.which("paplay"):
        return ["paplay"]
    return None


def _command_exists(cmd: list[str]) -> bool:
    if not cmd:
        return False
    binary = cmd[0]
    return bool(shutil.which(binary) or Path(binary).exists())


async def _run_subprocess(cmd: list[str], *, name: str, input_text: str | None = None) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if input_text is not None and proc.stdin is not None:
            proc.stdin.write((input_text + "\n").encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        returncode = await proc.wait()
    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
        raise

    if returncode == 0:
        return

    stderr = await proc.stderr.read() if proc.stderr else b""
    raise RuntimeError(f"{name} failed with code {returncode}: {stderr.decode('utf-8', errors='ignore').strip()}")
