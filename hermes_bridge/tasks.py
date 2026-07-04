import asyncio
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from hermes_bridge.client import HermesClient, HermesClientError
from hermes_bridge.settings import HermesSettings, is_configured

logger = logging.getLogger(__name__)

TaskStatus = Literal["queued", "running", "completed", "failed", "cancelled", "waiting_approval"]
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

HERMES_RESULTS_CONTEXT_ID = "nyra_hermes_results"
MAX_CONTEXT_TASKS = 5

_SUMMARIZE_PROMPT = """Summarize this background task result for a voice assistant to speak aloud.
Rules:
- 2 to 4 short sentences maximum
- Plain spoken English, no markdown, bullets, or URLs
- Spell out numbers naturally for speech
- Start with what was accomplished or what went wrong
- Be specific but concise

Task: {label}
User request: {prompt}
Result:
{output}
"""


@dataclass
class HermesTask:
    run_id: str
    label: str
    prompt: str
    status: TaskStatus = "queued"
    submitted_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    raw_output: str | None = None
    voice_summary: str | None = None
    error_message: str | None = None
    announced: bool = False
    is_scheduled: bool = False
    tokens_used: int | None = None


class HermesTaskManager:
    def __init__(self, settings: HermesSettings, *, room_name: str, ui_client=None) -> None:
        self._settings = settings
        self._room_name = room_name
        self._session_key = settings.session_key_for_room(room_name)
        self._client = HermesClient(settings, session_key=self._session_key)
        self._tasks: dict[str, HermesTask] = {}
        self._label_to_run_id: dict[str, str] = {}
        self._next_label_num = 0
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._announce_queue: asyncio.Queue[str] = asyncio.Queue()
        self._healthy: bool | None = None
        self._local_queue: list[tuple[str, str]] = []
        self._ui = ui_client

    @property
    def announce_queue(self) -> asyncio.Queue[str]:
        return self._announce_queue

    @property
    def is_available(self) -> bool:
        return is_configured(self._settings) and self._healthy is not False

    async def startup(self) -> None:
        if not is_configured(self._settings):
            logger.warning("[hermes] HERMES_API_KEY not set — delegation disabled")
            self._healthy = False
            return
        self._healthy = await self._client.health_check()
        if self._healthy:
            logger.info("[hermes] gateway healthy at %s", self._settings.api_url)
        else:
            logger.warning("[hermes] gateway unreachable at %s", self._settings.api_url)

    def _next_label(self) -> str:
        self._next_label_num += 1
        return f"task-{self._next_label_num}"

    def _active_count(self) -> int:
        return sum(
            1
            for t in self._tasks.values()
            if t.status in ("queued", "running", "waiting_approval")
        )

    def _register_task(self, task: HermesTask) -> None:
        self._tasks[task.run_id] = task
        self._label_to_run_id[task.label] = task.run_id
        self._publish_snapshot()

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "label": task.label,
                "status": task.status,
                "elapsed_seconds": time.monotonic() - task.submitted_at,
                "tokens_used": task.tokens_used,
            }
            for task in sorted(self._tasks.values(), key=lambda t: t.submitted_at)
        ]

    def _publish_snapshot(self) -> None:
        if self._ui is not None:
            self._ui.publish_hermes_tasks(self.snapshot())

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def delegate(
        self,
        task_prompt: str,
        *,
        context: str = "",
    ) -> str:
        if not is_configured(self._settings):
            return "Hermes is not configured. Set HERMES_API_KEY to enable background tasks."
        if self._healthy is False:
            return "I can't reach the Hermes gateway right now. Make sure hermes gateway is running."

        full_prompt = task_prompt
        if context.strip():
            full_prompt = f"{task_prompt}\n\nAdditional context: {context.strip()}"

        if self._active_count() >= self._settings.max_concurrent:
            label = self._next_label()
            self._local_queue.append((label, full_prompt))
            position = len(self._local_queue)
            return (
                f"Hermes is busy — I've queued this as {label} "
                f"(position {position} in the local queue)."
            )

        return await self._submit_run(full_prompt)

    async def _submit_run(self, full_prompt: str, *, label: str | None = None) -> str:
        label = label or self._next_label()
        short_desc = _short_description(full_prompt)

        try:
            result = await self._client.submit_run(
                full_prompt,
                session_id=f"nyra-{self._room_name}",
                instructions=(
                    "You are working on a background task delegated from Nyra, "
                    "a real-time voice assistant. Be thorough but return a clear, "
                    "complete answer the voice agent can summarize for the user."
                ),
            )
        except HermesClientError as exc:
            if exc.status_code == 429:
                queued_label = self._next_label()
                self._local_queue.append((queued_label, full_prompt))
                return (
                    f"Hermes is at capacity — I've queued this as {queued_label} "
                    f"(position {len(self._local_queue)})."
                )
            return f"I couldn't send that to Hermes: {exc}"

        task = HermesTask(
            run_id=result.run_id,
            label=label,
            prompt=full_prompt,
            status="queued",
        )
        self._register_task(task)
        self._spawn(self._watch_run(result.run_id))
        self._spawn(self._drain_local_queue())
        logger.info("[hermes] submitted %s as %s", result.run_id, label)
        return f"Queued as {label}: {short_desc}"

    async def _drain_local_queue(self) -> None:
        while self._local_queue and self._active_count() < self._settings.max_concurrent:
            label, prompt = self._local_queue.pop(0)
            await self._submit_run(prompt, label=label)

    async def schedule(self, task_prompt: str, schedule: str) -> str:
        if not is_configured(self._settings):
            return "Hermes is not configured. Set HERMES_API_KEY to enable scheduled tasks."
        if self._healthy is False:
            return "I can't reach the Hermes gateway right now."

        label = self._next_label()
        name = f"nyra-{label}"
        try:
            result = await self._client.create_scheduled_job(
                name=name,
                schedule=schedule,
                prompt=task_prompt,
            )
        except HermesClientError as exc:
            return f"I couldn't schedule that task: {exc}"

        task = HermesTask(
            run_id=result.job_id or name,
            label=label,
            prompt=task_prompt,
            status="queued",
            is_scheduled=True,
        )
        self._register_task(task)
        short_desc = _short_description(task_prompt)
        return f"Scheduled as {label} ({schedule}): {short_desc}"

    def list_tasks(self) -> str:
        if not self._tasks and not self._local_queue:
            return "No background tasks right now."

        lines: list[str] = []
        for label, prompt in self._local_queue:
            short = _short_description(prompt)
            lines.append(f"- {label}: queued locally — {short}")

        for task in sorted(self._tasks.values(), key=lambda t: t.submitted_at):
            elapsed = _format_elapsed(time.monotonic() - task.submitted_at)
            if task.status in TERMINAL_STATUSES:
                if task.announced:
                    continue
                status_note = task.status
                if task.voice_summary:
                    status_note = f"{task.status}, ready to announce"
            else:
                status_note = f"{task.status}, running {elapsed}"
            short = _short_description(task.prompt)
            kind = "scheduled" if task.is_scheduled else "background"
            lines.append(f"- {task.label} ({kind}): {status_note} — {short}")

        if not lines:
            return "No active background tasks. Recently completed tasks were already announced."
        return "Background tasks:\n" + "\n".join(lines)

    async def cancel(self, label: str) -> str:
        normalized = label.strip().lower()
        if not normalized.startswith("task-"):
            normalized = f"task-{normalized}"

        run_id = self._label_to_run_id.get(normalized)
        if run_id is None:
            for i, (queued_label, _) in enumerate(self._local_queue):
                if queued_label == normalized:
                    self._local_queue.pop(i)
                    return f"Removed {normalized} from the local queue."
            return f"I couldn't find a task called {normalized}."

        task = self._tasks.get(run_id)
        if task is None:
            return f"I couldn't find a task called {normalized}."

        if task.status in TERMINAL_STATUSES:
            return f"{normalized} already finished ({task.status})."

        if task.is_scheduled:
            return (
                f"{normalized} is a scheduled job — cancel it from the Hermes dashboard "
                "or ask me to help remove it there."
            )

        try:
            await self._client.stop_run(run_id)
        except HermesClientError as exc:
            return f"I couldn't cancel {normalized}: {exc}"

        task.status = "cancelled"
        task.completed_at = time.monotonic()
        task.voice_summary = f"Task {normalized} was cancelled."
        self._publish_snapshot()
        return f"Cancelled {normalized}."

    async def _watch_run(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is None:
            return

        try:
            while True:
                await asyncio.sleep(self._settings.poll_interval)
                try:
                    status = await self._client.get_run(run_id)
                except HermesClientError:
                    logger.warning("[hermes] poll failed for %s", run_id, exc_info=True)
                    continue

                mapped = _map_status(status.status)
                task.status = mapped
                self._publish_snapshot()

                if mapped == "waiting_approval":
                    task.voice_summary = (
                        f"Task {task.label} is waiting for approval in the Hermes dashboard."
                    )
                    task.completed_at = time.monotonic()
                    await self._finalize_task(task)
                    return

                if mapped not in TERMINAL_STATUSES:
                    continue

                task.completed_at = time.monotonic()
                if mapped == "completed":
                    raw = status.output or ""
                    if len(raw) > self._settings.result_max_chars:
                        raw = raw[: self._settings.result_max_chars] + "…"
                    task.raw_output = raw
                    task.voice_summary = await self._summarize_for_voice(task)
                elif mapped == "failed":
                    task.error_message = status.error or "Unknown error"
                    task.voice_summary = (
                        f"Task {task.label} failed. {task.error_message}"
                    )
                elif mapped == "cancelled":
                    task.voice_summary = f"Task {task.label} was cancelled."

                await self._finalize_task(task)
                return
        except asyncio.CancelledError:
            return

    async def _finalize_task(self, task: HermesTask) -> None:
        if task.voice_summary:
            await self._announce_queue.put(task.voice_summary)
        self._spawn(self._drain_local_queue())
        self._publish_snapshot()
        logger.info("[hermes] %s finished: %s", task.label, task.status)

    async def _summarize_for_voice(self, task: HermesTask) -> str:
        raw = task.raw_output or ""
        if not raw.strip():
            return f"Task {task.label} completed but returned no output."

        if not self._settings.openai_api_key:
            return _truncate_for_speech(
                f"Task {task.label} finished. {raw}",
                max_chars=400,
            )

        prompt = _SUMMARIZE_PROMPT.format(
            label=task.label,
            prompt=_short_description(task.prompt),
            output=raw,
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._settings.summarize_model,
                        "temperature": 0.3,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                data = response.json()
                task.tokens_used = (data.get("usage") or {}).get("total_tokens")
                content = data["choices"][0]["message"]["content"]
                return str(content).strip()
        except Exception:
            logger.warning("[hermes] summarization failed for %s", task.label, exc_info=True)
            return _truncate_for_speech(
                f"Task {task.label} finished. {raw}",
                max_chars=400,
            )

    def get_results_context(self) -> str | None:
        completed = [
            t
            for t in self._tasks.values()
            if t.status == "completed" and t.voice_summary
        ]
        if not completed:
            return None

        completed.sort(key=lambda t: t.completed_at or t.submitted_at, reverse=True)
        lines = []
        for task in completed[:MAX_CONTEXT_TASKS]:
            lines.append(f"- {task.label}: {task.voice_summary}")
        return "[Background task results from Hermes]\n" + "\n".join(lines)

    def mark_announced(self, summary: str) -> None:
        for task in self._tasks.values():
            if task.voice_summary == summary and not task.announced:
                task.announced = True
                return
        for task in self._tasks.values():
            if task.voice_summary and summary in task.voice_summary and not task.announced:
                task.announced = True

    async def shutdown(self) -> None:
        for bg in list(self._bg_tasks):
            bg.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        await self._client.close()


def upsert_hermes_results_message(chat_ctx, results_text: str) -> None:
    from livekit.agents import llm

    existing = chat_ctx.get_by_id(HERMES_RESULTS_CONTEXT_ID)
    if existing is not None and existing.type == "message":
        idx = chat_ctx.index_by_id(HERMES_RESULTS_CONTEXT_ID)
        if idx is not None:
            chat_ctx.items[idx] = llm.ChatMessage(
                role="system",
                content=[results_text],
                id=HERMES_RESULTS_CONTEXT_ID,
            )
            return
    chat_ctx.add_message(role="system", content=results_text, id=HERMES_RESULTS_CONTEXT_ID)


def _map_status(hermes_status: str) -> TaskStatus:
    normalized = hermes_status.lower()
    if normalized in ("queued", "started"):
        return "queued"
    if normalized == "running":
        return "running"
    if normalized == "completed":
        return "completed"
    if normalized in ("failed", "error"):
        return "failed"
    if normalized in ("cancelled", "canceled", "stopped"):
        return "cancelled"
    if "approval" in normalized:
        return "waiting_approval"
    return "running"


def _short_description(prompt: str, max_len: int = 60) -> str:
    text = " ".join(prompt.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _truncate_for_speech(text: str, *, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
