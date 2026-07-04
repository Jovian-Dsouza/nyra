import asyncio
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from hermes_bridge.client import HermesClient, HermesClientError
from hermes_bridge.settings import HermesSettings, is_configured
from nyra_speech import load_max_response_words, truncate_to_word_limit

logger = logging.getLogger(__name__)

TaskStatus = Literal["queued", "running", "completed", "failed", "cancelled", "waiting_approval"]
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_HERMES_STATUSES = frozenset({"queued", "running", "waiting_approval"})

HERMES_RESULTS_CONTEXT_ID = "nyra_hermes_results"
MAX_CONTEXT_TASKS = 5

_SUMMARIZE_PROMPT = """Summarize this background task result for a voice assistant to speak aloud.
Rules:
- At most {max_words} words total
- Plain spoken English, no markdown, bullets, or URLs
- Spell out numbers naturally for speech
- Start with what was accomplished or what went wrong
- Be specific but concise — summary only, not a full report

Task: {label}
User request: {prompt}
Result:
{output}
"""

_LABEL_PROMPT = """Give a short task name (2-5 words, Title Case) for this background job request.
Return ONLY the name — no quotes, punctuation, or extra text.

Request:
{prompt}
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
    def __init__(
        self,
        settings: HermesSettings,
        *,
        room_name: str,
        ui_client=None,
        on_long_running=None,
    ) -> None:
        self._settings = settings
        self._room_name = room_name
        self._session_key = settings.session_key_for_room(room_name)
        self._client = HermesClient(settings, session_key=self._session_key)
        self._tasks: dict[str, HermesTask] = {}
        self._label_to_run_id: dict[str, str] = {}
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._announce_queue: asyncio.Queue[str] = asyncio.Queue()
        self._healthy: bool | None = None
        self._local_queue: list[tuple[str, str]] = []
        self._ui = ui_client
        self._on_long_running = on_long_running
        self._standby_timer: asyncio.Task[None] | None = None

    @property
    def announce_queue(self) -> asyncio.Queue[str]:
        return self._announce_queue

    @property
    def is_available(self) -> bool:
        return is_configured(self._settings) and self._healthy is not False

    def has_active_tasks(self) -> bool:
        if self._local_queue:
            return True
        return any(task.status in _ACTIVE_HERMES_STATUSES for task in self._tasks.values())

    def _cancel_standby_timer(self) -> None:
        if self._standby_timer is not None:
            self._standby_timer.cancel()
            self._standby_timer = None

    def _schedule_standby_timer(self) -> None:
        delay = self._settings.standby_after_seconds
        if delay <= 0 or self._on_long_running is None or not self.has_active_tasks():
            return
        self._cancel_standby_timer()

        async def _wait_then_standby() -> None:
            try:
                await asyncio.sleep(delay)
                if self.has_active_tasks() and self._on_long_running is not None:
                    result = self._on_long_running()
                    if asyncio.iscoroutine(result):
                        await result
            except asyncio.CancelledError:
                pass

        self._standby_timer = asyncio.create_task(
            _wait_then_standby(),
            name="HermesTaskManager._wait_then_standby",
        )

    def _sync_standby_timer(self) -> None:
        if self.has_active_tasks():
            if self._standby_timer is None:
                self._schedule_standby_timer()
        else:
            self._cancel_standby_timer()

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

    def _existing_labels(self) -> set[str]:
        labels = {label.lower() for label in self._label_to_run_id}
        labels.update(label.lower() for label, _ in self._local_queue)
        return labels

    def _unique_label(self, base: str) -> str:
        candidate = base.strip() or "Background Task"
        existing = self._existing_labels()
        if candidate.lower() not in existing:
            return candidate
        suffix = 2
        while True:
            numbered = f"{candidate} ({suffix})"
            if numbered.lower() not in existing:
                return numbered
            suffix += 1

    async def _resolve_task_label(self, prompt: str) -> str:
        base = _derive_label_from_prompt(prompt)
        if self._settings.openai_api_key:
            try:
                llm_name = await self._llm_task_label(_prompt_only(prompt))
                if llm_name:
                    base = llm_name
            except Exception:
                logger.debug("[hermes] LLM task label failed, using heuristic", exc_info=True)
        return self._unique_label(base)

    async def _llm_task_label(self, prompt: str) -> str | None:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._settings.summarize_model,
                    "temperature": 0.2,
                    "max_tokens": 24,
                    "messages": [{"role": "user", "content": _LABEL_PROMPT.format(prompt=prompt)}],
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            cleaned = " ".join(str(content).strip().strip('"\'').split())
            if not cleaned:
                return None
            if len(cleaned) > 48:
                cleaned = cleaned[:47].rstrip() + "…"
            return cleaned

    def _find_label(self, query: str) -> str | None:
        if not query:
            return None

        if query in self._label_to_run_id:
            return query

        lowered = query.lower()
        for label in self._label_to_run_id:
            if label.lower() == lowered:
                return label

        for label, _ in self._local_queue:
            if label.lower() == lowered:
                return label

        partial = [
            label
            for label in self._label_to_run_id
            if lowered in label.lower() or label.lower() in lowered
        ]
        if len(partial) == 1:
            return partial[0]

        if not query.startswith("task-"):
            legacy = f"task-{query}"
            if legacy in self._label_to_run_id:
                return legacy
            for label, _ in self._local_queue:
                if label == legacy:
                    return legacy

        return None

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
        self._sync_standby_timer()

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
            label = await self._resolve_task_label(full_prompt)
            self._local_queue.append((label, full_prompt))
            position = len(self._local_queue)
            return (
                f"Hermes is busy — I've queued this as {label} "
                f"(position {position} in the local queue)."
            )

        return await self._submit_run(full_prompt)

    async def _submit_run(self, full_prompt: str, *, label: str | None = None) -> str:
        label = label or await self._resolve_task_label(full_prompt)
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
                queued_label = await self._resolve_task_label(full_prompt)
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

        label = await self._resolve_task_label(task_prompt)
        name = f"nyra-{label.lower().replace(' ', '-')}"
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
        normalized = self._find_label(label.strip())
        if normalized is None:
            return f"I couldn't find a task called {label.strip()}."

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
            max_words = load_max_response_words() or 50
            return truncate_to_word_limit(
                f"Task {task.label} finished. {raw}",
                max_words=max_words,
            )

        prompt = _SUMMARIZE_PROMPT.format(
            max_words=load_max_response_words() or 50,
            label=task.label,
            prompt=_short_description(task.prompt),
            output=raw,
        )
        max_words = load_max_response_words()
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
                        "max_tokens": max(24, (max_words or 50) * 2),
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                data = response.json()
                task.tokens_used = (data.get("usage") or {}).get("total_tokens")
                content = data["choices"][0]["message"]["content"]
                summary = str(content).strip()
                if max_words > 0:
                    summary = truncate_to_word_limit(summary, max_words)
                return summary
        except Exception:
            logger.warning("[hermes] summarization failed for %s", task.label, exc_info=True)
            max_words = load_max_response_words() or 50
            return truncate_to_word_limit(
                f"Task {task.label} finished. {raw}",
                max_words=max_words,
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
        self._cancel_standby_timer()
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


def _prompt_only(full_prompt: str) -> str:
    return full_prompt.split("\n\nAdditional context:")[0].strip()


def _derive_label_from_prompt(prompt: str, *, max_words: int = 5, max_chars: int = 48) -> str:
    text = " ".join(_prompt_only(prompt).split())
    if not text:
        return "Background Task"

    words = text.split()
    skip = frozenset({
        "please", "can", "you", "could", "would", "will", "want", "need",
        "help", "me", "to", "the", "a", "an", "my", "i", "and", "for",
    })
    while len(words) > 1 and words[0].lower() in skip:
        words.pop(0)

    snippet = " ".join(words[:max_words])
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 1].rstrip() + "…"
    return snippet.title()


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
