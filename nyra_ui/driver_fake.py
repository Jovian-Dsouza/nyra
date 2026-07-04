"""Scripted, looping event sequence for visual iteration on the status window.

Not a substitute for real backend signals — see `nyra_agent.py` and
`hermes_bridge/tasks.py` for where production events actually come from.
This exists only so the layout can be iterated on without running the full
voice agent.
"""

import threading
import time

from nyra_ui.bridge import UIStateStore
from nyra_ui.protocol import (
    goodbye_event,
    hello_event,
    hermes_tasks_event,
    llm_event,
    memory_event,
    phase_event,
    stt_event,
)

DEMO_STEP_SECONDS = 2.0


def _build_script() -> list[dict]:
    return [
        phase_event("idle"),
        hello_event("demo-room"),
        phase_event("listening"),
        stt_event("What's the weather like tomorrow", False),
        stt_event("What's the weather like tomorrow?", True),
        phase_event("thinking"),
        memory_event("recalling"),
        memory_event("done", match_count=2),
        hermes_tasks_event(
            [{"label": "task-1", "status": "running", "elapsed_seconds": 4, "tokens_used": None}]
        ),
        phase_event("speaking"),
        llm_event("Tomorrow looks sunny with a high of seventy five."),
        hermes_tasks_event(
            [{"label": "task-1", "status": "completed", "elapsed_seconds": 12, "tokens_used": 842}]
        ),
        phase_event("idle"),
        goodbye_event(),
    ]


class FakeStateDriver:
    def __init__(self, store: UIStateStore, *, step_seconds: float = DEMO_STEP_SECONDS) -> None:
        self._store = store
        self._step_seconds = step_seconds
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="nyra-ui-fake-driver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        script = _build_script()
        while not self._stop.is_set():
            for event in script:
                if self._stop.is_set():
                    return
                self._store.apply(event)
                time.sleep(self._step_seconds)
