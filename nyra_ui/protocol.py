"""Wire format for agent -> UI status events.

Newline-delimited JSON, one object per line, UTF-8. Every message has a
`type` discriminator, matching the event-model convention livekit-agents
already uses (`UserInputTranscribedEvent.type`, etc.).

This module is the only place that knows how a raw JSON dict turns into a
`UIState` transition, so the socket-handling code in `bridge.py` stays thin.
"""

import json
import time
from typing import Any

from nyra_ui.state import AGENT_STATE_TO_PHASE, HermesTaskView, UIPhase, UIState


def encode_event(event: dict[str, Any]) -> bytes:
    event.setdefault("ts", time.time())
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


def phase_event(agent_state: str) -> dict[str, Any]:
    return {"type": "phase", "agent_state": agent_state}


def hello_event(room_name: str | None) -> dict[str, Any]:
    return {"type": "hello", "room_name": room_name}


def goodbye_event() -> dict[str, Any]:
    return {"type": "goodbye"}


def stt_event(text: str, is_final: bool) -> dict[str, Any]:
    return {"type": "stt", "text": text, "is_final": is_final}


def llm_event(text: str, is_final: bool = True) -> dict[str, Any]:
    return {"type": "llm", "text": text, "is_final": is_final}


def memory_event(status: str, match_count: int | None = None) -> dict[str, Any]:
    return {"type": "memory", "status": status, "match_count": match_count}


def hermes_tasks_event(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "hermes_tasks", "tasks": tasks}


def apply_event(state: UIState, event: dict[str, Any]) -> UIState:
    """Fold one decoded event into the previous UIState. Unknown or malformed
    events are ignored — a slow/confused agent connection must never crash
    the render loop."""
    event_type = event.get("type")

    if event_type == "phase":
        phase = AGENT_STATE_TO_PHASE.get(event.get("agent_state", "idle"), UIPhase.IDLE)
        return state.with_phase(phase)

    if event_type == "hello":
        return state.with_attached(True)

    if event_type == "goodbye":
        return state.with_attached(False)

    if event_type == "stt":
        # `event.get(key, default)` only falls back when the key is absent —
        # an explicit JSON null still comes through as None, so coalesce with
        # `or` too, otherwise str(None) would render the literal word "None".
        return state.with_transcript(str(event.get("text") or ""), bool(event.get("is_final", False)))

    if event_type == "llm":
        return state.with_llm_output(str(event.get("text") or ""), bool(event.get("is_final", True)))

    if event_type == "memory":
        return state.with_memory_status(
            str(event.get("status") or "idle"), event.get("match_count")
        )

    if event_type == "hermes_tasks":
        tasks = tuple(
            HermesTaskView(
                label=str(t.get("label", "")),
                status=str(t.get("status", "")),
                elapsed_seconds=float(t.get("elapsed_seconds", 0.0)),
                tokens_used=t.get("tokens_used"),
            )
            for t in event.get("tasks", [])
        )
        return state.with_hermes_tasks(tasks)

    return state
