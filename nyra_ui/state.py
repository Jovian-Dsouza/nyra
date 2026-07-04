"""Immutable UI state for the Nyra status window.

`UIPhase` maps 1:1 onto real backend signals the agent already emits — no
fabricated wake-word, planning, or delegation phases. See `AGENT_STATE_TO_PHASE`
for the mapping from LiveKit's `AgentState` to `UIPhase`. `TOOL_RUNNING` and
`ERROR` are reached only via explicit `with_tool`/`with_error` calls, since
`agent_state_changed` never emits those states itself.
"""

import time
from dataclasses import dataclass, field, replace
from enum import Enum


class UIPhase(Enum):
    IDLE = "idle"
    STANDBY = "standby"
    LISTENING = "listening"
    THINKING = "thinking"
    TOOL_RUNNING = "tool_running"
    SPEAKING = "speaking"
    ERROR = "error"


AGENT_STATE_TO_PHASE = {
    "initializing": UIPhase.IDLE,
    "idle": UIPhase.IDLE,
    "standby": UIPhase.STANDBY,
    "listening": UIPhase.LISTENING,
    "thinking": UIPhase.THINKING,
    "speaking": UIPhase.SPEAKING,
}

_PHASES_THAT_CLEAR_TRANSCRIPT = frozenset({UIPhase.IDLE, UIPhase.STANDBY, UIPhase.SPEAKING})
_PHASES_THAT_CLEAR_LLM_TEXT = frozenset({UIPhase.IDLE, UIPhase.STANDBY})

_ACTIVE_HERMES_STATUSES = frozenset({"queued", "running", "waiting_approval"})


def _has_active_hermes_tasks(state: "UIState") -> bool:
    return any(task.status in _ACTIVE_HERMES_STATUSES for task in state.hermes_tasks)


@dataclass(frozen=True)
class HermesTaskView:
    label: str
    status: str
    elapsed_seconds: float
    tokens_used: int | None = None


@dataclass(frozen=True)
class UIState:
    phase: UIPhase
    updated_at: float
    partial_transcript: str = ""
    is_final_transcript: bool = False
    llm_text: str = ""
    is_final_llm: bool = True
    memory_status: str = "idle"
    memory_match_count: int | None = None
    hermes_tasks: tuple[HermesTaskView, ...] = field(default_factory=tuple)
    tool_name: str | None = None
    error_message: str | None = None
    attached: bool = False
    """Whether an agent job process is currently connected to the UI socket."""

    def __post_init__(self) -> None:
        if self.tool_name is not None and self.phase is not UIPhase.TOOL_RUNNING:
            raise ValueError("tool_name may only be set when phase is TOOL_RUNNING")
        if self.error_message is not None and self.phase is not UIPhase.ERROR:
            raise ValueError("error_message may only be set when phase is ERROR")
        if self.memory_match_count is not None and self.memory_status != "done":
            raise ValueError("memory_match_count may only be set when memory_status is 'done'")

    @staticmethod
    def initial() -> "UIState":
        return UIState(phase=UIPhase.IDLE, updated_at=time.monotonic())

    def with_phase(self, phase: UIPhase, *, at: float | None = None) -> "UIState":
        keep_transcript = phase not in _PHASES_THAT_CLEAR_TRANSCRIPT
        keep_llm_text = phase not in _PHASES_THAT_CLEAR_LLM_TEXT
        if phase is UIPhase.LISTENING and _has_active_hermes_tasks(self):
            keep_llm_text = True
        return replace(
            self,
            phase=phase,
            updated_at=at if at is not None else time.monotonic(),
            partial_transcript=self.partial_transcript if keep_transcript else "",
            is_final_transcript=self.is_final_transcript if keep_transcript else False,
            llm_text=self.llm_text if keep_llm_text else "",
            is_final_llm=self.is_final_llm if keep_llm_text else True,
            tool_name=None,
            error_message=None,
        )

    def with_transcript(self, text: str, is_final: bool, *, at: float | None = None) -> "UIState":
        return replace(
            self,
            partial_transcript=text,
            is_final_transcript=is_final,
            updated_at=at if at is not None else time.monotonic(),
        )

    def with_llm_output(self, text: str, is_final: bool, *, at: float | None = None) -> "UIState":
        return replace(
            self,
            llm_text=text,
            is_final_llm=is_final,
            updated_at=at if at is not None else time.monotonic(),
        )

    def with_memory_status(
        self, status: str, match_count: int | None = None, *, at: float | None = None
    ) -> "UIState":
        return replace(
            self,
            memory_status=status,
            memory_match_count=match_count,
            updated_at=at if at is not None else time.monotonic(),
        )

    def with_hermes_tasks(
        self, tasks: tuple[HermesTaskView, ...], *, at: float | None = None
    ) -> "UIState":
        return replace(
            self,
            hermes_tasks=tasks,
            updated_at=at if at is not None else time.monotonic(),
        )

    def with_tool(self, tool_name: str, *, at: float | None = None) -> "UIState":
        return replace(
            self,
            phase=UIPhase.TOOL_RUNNING,
            tool_name=tool_name,
            error_message=None,
            updated_at=at if at is not None else time.monotonic(),
        )

    def with_error(self, error_message: str, *, at: float | None = None) -> "UIState":
        return replace(
            self,
            phase=UIPhase.ERROR,
            error_message=error_message,
            tool_name=None,
            updated_at=at if at is not None else time.monotonic(),
        )

    def with_attached(self, attached: bool, *, at: float | None = None) -> "UIState":
        return replace(
            self,
            attached=attached,
            updated_at=at if at is not None else time.monotonic(),
        )
