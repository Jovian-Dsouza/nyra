from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING},
    State.LISTENING: {State.THINKING, State.IDLE},
    State.THINKING: {State.SPEAKING, State.IDLE, State.LISTENING},
    State.SPEAKING: {State.IDLE, State.LISTENING},
}


@dataclass
class StateMachine:
    state: State = State.IDLE
    history: list[tuple[State, State, str]] = field(default_factory=list)

    def transition(self, target: State, reason: str) -> None:
        allowed = ALLOWED_TRANSITIONS[self.state]
        if target not in allowed:
            raise ValueError(f"illegal transition: {self.state.value} -> {target.value} ({reason})")
        previous = self.state
        self.state = target
        self.history.append((previous, target, reason))

    def set_listening_for_barge_in(self, reason: str = "barge-in") -> None:
        if self.state == State.LISTENING:
            return
        previous = self.state
        self.state = State.LISTENING
        self.history.append((previous, State.LISTENING, reason))

