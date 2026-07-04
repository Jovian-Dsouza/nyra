"""Per-phase label and glow-breathing speed.

Warm monochrome direction: one accent family (the ember gradient), never a
rainbow of per-phase colors. Phases differentiate through motion (how fast
the glow breathes) and the status label text, not hue.
"""

from dataclasses import dataclass

from nyra_ui import constants as const
from nyra_ui.state import UIPhase


@dataclass(frozen=True)
class PhaseTheme:
    label: str
    breathe_seconds: float


_THEMES: dict[UIPhase, PhaseTheme] = {
    UIPhase.IDLE: PhaseTheme("Idle", const.BREATHE_SECONDS_IDLE),
    UIPhase.STANDBY: PhaseTheme("Listening for wake word", const.BREATHE_SECONDS_STANDBY),
    UIPhase.LISTENING: PhaseTheme("Listening", const.BREATHE_SECONDS_LISTENING),
    UIPhase.THINKING: PhaseTheme("Thinking", const.BREATHE_SECONDS_THINKING),
    UIPhase.TOOL_RUNNING: PhaseTheme("Tool running", const.BREATHE_SECONDS_TOOL_RUNNING),
    UIPhase.SPEAKING: PhaseTheme("Speaking", const.BREATHE_SECONDS_SPEAKING),
    UIPhase.ERROR: PhaseTheme("Error", const.BREATHE_SECONDS_ERROR),
}


def get_theme(phase: UIPhase) -> PhaseTheme:
    return _THEMES[phase]
