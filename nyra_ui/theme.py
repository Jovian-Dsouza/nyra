"""Per-phase label for the status tag.

Neo Supreme is monochrome-plus-one-red, not a rainbow of phase colors: every
phase renders in the same `constants.SUPREME_RED`, so this module only tracks
the label text. Pure data — no rendering logic lives here.
"""

from dataclasses import dataclass

from nyra_ui.state import UIPhase


@dataclass(frozen=True)
class PhaseTheme:
    label: str


_THEMES: dict[UIPhase, PhaseTheme] = {
    UIPhase.IDLE: PhaseTheme(label="IDLE"),
    UIPhase.LISTENING: PhaseTheme(label="LISTENING"),
    UIPhase.THINKING: PhaseTheme(label="THINKING"),
    UIPhase.TOOL_RUNNING: PhaseTheme(label="TOOL RUNNING"),
    UIPhase.SPEAKING: PhaseTheme(label="SPEAKING"),
    UIPhase.ERROR: PhaseTheme(label="ERROR"),
}


def get_theme(phase: UIPhase) -> PhaseTheme:
    return _THEMES[phase]
