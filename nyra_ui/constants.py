"""Layout, timing, and sizing constants for the Nyra status window.

Centralized so nothing else in the package hardcodes a magic number.
"""

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 360
TARGET_FPS = 30

# Neo Supreme: stark black/white canvas, one loud red hit for branding/status —
# no soft gradients, no pastel per-phase colors, no thin hairline dividers.
BACKGROUND_COLOR = (0, 0, 0)
PRIMARY_TEXT_COLOR = (255, 255, 255)
SECONDARY_TEXT_COLOR = (190, 190, 190)
MUTED_TEXT_COLOR = (120, 120, 120)
DIVIDER_COLOR = (237, 28, 36)

SUPREME_RED = (237, 28, 36)
TAG_TEXT_COLOR = (255, 255, 255)
DIVIDER_THICKNESS = 3

FONT_NAME_HEAVY = "Arial Black,Helvetica Neue,Helvetica"
FONT_NAME_BODY = "Helvetica Neue,Helvetica,Arial"

FONT_SIZE_HEADER = 18
FONT_SIZE_BODY = 15
FONT_SIZE_SMALL = 13
FONT_SIZE_CLOCK_LARGE = 56
FONT_SIZE_CLOCK_SMALL = 15

MARGIN_X = 14
HEADER_HEIGHT = 38
CLOCK_SMALL_HEIGHT = 22
STT_PANEL_HEIGHT = 56
LLM_PANEL_HEIGHT = 56
MEMORY_LINE_HEIGHT = 20
HERMES_PANEL_HEIGHT = 64
HERMES_MAX_VISIBLE_TASKS = 3
BRANDING_STRIP_HEIGHT = 34
LOGO_HEIGHT = 20

PANEL_MAX_CHARS = 90
IDLE_ATTACH_TIMEOUT_SECONDS = 3.0
"""How long the UI keeps showing the small clock after the agent process
disconnects before falling back to the dominant idle clock."""
