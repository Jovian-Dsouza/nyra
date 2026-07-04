"""Layout, timing, and sizing constants for the Nyra status window.

Centralized so nothing else in the package hardcodes a magic number.

Design direction: warm monochrome dark canvas (Linear/Raycast-style
restraint) with a single reserved "ember" gradient (amber -> crimson) used
only for the ambient glow, the live status dot, and the "Nyra" field label —
never as a rainbow of per-phase colors and never as flat filled tag boxes.
"""

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 320
TARGET_FPS = 30

BG_CENTER = (32, 26, 21)
BG_MID = (23, 20, 17)
BG_EDGE = (13, 11, 10)

INK = (242, 237, 230)
INK_DIM = (156, 150, 143)
INK_FAINT = (107, 102, 95)
HAIRLINE_COLOR = (34, 30, 27)

EMBER_1 = (255, 180, 67)
EMBER_2 = (255, 106, 61)
EMBER_3 = (255, 61, 104)

FONT_NAME_MONO = "Menlo,Consolas,Courier New"
FONT_NAME_SANS = "Avenir Next,Helvetica Neue,Helvetica"

WORDMARK_HEIGHT = 18
FONT_SIZE_STATUS = 11
FONT_SIZE_LABEL = 10
FONT_SIZE_BODY = 15
FONT_SIZE_CLOCK_LARGE = 52
FONT_SIZE_CLOCK_SMALL = 12
FONT_SIZE_DATE = 11

MARGIN_X = 18
HEADER_TOP = 14
CONTENT_TOP = 54
FIELD_GAP = 14
CONVERSATION_LINE_HEIGHT = 18
HERMES_ROW_HEIGHT = 22
HERMES_MAX_VISIBLE_TASKS = 3
FOOTER_HEIGHT = 56
LOGO_HEIGHT = 36
MEMORY_LOGO_HEIGHT = 16

LISTENING_PROMPT = "Speak something to start working."
"""Shown under the You field while Nyra is listening and no speech has arrived yet."""

GLOW_IDLE_DIAMETER = 260
GLOW_IDLE_CENTER_RATIO = 0.40
GLOW_ACTIVE_DIAMETER = 70
GLOW_ACTIVE_CENTER = (SCREEN_WIDTH - 34, 24)

# Seconds per full breathe cycle, per agent phase — the one place phases are
# differentiated visually (motion, not hue: the accent stays a single ember).
BREATHE_SECONDS_IDLE = 4.5
BREATHE_SECONDS_STANDBY = 3.6
BREATHE_SECONDS_LISTENING = 2.2
BREATHE_SECONDS_THINKING = 0.9
BREATHE_SECONDS_SPEAKING = 1.1
BREATHE_SECONDS_TOOL_RUNNING = 0.7
BREATHE_SECONDS_ERROR = 0.5

IDLE_ATTACH_TIMEOUT_SECONDS = 3.0
"""How long the UI keeps showing the small clock after the agent process
disconnects before falling back to the dominant idle clock."""
