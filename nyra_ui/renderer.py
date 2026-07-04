"""The only module that draws with pygame.

Warm monochrome canvas, one reserved "ember" gradient (amber -> crimson) used
only for the ambient glow, the live status dot, and the "Nyra" field label.
Everything else is flat ink-on-dark text with hairline dividers — no boxes,
no per-phase rainbow, no hard drop shadows.
"""

import math
from datetime import datetime

import pygame

from nyra_ui import constants as const
from nyra_ui.state import UIPhase, UIState, _has_active_hermes_tasks
from nyra_ui.theme import get_theme


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(_lerp(c1[i], c2[i], t)) for i in range(3))


def _wrap_text(text: str, font: "pygame.font.Font", max_width: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) == max_lines:
        last = lines[-1]
        while font.size(last + "…")[0] > max_width and len(last) > 1:
            last = last[:-1]
        joined_len = sum(len(w) + 1 for w in words)
        if joined_len > sum(len(line) for line in lines):
            lines[-1] = last.rstrip() + "…"
    return lines or [""]


def _build_background(width: int, height: int) -> "pygame.Surface":
    """Warm radial vignette, centered high, fading to near-black at the edges.
    Built once with concentric rings — a cheap stand-in for a CSS radial-gradient."""
    surface = pygame.Surface((width, height))
    surface.fill(const.BG_EDGE)
    center = (width // 2, int(height * 0.16))
    max_radius = int(width * 0.72)
    steps = 90
    for i in range(steps, -1, -1):
        t = i / steps
        radius = int(max_radius * t)
        if t > 0.5:
            color = _lerp_color(const.BG_MID, const.BG_EDGE, (t - 0.5) * 2)
        else:
            color = _lerp_color(const.BG_CENTER, const.BG_MID, t * 2)
        pygame.draw.circle(surface, color, center, radius)
    return surface


def _build_glow_sprite(diameter: int) -> "pygame.Surface":
    """Soft ember radial glow: amber core -> orange -> crimson -> transparent.
    Concentric alpha-blended rings at high step count fake a gaussian blur."""
    surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    center = (diameter // 2, diameter // 2)
    max_radius = diameter // 2
    steps = 120
    stops = [
        (0.0, const.EMBER_1, 235),
        (0.35, const.EMBER_2, 150),
        (0.55, const.EMBER_3, 90),
        (0.72, const.EMBER_3, 0),
    ]
    for i in range(steps, -1, -1):
        t = i / steps
        radius = max(1, int(max_radius * t))
        color = stops[-1][1]
        alpha = 0
        for j in range(len(stops) - 1):
            t0, c0, a0 = stops[j]
            t1, c1, a1 = stops[j + 1]
            if t0 <= t <= t1:
                local_t = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                color = _lerp_color(c0, c1, local_t)
                alpha = int(_lerp(a0, a1, local_t))
                break
        else:
            if t < stops[0][0]:
                color, alpha = stops[0][1], stops[0][2]
        pygame.draw.circle(surface, (*color, alpha), center, radius)
    return surface


def _crop_to_content(image: "pygame.Surface") -> "pygame.Surface":
    """Trim solid padding around a logo so scaling targets the mark, not the canvas."""
    width, height = image.get_size()
    min_x, min_y, max_x, max_y = width, height, 0, 0
    for y in range(height):
        for x in range(width):
            red, green, blue, _alpha = image.get_at((x, y))
            if red + green + blue > 30:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if min_x >= max_x or min_y >= max_y:
        return image
    crop = pygame.Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    return image.subsurface(crop).copy()


def _tint_logo(image: "pygame.Surface", color: tuple[int, int, int]) -> "pygame.Surface":
    """Recolor a light-on-dark logo using pixel luminance as alpha."""
    width, height = image.get_size()
    tinted = pygame.Surface((width, height), pygame.SRCALPHA)
    for y in range(height):
        for x in range(width):
            red, green, blue, alpha = image.get_at((x, y))
            luminance = max(red, green, blue)
            if luminance > 30:
                pixel_alpha = int(luminance * alpha / 255)
                tinted.set_at((x, y), (*color, pixel_alpha))
    return tinted


def _gradient_text(text: str, font: "pygame.font.Font", start: tuple, end: tuple) -> "pygame.Surface":
    """Renders text filled with a horizontal amber->crimson gradient."""
    mask = font.render(text, True, (255, 255, 255))
    width, height = mask.get_size()
    gradient = pygame.Surface((max(width, 1), max(height, 1)), pygame.SRCALPHA)
    for x in range(width):
        t = x / max(width - 1, 1)
        color = _lerp_color(start, end, t)
        pygame.draw.line(gradient, (*color, 255), (x, 0), (x, height))
    gradient.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return gradient


class PygameRenderer:
    def __init__(self) -> None:
        pygame.font.init()
        self._font_status = pygame.font.SysFont(const.FONT_NAME_SANS, const.FONT_SIZE_STATUS, bold=True)
        self._font_label = pygame.font.SysFont(const.FONT_NAME_SANS, const.FONT_SIZE_LABEL, bold=True)
        self._font_body = pygame.font.SysFont(const.FONT_NAME_SANS, const.FONT_SIZE_BODY)
        self._font_mono = pygame.font.SysFont(const.FONT_NAME_MONO, const.FONT_SIZE_STATUS)
        self._font_clock_large = pygame.font.SysFont(const.FONT_NAME_MONO, const.FONT_SIZE_CLOCK_LARGE, bold=True)
        self._font_clock_small = pygame.font.SysFont(const.FONT_NAME_MONO, const.FONT_SIZE_CLOCK_SMALL)
        self._font_date = pygame.font.SysFont(const.FONT_NAME_SANS, const.FONT_SIZE_DATE, bold=True)

        self._background = _build_background(const.SCREEN_WIDTH, const.SCREEN_HEIGHT)
        self._glow_source = _build_glow_sprite(360)
        self._dot_glow_source = _build_glow_sprite(120)
        self._wordmark = self._load_wordmark()
        self._logos = self._load_logos()

    def _load_wordmark(self) -> "pygame.Surface | None":
        assets_dir = __file__.rsplit("/", 1)[0] + "/assets"
        path = f"{assets_dir}/nyra_logo.png"
        try:
            image = pygame.image.load(path).convert_alpha()
            cropped = _crop_to_content(image)
            scale = const.WORDMARK_HEIGHT / cropped.get_height()
            size = (max(1, int(cropped.get_width() * scale)), const.WORDMARK_HEIGHT)
            return pygame.transform.smoothscale(cropped, size)
        except (FileNotFoundError, pygame.error):
            return None

    def _load_logos(self) -> dict[str, "pygame.Surface"]:
        logos = {}
        assets_dir = __file__.rsplit("/", 1)[0] + "/assets"
        for name, filename in (("cognee", "cognee_logo.png"), ("hermes", "hermes_logo.png")):
            path = f"{assets_dir}/{filename}"
            try:
                image = _crop_to_content(pygame.image.load(path).convert_alpha())
                scale = const.LOGO_HEIGHT / image.get_height()
                size = (max(1, int(image.get_width() * scale)), const.LOGO_HEIGHT)
                scaled = pygame.transform.smoothscale(image, size)
                logos[name] = _tint_logo(scaled, const.INK_DIM)
            except (FileNotFoundError, pygame.error):
                logos[name] = None
        return logos

    def _blit_glow(self, surface, source: "pygame.Surface", center: tuple, diameter: float) -> None:
        # Plain alpha compositing, not BLEND_RGBA_ADD: ADD ignores per-pixel
        # alpha when the destination is opaque, so the "soft falloff" edge
        # would render as a hard, fully-saturated disc instead of fading.
        diameter = max(1, int(diameter))
        scaled = pygame.transform.smoothscale(source, (diameter, diameter))
        rect = scaled.get_rect(center=center)
        surface.blit(scaled, rect)

    def render(self, surface: "pygame.Surface", state: UIState, now: datetime, elapsed: float) -> None:
        surface.blit(self._background, (0, 0))
        theme = get_theme(state.phase)
        breathe = 0.5 + 0.5 * math.sin(2 * math.pi * elapsed / theme.breathe_seconds)

        is_wake_listening = state.phase is UIPhase.STANDBY
        is_dominant_clock = is_wake_listening or (
            state.phase is UIPhase.IDLE
            and not state.attached
            and not state.partial_transcript
            and not state.llm_text
            and not state.hermes_tasks
        )

        if is_dominant_clock:
            self._draw_idle(surface, now, breathe)
        else:
            self._draw_header(surface, theme, now, breathe, active=True)
            y = const.CONTENT_TOP
            y = self._draw_conversation_field(surface, state, y)
            y = self._draw_memory_field(surface, state, y)
            self._draw_hermes_field(surface, state, y)
            self._draw_footer(surface)
            return

        self._draw_header(surface, theme, now, breathe, active=False)
        self._draw_footer(surface)

    def _draw_header(self, surface, theme, now: datetime, breathe: float, *, active: bool) -> None:
        if self._wordmark is not None:
            surface.blit(self._wordmark, (const.MARGIN_X, const.HEADER_TOP))

        if not active:
            return

        label = self._font_status.render(theme.label.upper(), True, const.INK_DIM)
        clock_text = self._font_clock_small.render(now.strftime("%H:%M:%S"), True, const.INK_DIM)

        right = const.SCREEN_WIDTH - const.MARGIN_X
        label_x = right - label.get_width()
        dot_x = label_x - 12
        clock_x = dot_x - 10 - clock_text.get_width()

        baseline_y = const.HEADER_TOP + 2
        surface.blit(clock_text, (clock_x, baseline_y))
        surface.blit(label, (label_x, baseline_y))

        dot_diameter = 5 + 3 * breathe
        self._blit_glow(surface, self._dot_glow_source, (dot_x, baseline_y + 5), 14 + 10 * breathe)
        color = _lerp_color(const.EMBER_2, const.EMBER_1, breathe)
        pygame.draw.circle(surface, color, (dot_x, baseline_y + 5), max(2, int(dot_diameter / 2)))

    def _draw_idle(self, surface, now: datetime, breathe: float) -> None:
        center = (const.SCREEN_WIDTH // 2, int(const.SCREEN_HEIGHT * const.GLOW_IDLE_CENTER_RATIO))
        diameter = const.GLOW_IDLE_DIAMETER * (0.94 + 0.12 * breathe)
        self._blit_glow(surface, self._glow_source, center, diameter)

        time_text = self._font_clock_large.render(now.strftime("%H:%M:%S"), True, const.INK)
        time_rect = time_text.get_rect(center=center)
        surface.blit(time_text, time_rect)

        date_text = self._font_date.render(now.strftime("%A, %B %-d").upper(), True, const.INK_FAINT)
        date_rect = date_text.get_rect(center=(center[0], time_rect.bottom + 18))
        surface.blit(date_text, date_rect)

    def _draw_field_label(self, surface, text: str, x: int, y: int, *, gradient: bool = False) -> int:
        if gradient:
            label = _gradient_text(text, self._font_label, const.EMBER_1, const.EMBER_3)
        else:
            label = self._font_label.render(text.upper(), True, const.INK_FAINT)
        surface.blit(label, (x, y))
        return y + label.get_height() + 4

    def _draw_conversation_field(self, surface, state: UIState, y: int) -> int:
        """Shows exactly one side of the conversation at a time — whichever is
        actively speaking — instead of stacking "You" and "Nyra" together."""
        if state.phase is UIPhase.SPEAKING:
            label, text, is_final = "Nyra", state.llm_text, state.is_final_llm
            gradient = True
        elif state.partial_transcript:
            label, text, is_final = "You", state.partial_transcript, state.is_final_transcript
            gradient = False
        elif state.llm_text and _has_active_hermes_tasks(state):
            label, text, is_final = "Nyra", state.llm_text, state.is_final_llm
            gradient = True
        else:
            label = "You:"
            text = state.partial_transcript
            is_final = state.is_final_transcript
            gradient = False
            placeholder = (
                state.phase is UIPhase.LISTENING
                and not text.strip()
            )

        y = self._draw_field_label(surface, label, const.MARGIN_X, y, gradient=gradient)
        if placeholder:
            color = const.INK_FAINT
            text = const.LISTENING_PROMPT
        else:
            color = const.INK if is_final else const.INK_DIM
        lines = _wrap_text(
            text, self._font_body, const.SCREEN_WIDTH - 2 * const.MARGIN_X, max_lines=3
        )
        for i, line in enumerate(lines):
            rendered = self._font_body.render(line, True, color)
            surface.blit(rendered, (const.MARGIN_X, y + i * const.CONVERSATION_LINE_HEIGHT))
        return y + len(lines) * const.CONVERSATION_LINE_HEIGHT + const.FIELD_GAP

    def _draw_memory_field(self, surface, state: UIState, y: int) -> int:
        if state.memory_status == "recalling":
            text = "Memory — recalling…"
        elif state.memory_status == "done":
            count = state.memory_match_count or 0
            noun = "match" if count == 1 else "matches"
            text = f"Memory — {count} {noun} recalled" if count else "Memory — no matches"
        else:
            text = ""
        if text:
            rendered = self._font_mono.render(text, True, const.INK_FAINT)
            surface.blit(rendered, (const.MARGIN_X, y))
        return y + 18 + const.FIELD_GAP // 2

    def _draw_hermes_field(self, surface, state: UIState, y: int) -> int:
        tasks = state.hermes_tasks
        if not tasks:
            return y

        y = self._draw_field_label(surface, "Hermes", const.MARGIN_X, y)
        row_width = const.SCREEN_WIDTH - 2 * const.MARGIN_X

        visible = tasks[: const.HERMES_MAX_VISIBLE_TASKS]
        for i, task in enumerate(visible):
            row_y = y + i * const.HERMES_ROW_HEIGHT
            if i > 0:
                pygame.draw.line(
                    surface, const.HAIRLINE_COLOR,
                    (const.MARGIN_X, row_y - 4), (const.MARGIN_X + row_width, row_y - 4),
                )
            left = self._font_body.render(task.label, True, const.INK)
            surface.blit(left, (const.MARGIN_X, row_y))
            status = self._font_mono.render(task.status, True, const.INK_FAINT)
            surface.blit(status, (const.MARGIN_X + left.get_width() + 10, row_y + 2))

            tokens = f"{task.tokens_used} tok" if task.tokens_used is not None else "—"
            meta = f"{int(task.elapsed_seconds)}s · {tokens}"
            meta_text = self._font_mono.render(meta, True, const.INK_DIM)
            surface.blit(meta_text, (const.MARGIN_X + row_width - meta_text.get_width(), row_y + 1))

        remaining = len(tasks) - len(visible)
        end_y = y + len(visible) * const.HERMES_ROW_HEIGHT
        if remaining > 0:
            more = self._font_mono.render(f"+{remaining} more", True, const.INK_FAINT)
            surface.blit(more, (const.MARGIN_X, end_y))
            end_y += const.HERMES_ROW_HEIGHT
        return end_y

    def _draw_footer(self, surface) -> None:
        y = const.SCREEN_HEIGHT - const.FOOTER_HEIGHT
        pygame.draw.line(surface, const.HAIRLINE_COLOR, (0, y), (const.SCREEN_WIDTH, y), 1)
        center_y = y + (const.FOOTER_HEIGHT // 2)

        cognee = self._logos.get("cognee")
        if cognee is not None:
            surface.blit(cognee, (const.MARGIN_X, center_y - cognee.get_height() // 2))

        hermes = self._logos.get("hermes")
        if hermes is not None:
            x = const.SCREEN_WIDTH - const.MARGIN_X - hermes.get_width()
            surface.blit(hermes, (x, center_y - hermes.get_height() // 2))
