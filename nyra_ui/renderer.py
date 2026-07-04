"""The only module that draws with pygame.

Neo Supreme: a stark black/white canvas with one loud red hit reserved for
the brand tag and status tag — solid rectangular "box logo" blocks, heavy
bold type, thick square-edged dividers. No gradients, no soft per-phase
colors, no rounded corners.
"""

from datetime import datetime

import pygame

from nyra_ui import constants as const
from nyra_ui.state import UIPhase, UIState
from nyra_ui.theme import get_theme


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


class PygameRenderer:
    def __init__(self) -> None:
        pygame.font.init()
        self._font_heavy = pygame.font.SysFont(const.FONT_NAME_HEAVY, const.FONT_SIZE_HEADER, bold=True)
        self._font_heavy.set_italic(True)
        self._font_tag = pygame.font.SysFont(const.FONT_NAME_HEAVY, const.FONT_SIZE_SMALL, bold=True)
        self._font_body = pygame.font.SysFont(const.FONT_NAME_BODY, const.FONT_SIZE_BODY, bold=True)
        self._font_small = pygame.font.SysFont(const.FONT_NAME_BODY, const.FONT_SIZE_SMALL, bold=True)
        self._font_clock_large = pygame.font.SysFont(const.FONT_NAME_HEAVY, const.FONT_SIZE_CLOCK_LARGE, bold=True)
        self._font_clock_small = pygame.font.SysFont(const.FONT_NAME_HEAVY, const.FONT_SIZE_CLOCK_SMALL, bold=True)
        self._logos = self._load_logos()

    def _load_logos(self) -> dict[str, "pygame.Surface"]:
        logos = {}
        assets_dir = __file__.rsplit("/", 1)[0] + "/assets"
        for name, filename in (("cognee", "cognee_logo.png"), ("hermes", "hermes_logo.png")):
            path = f"{assets_dir}/{filename}"
            try:
                image = pygame.image.load(path).convert_alpha()
                scale = const.LOGO_HEIGHT / image.get_height()
                size = (max(1, int(image.get_width() * scale)), const.LOGO_HEIGHT)
                logos[name] = pygame.transform.smoothscale(image, size)
            except (FileNotFoundError, pygame.error):
                logos[name] = None
        return logos

    def _draw_tag(
        self,
        surface: "pygame.Surface",
        text: str,
        x: int,
        y: int,
        font: "pygame.font.Font",
        *,
        align_right: bool = False,
        pad_x: int = 8,
        pad_y: int = 4,
    ) -> "pygame.Rect":
        """Solid red rectangle with bold white text — the "box logo" tag."""
        label = font.render(text, True, const.TAG_TEXT_COLOR)
        width = label.get_width() + pad_x * 2
        height = label.get_height() + pad_y * 2
        left = x - width if align_right else x
        rect = pygame.Rect(left, y, width, height)
        pygame.draw.rect(surface, const.SUPREME_RED, rect)
        surface.blit(label, (rect.x + pad_x, rect.y + pad_y))
        return rect

    def _draw_marker_label(self, surface: "pygame.Surface", text: str, x: int, y: int) -> int:
        """Small red square + bold caps label, used for section headings."""
        square = pygame.Rect(x, y + 3, 8, 8)
        pygame.draw.rect(surface, const.SUPREME_RED, square)
        label = self._font_small.render(text.upper(), True, const.PRIMARY_TEXT_COLOR)
        surface.blit(label, (x + 14, y))
        return y + max(label.get_height(), 8)

    def render(self, surface: "pygame.Surface", state: UIState, now: datetime) -> None:
        surface.fill(const.BACKGROUND_COLOR)
        theme = get_theme(state.phase)

        is_dominant_clock = (
            state.phase is UIPhase.IDLE
            and not state.attached
            and not state.partial_transcript
            and not state.llm_text
            and not state.hermes_tasks
        )

        self._draw_header(surface, theme, show_small_clock=not is_dominant_clock, now=now)

        if is_dominant_clock:
            self._draw_dominant_clock(surface, now)
        else:
            y = const.HEADER_HEIGHT + const.DIVIDER_THICKNESS + 8
            y = self._draw_stt_panel(surface, state, y)
            y = self._draw_llm_panel(surface, state, y)
            y = self._draw_memory_line(surface, state, y)
            self._draw_hermes_panel(surface, state, y)

        self._draw_branding_strip(surface)

    def _draw_header(self, surface, theme, *, show_small_clock: bool, now: datetime) -> None:
        brand_rect = self._draw_tag(surface, "NYRA", const.MARGIN_X, 8, self._font_heavy)

        right_edge = const.SCREEN_WIDTH - const.MARGIN_X
        if show_small_clock:
            clock_text = self._font_clock_small.render(now.strftime("%H:%M:%S"), True, const.PRIMARY_TEXT_COLOR)
            clock_x = right_edge - clock_text.get_width()
            surface.blit(clock_text, (clock_x, brand_rect.y + (brand_rect.height - clock_text.get_height()) // 2))
            right_edge = clock_x - 10

        self._draw_tag(surface, theme.label, right_edge, 8, self._font_tag, align_right=True)

        divider_y = const.HEADER_HEIGHT
        pygame.draw.rect(
            surface, const.DIVIDER_COLOR, (0, divider_y, const.SCREEN_WIDTH, const.DIVIDER_THICKNESS)
        )

    def _draw_dominant_clock(self, surface, now: datetime) -> None:
        text = self._font_clock_large.render(now.strftime("%H:%M:%S"), True, const.PRIMARY_TEXT_COLOR)
        body_top = const.HEADER_HEIGHT + const.DIVIDER_THICKNESS
        body_bottom = const.SCREEN_HEIGHT - const.BRANDING_STRIP_HEIGHT
        top = body_top + (body_bottom - body_top) // 2 - text.get_height() // 2
        text_x = (const.SCREEN_WIDTH - text.get_width()) // 2
        surface.blit(text, (text_x, top))

        bar_width = text.get_width() // 2
        bar_y = top + text.get_height() + 10
        pygame.draw.rect(
            surface,
            const.SUPREME_RED,
            ((const.SCREEN_WIDTH - bar_width) // 2, bar_y, bar_width, const.DIVIDER_THICKNESS),
        )

    def _draw_stt_panel(self, surface, state: UIState, y: int) -> int:
        y = self._draw_marker_label(surface, "You", const.MARGIN_X, y)
        color = const.PRIMARY_TEXT_COLOR if state.is_final_transcript else const.SECONDARY_TEXT_COLOR
        for i, line in enumerate(_wrap_text(state.partial_transcript, self._font_body, const.SCREEN_WIDTH - 2 * const.MARGIN_X)):
            text = self._font_body.render(line, True, color)
            surface.blit(text, (const.MARGIN_X, y + 4 + i * 18))
        return y + const.STT_PANEL_HEIGHT - 16

    def _draw_llm_panel(self, surface, state: UIState, y: int) -> int:
        y = self._draw_marker_label(surface, "Reply", const.MARGIN_X, y)
        color = const.PRIMARY_TEXT_COLOR if state.is_final_llm else const.SECONDARY_TEXT_COLOR
        for i, line in enumerate(_wrap_text(state.llm_text, self._font_body, const.SCREEN_WIDTH - 2 * const.MARGIN_X)):
            text = self._font_body.render(line, True, color)
            surface.blit(text, (const.MARGIN_X, y + 4 + i * 18))
        return y + const.LLM_PANEL_HEIGHT - 16

    def _draw_memory_line(self, surface, state: UIState, y: int) -> int:
        if state.memory_status == "recalling":
            text = "MEMORY: RECALLING…"
        elif state.memory_status == "done":
            count = state.memory_match_count or 0
            noun = "MATCH" if count == 1 else "MATCHES"
            text = f"MEMORY: {count} {noun}" if count else "MEMORY: NO MATCHES"
        else:
            text = ""
        if text:
            rendered = self._font_small.render(text, True, const.MUTED_TEXT_COLOR)
            surface.blit(rendered, (const.MARGIN_X, y + 2))
        return y + const.MEMORY_LINE_HEIGHT

    def _draw_hermes_panel(self, surface, state: UIState, y: int) -> int:
        y = self._draw_marker_label(surface, "Hermes", const.MARGIN_X, y)
        tasks = state.hermes_tasks
        if not tasks:
            text = self._font_small.render("NO BACKGROUND TASKS", True, const.MUTED_TEXT_COLOR)
            surface.blit(text, (const.MARGIN_X, y + 4))
            return y + const.HERMES_PANEL_HEIGHT

        visible = tasks[: const.HERMES_MAX_VISIBLE_TASKS]
        for i, task in enumerate(visible):
            tokens = f"{task.tokens_used}TOK" if task.tokens_used is not None else "—"
            line = f"{task.label.upper()}  {task.status.upper()}  {int(task.elapsed_seconds)}S  {tokens}"
            rendered = self._font_small.render(line, True, const.PRIMARY_TEXT_COLOR)
            surface.blit(rendered, (const.MARGIN_X, y + 4 + i * 16))

        remaining = len(tasks) - len(visible)
        if remaining > 0:
            more = self._font_small.render(f"+{remaining} MORE", True, const.MUTED_TEXT_COLOR)
            surface.blit(more, (const.MARGIN_X, y + 4 + len(visible) * 16))

        return y + const.HERMES_PANEL_HEIGHT

    def _draw_branding_strip(self, surface) -> None:
        y = const.SCREEN_HEIGHT - const.BRANDING_STRIP_HEIGHT
        pygame.draw.rect(surface, const.SUPREME_RED, (0, y, const.SCREEN_WIDTH, const.DIVIDER_THICKNESS))
        center_y = y + const.DIVIDER_THICKNESS + (const.BRANDING_STRIP_HEIGHT - const.DIVIDER_THICKNESS) // 2

        cognee = self._logos.get("cognee")
        if cognee is not None:
            surface.blit(cognee, (const.MARGIN_X, center_y - cognee.get_height() // 2))

        hermes = self._logos.get("hermes")
        if hermes is not None:
            x = const.SCREEN_WIDTH - const.MARGIN_X - hermes.get_width()
            surface.blit(hermes, (x, center_y - hermes.get_height() // 2))
