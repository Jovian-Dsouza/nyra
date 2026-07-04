"""UI application loop.

Always runs on this process's true main thread — this process exists only to
run the UI, so there is no Cocoa/SDL main-thread conflict to work around.
Any failure here (missing pygame, no display, a render-loop crash) is caught
and logged by the caller; it must never look like a crash to the agent worker
that spawned this process.
"""

import logging
import os
from datetime import datetime
from typing import Protocol

import pygame

from nyra_ui import constants as const
from nyra_ui.renderer import PygameRenderer
from nyra_ui.state import UIState

logger = logging.getLogger(__name__)


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _fullscreen_enabled() -> bool:
    return _strip_env(os.environ.get("NYRA_UI_FULLSCREEN", "false")).lower() in (
        "1",
        "true",
        "yes",
    )


class StateSource(Protocol):
    def snapshot(self) -> UIState: ...


class UIApp:
    def __init__(self, store: StateSource) -> None:
        self._store = store

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Nyra")
        flags = pygame.FULLSCREEN if _fullscreen_enabled() else 0
        screen = pygame.display.set_mode((const.SCREEN_WIDTH, const.SCREEN_HEIGHT), flags)
        if flags & pygame.FULLSCREEN:
            pygame.mouse.set_visible(False)
        renderer = PygameRenderer()
        clock = pygame.time.Clock()
        start = pygame.time.get_ticks()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            elapsed = (pygame.time.get_ticks() - start) / 1000.0
            renderer.render(screen, self._store.snapshot(), datetime.now(), elapsed)
            pygame.display.flip()
            clock.tick(const.TARGET_FPS)

        pygame.quit()
