"""UI application loop.

Always runs on this process's true main thread — this process exists only to
run the UI, so there is no Cocoa/SDL main-thread conflict to work around.
Any failure here (missing pygame, no display, a render-loop crash) is caught
and logged by the caller; it must never look like a crash to the agent worker
that spawned this process.
"""

import logging
from datetime import datetime
from typing import Protocol

import pygame

from nyra_ui import constants as const
from nyra_ui.renderer import PygameRenderer
from nyra_ui.state import UIState

logger = logging.getLogger(__name__)


class StateSource(Protocol):
    def snapshot(self) -> UIState: ...


class UIApp:
    def __init__(self, store: StateSource) -> None:
        self._store = store

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Nyra")
        screen = pygame.display.set_mode((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        renderer = PygameRenderer()
        clock = pygame.time.Clock()
        start = pygame.time.get_ticks()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            elapsed = (pygame.time.get_ticks() - start) / 1000.0
            renderer.render(screen, self._store.snapshot(), datetime.now(), elapsed)
            pygame.display.flip()
            clock.tick(const.TARGET_FPS)

        pygame.quit()
