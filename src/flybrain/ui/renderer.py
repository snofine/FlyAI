from __future__ import annotations

import math
from typing import Any


class Renderer:
    def __init__(self, world_width: float, world_height: float, debug: bool = False) -> None:
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("Interactive mode requires: uv sync --extra ui") from error
        self.pg = pygame
        pygame.init()
        self.debug = debug
        self.world_width, self.world_height = world_width, world_height
        self.view_width, self.view_height = 900, 630
        self.panel_width = 330
        self.screen = pygame.display.set_mode((self.view_width + self.panel_width, self.view_height))
        pygame.display.set_caption("FlyBrain — MaleCNS connectome simulation")
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.clock = pygame.time.Clock()

    def events(self, fly: Any | None = None) -> tuple[bool, bool, list[str]]:
        reset = False
        stimuli: list[str] = []
        key_stimuli = {
            self.pg.K_1: "eye_left",
            self.pg.K_2: "eye_right",
            self.pg.K_3: "leg_front",
            self.pg.K_4: "leg_middle",
            self.pg.K_5: "leg_hind",
            self.pg.K_6: "touch_head",
        }
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE):
                return False, reset, stimuli
            if event.type == self.pg.KEYDOWN and event.key == self.pg.K_SPACE:
                reset = True
            if event.type == self.pg.KEYDOWN and event.key in key_stimuli:
                stimuli.append(key_stimuli[event.key])
            if event.type == self.pg.MOUSEBUTTONDOWN and fly is not None:
                fx, fy = self._point(fly.x, fly.y)
                mx, my = event.pos
                if (mx - fx) ** 2 + (my - fy) ** 2 <= 28**2:
                    self.debug = not self.debug
        return True, reset, stimuli

    def _point(self, x: float, y: float) -> tuple[int, int]:
        return int(x / self.world_width * self.view_width), int(y / self.world_height * self.view_height)

    def draw(self, fly: Any, world: Any, telemetry: dict[str, Any]) -> None:
        pg = self.pg
        self.screen.fill((12, 17, 24))
        pg.draw.rect(self.screen, (22, 31, 42), (0, 0, self.view_width, self.view_height))
        tx, ty = self._point(world.threat.x, world.threat.y)
        radius = max(5, int(world.threat.radius / self.world_width * self.view_width))
        pg.draw.circle(self.screen, (215, 67, 64), (tx, ty), radius, 3)
        fx, fy = self._point(fly.x, fly.y)
        nose = (fx + int(20 * math.cos(fly.angle)), fy + int(20 * math.sin(fly.angle)))
        left = (fx + int(11 * math.cos(fly.angle + 2.5)), fy + int(11 * math.sin(fly.angle + 2.5)))
        right = (fx + int(11 * math.cos(fly.angle - 2.5)), fy + int(11 * math.sin(fly.angle - 2.5)))
        pg.draw.polygon(self.screen, (109, 220, 151), [nose, left, right])
        pg.draw.line(self.screen, (255, 130, 120), (fx, fy), (tx, ty), 1)
        pg.draw.rect(self.screen, (8, 12, 18), (self.view_width, 0, self.panel_width, self.view_height))

        lines = [
            ("FLYBRAIN", (109, 220, 151)),
            (f"Mode: {'DEMO graph' if telemetry['is_demo'] else 'MaleCNS v1.0'}", (245, 194, 107)),
            (f"Time: {telemetry['time_ms'] / 1000:8.3f} s", (220, 226, 234)),
            (f"Neurons: {telemetry['neurons']:,}", (220, 226, 234)),
            (f"Edges: {telemetry['edges']:,}", (220, 226, 234)),
            (f"Spiking now: {telemetry['spikes']:,}", (220, 226, 234)),
            (f"Active: {telemetry['active']:,}", (220, 226, 234)),
            ("", (255, 255, 255)),
            (f"Vision L: {telemetry['vision_l']:.2f}", (120, 180, 255)),
            (f"Vision R: {telemetry['vision_r']:.2f}", (120, 180, 255)),
            (f"Looming:  {telemetry['looming']:.2f}", (255, 130, 120)),
            ("", (255, 255, 255)),
            (f"Forward: {telemetry['forward']:.2f}", (109, 220, 151)),
            (f"Turn L:  {telemetry['turn_left']:.2f}", (109, 220, 151)),
            (f"Turn R:  {telemetry['turn_right']:.2f}", (109, 220, 151)),
            (f"Escape:  {telemetry['escape']:.2f}", (245, 194, 107)),
            (f"Manual:  {telemetry.get('manual_stimuli', '-')}", (245, 194, 107)),
            ("", (255, 255, 255)),
            ("CLICK fly  debug IDs", (145, 155, 168)),
            ("1/2 eyes  3/4/5 legs", (145, 155, 168)),
            ("6 head touch", (145, 155, 168)),
            ("SPACE  reset threat", (145, 155, 168)),
            ("ESC    quit", (145, 155, 168)),
        ]
        y = 24
        for value, color in lines:
            self.screen.blit(self.font.render(value, True, color), (self.view_width + 18, y))
            y += 27
        if self.debug:
            body_ids = telemetry.get("active_body_ids", [])
            body_label = "Spiking body IDs: " + (", ".join(map(str, body_ids)) if body_ids else "none")
            self.screen.blit(self.small.render(body_label, True, (245, 194, 107)), (14, 14))
            y += 4
            for name, activity in telemetry.get("brain_groups", {}).items():
                label = f"{name[:11]:11}"
                self.screen.blit(self.small.render(label, True, (190, 200, 210)), (self.view_width + 18, y))
                pg.draw.rect(self.screen, (43, 55, 68), (self.view_width + 126, y + 2, 150, 12))
                pg.draw.rect(self.screen, (92, 168, 255), (self.view_width + 126, y + 2, int(150 * activity), 12))
                y += 20
        pg.display.flip()

    def tick(self, fps: int = 60) -> float:
        return self.clock.tick(fps) / 1000.0

    def close(self) -> None:
        self.pg.quit()
