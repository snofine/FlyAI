from __future__ import annotations

from dataclasses import dataclass
import math

from flybrain.body.fly import FlyBody


@dataclass(slots=True)
class Threat:
    x: float
    y: float
    radius: float = 8.0
    speed: float = 28.0
    active: bool = True


class World:
    def __init__(self, width: float = 1000.0, height: float = 700.0, threat_speed: float = 28.0) -> None:
        self.width, self.height = float(width), float(height)
        self.threat = Threat(width * 0.86, height * 0.5, speed=threat_speed)

    def update(self, fly: FlyBody, dt_s: float) -> None:
        threat = self.threat
        if not threat.active:
            return
        dx, dy = fly.x - threat.x, fly.y - threat.y
        distance = math.hypot(dx, dy)
        fly_radius = 12.0
        threat.radius = min(80.0, threat.radius + 3.5 * dt_s)
        minimum_distance = threat.radius + fly_radius
        if distance <= 1e-6:
            # Deterministic fallback if a snapshot or external edit places the
            # centres at exactly the same coordinate.
            threat.x = fly.x - math.cos(fly.angle) * minimum_distance
            threat.y = fly.y - math.sin(fly.angle) * minimum_distance
        else:
            travel = min(threat.speed * dt_s, max(0.0, distance - minimum_distance))
            threat.x += dx / distance * travel
            threat.y += dy / distance * travel
            # Growing looming objects may expand into the body even without
            # centre movement. Project the centre back to the contact boundary.
            next_dx, next_dy = threat.x - fly.x, threat.y - fly.y
            next_distance = math.hypot(next_dx, next_dy)
            if next_distance < minimum_distance:
                ux, uy = next_dx / max(next_distance, 1e-6), next_dy / max(next_distance, 1e-6)
                threat.x = fly.x + ux * minimum_distance
                threat.y = fly.y + uy * minimum_distance

    def contact_strength(self, fly: FlyBody) -> float:
        distance = math.hypot(self.threat.x - fly.x, self.threat.y - fly.y)
        contact_distance = self.threat.radius + 12.5
        if distance > contact_distance:
            return 0.0
        return min(1.0, max(0.1, (contact_distance - distance + 0.5) / 2.0))

    def reset_threat(self, fly: FlyBody) -> None:
        self.threat.x = min(self.width, fly.x + self.width * 0.4)
        self.threat.y = fly.y
        self.threat.radius = 8.0
        self.threat.active = True
