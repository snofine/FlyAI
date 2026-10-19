from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class EyeSignals:
    brightness_left: float = 0.0
    brightness_right: float = 0.0
    looming: float = 0.0
    object_bearing: float = 0.0
    apparent_size: float = 0.0


class VisionSensor:
    def __init__(self, eye_separation: float = 0.16) -> None:
        self.eye_separation = eye_separation
        self._last_size: float | None = None

    def sense(
        self,
        fly_x: float,
        fly_y: float,
        fly_angle: float,
        object_x: float,
        object_y: float,
        object_radius: float,
        dt_s: float,
    ) -> EyeSignals:
        dx, dy = object_x - fly_x, object_y - fly_y
        distance = max(1e-4, math.hypot(dx, dy))
        bearing = math.atan2(dy, dx) - fly_angle
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        size = min(1.0, 2.0 * math.atan2(object_radius, distance) / math.pi)
        growth = 0.0 if self._last_size is None else max(0.0, (size - self._last_size) / max(dt_s, 1e-6))
        self._last_size = size
        # Compound eyes cover most of the azimuth. Frontal stimuli dominate,
        # but an approaching object behind the fly must not become invisible.
        in_view = 0.18 + 0.82 * max(0.0, math.cos(bearing))
        # Angular-size derivative, amplified into a normalized salience signal.
        # This is sensory transduction only; it never issues a motor command.
        looming = min(1.0, growth * 60.0) * in_view
        left = size * in_view * (1.0 + max(0.0, -math.sin(bearing)))
        right = size * in_view * (1.0 + max(0.0, math.sin(bearing)))
        return EyeSignals(
            brightness_left=min(1.0, left),
            brightness_right=min(1.0, right),
            looming=looming,
            object_bearing=bearing,
            apparent_size=size,
        )
