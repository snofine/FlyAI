from __future__ import annotations

from dataclasses import dataclass
import math

from flybrain.motor.motor_decoder import MotorCommand


@dataclass(slots=True)
class FlyBody:
    x: float
    y: float
    angle: float = 0.0
    velocity: float = 0.0
    angular_velocity: float = 0.0
    max_speed: float = 150.0
    turn_rate: float = 3.5
    drag: float = 2.5

    def update(self, command: MotorCommand, dt_s: float, width: float, height: float) -> None:
        turn = command.turn_right - command.turn_left
        if command.escape > 0:
            turn += 0.35 * (1.0 if turn >= 0 else -1.0)
        self.angular_velocity = self.turn_rate * turn
        self.angle = (self.angle + self.angular_velocity * dt_s) % (2 * math.pi)
        drive = max(command.forward, command.escape)
        if command.stop > drive:
            drive = 0.0
        target_speed = self.max_speed * drive
        blend = min(1.0, (self.drag + 4.0 * command.escape) * dt_s)
        self.velocity += (target_speed - self.velocity) * blend
        self.x = min(width, max(0.0, self.x + math.cos(self.angle) * self.velocity * dt_s))
        self.y = min(height, max(0.0, self.y + math.sin(self.angle) * self.velocity * dt_s))

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "angle": self.angle, "velocity": self.velocity}

