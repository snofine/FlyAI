from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MotorCommand:
    forward: float = 0.0
    turn_left: float = 0.0
    turn_right: float = 0.0
    stop: float = 0.0
    escape: float = 0.0


class MotorDecoder:
    def __init__(self, roles: dict[str, np.ndarray], window_steps: int = 50, gain: float = 1.8) -> None:
        self.groups = {
            name: np.asarray(roles.get(name, []), dtype=np.int64)
            for name in ("forward", "turn_left", "turn_right", "stop", "escape")
        }
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(window_steps)))
        self.gain = float(gain)

    def update(self, spikes: np.ndarray) -> MotorCommand:
        activities = np.zeros(5, dtype=np.float32)
        for pos, group in enumerate(self.groups.values()):
            if len(group):
                activities[pos] = np.count_nonzero(spikes[group]) / len(group)
        self.history.append(activities)
        mean = np.mean(np.stack(self.history), axis=0) if self.history else activities
        # Convert sparse spike fraction to stable [0, 1] actuator activity.
        output = np.clip(mean * self.gain * len(self.history), 0.0, 1.0)
        return MotorCommand(*map(float, output))

