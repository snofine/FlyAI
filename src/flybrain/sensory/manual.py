from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ActiveStimulus:
    remaining_steps: int
    strength: float


class ManualStimulusController:
    """Inject user-triggered current into annotated sensory neuron groups."""

    def __init__(
        self,
        neuron_count: int,
        roles: dict[str, np.ndarray],
        channel_config: dict,
        dt_ms: float,
    ) -> None:
        self.neuron_count = int(neuron_count)
        self.dt_ms = float(dt_ms)
        self.channels = {
            name: np.asarray(roles.get(f"stimulus_{name}", []), dtype=np.int64)
            for name in channel_config
        }
        self.currents = {
            name: float(spec.get("current", 0.35)) for name, spec in channel_config.items()
        }
        self.active: dict[str, ActiveStimulus] = {}

    @property
    def available_channels(self) -> tuple[str, ...]:
        return tuple(name for name, indices in self.channels.items() if len(indices))

    def trigger(self, name: str, duration_ms: float = 180.0, strength: float = 1.0) -> int:
        if name not in self.channels:
            raise KeyError(f"Unknown stimulus {name!r}; available: {', '.join(self.available_channels)}")
        if not len(self.channels[name]):
            raise ValueError(f"Stimulus {name!r} has no mapped neurons; rebuild the processed graph")
        steps = max(1, int(round(duration_ms / self.dt_ms)))
        current = self.active.get(name)
        if current is None:
            self.active[name] = ActiveStimulus(steps, float(strength))
        else:
            current.remaining_steps = max(current.remaining_steps, steps)
            current.strength = max(current.strength, float(strength))
        return int(len(self.channels[name]))

    def encode(self) -> np.ndarray:
        output = np.zeros(self.neuron_count, dtype=np.float32)
        expired: list[str] = []
        for name, stimulus in self.active.items():
            indices = self.channels[name]
            output[indices] += np.float32(self.currents[name] * stimulus.strength)
            stimulus.remaining_steps -= 1
            if stimulus.remaining_steps <= 0:
                expired.append(name)
        for name in expired:
            del self.active[name]
        return output

    def active_names(self) -> tuple[str, ...]:
        return tuple(self.active)

