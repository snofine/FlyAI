from __future__ import annotations

import numpy as np


class StimulusEncoder:
    def __init__(
        self,
        neuron_count: int,
        looming_indices: np.ndarray,
        gain: float = 1.35,
        baseline_current: float = 0.0,
    ) -> None:
        self.neuron_count = neuron_count
        self.looming_indices = np.asarray(looming_indices, dtype=np.int64)
        self.gain = float(gain)
        self.baseline_current = float(baseline_current)

    def encode_looming(self, strength: float) -> np.ndarray:
        current = np.zeros(self.neuron_count, dtype=np.float32)
        if len(self.looming_indices) and strength > 0:
            current[self.looming_indices] = np.float32(
                self.baseline_current + strength * self.gain
            )
        return current
