from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NeuronState:
    voltage: np.ndarray
    refractory: np.ndarray
    spikes: np.ndarray
    last_spike_step: np.ndarray

    @classmethod
    def create(cls, count: int) -> "NeuronState":
        return cls(
            voltage=np.zeros(count, dtype=np.float32),
            refractory=np.zeros(count, dtype=np.int16),
            spikes=np.zeros(count, dtype=np.bool_),
            last_spike_step=np.full(count, -1, dtype=np.int64),
        )


class NeuralModel(ABC):
    @abstractmethod
    def step(
        self, state: NeuronState, inputs: np.ndarray, dt_ms: float, step_number: int
    ) -> np.ndarray:
        """Advance every neuron and return the current spike mask."""


class LIFModel(NeuralModel):
    def __init__(
        self,
        tau_ms: float = 20.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        refractory_ms: float = 2.0,
    ) -> None:
        if tau_ms <= 0 or threshold <= reset or refractory_ms < 0:
            raise ValueError("Invalid LIF parameters")
        self.tau_ms = float(tau_ms)
        self.threshold = np.float32(threshold)
        self.reset = np.float32(reset)
        self.refractory_ms = float(refractory_ms)

    def step(
        self, state: NeuronState, inputs: np.ndarray, dt_ms: float, step_number: int
    ) -> np.ndarray:
        refractory_steps = max(0, int(round(self.refractory_ms / dt_ms)))
        resting = state.refractory == 0
        state.voltage[resting] += np.float32(dt_ms) * (
            -state.voltage[resting] / np.float32(self.tau_ms) + inputs[resting]
        )
        state.spikes[:] = resting & (state.voltage >= self.threshold)
        state.voltage[state.spikes] = self.reset
        state.last_spike_step[state.spikes] = step_number
        state.refractory[state.refractory > 0] -= 1
        if refractory_steps:
            state.refractory[state.spikes] = refractory_steps
        return state.spikes

