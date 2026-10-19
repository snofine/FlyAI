from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome
from .neuron import LIFModel, NeuralModel, NeuronState


@dataclass(slots=True)
class BrainStep:
    step: int
    time_ms: float
    spike_count: int
    active_count: int
    spikes: np.ndarray


class BrainSimulator:
    def __init__(
        self,
        connectome: Connectome,
        model: NeuralModel | None = None,
        dt_ms: float = 1.0,
    ) -> None:
        self.connectome = connectome
        self.model = model or LIFModel()
        self.dt_ms = float(dt_ms)
        self.state = NeuronState.create(connectome.neuron_count)
        self.pending_input = np.zeros(connectome.neuron_count, dtype=np.float32)
        self.step_number = 0

    @property
    def time_ms(self) -> float:
        return self.step_number * self.dt_ms

    def inject(self, indices: np.ndarray, currents: float | np.ndarray) -> None:
        if len(indices):
            np.add.at(self.pending_input, np.asarray(indices, dtype=np.int64), currents)

    def _propagate(self, fired: np.ndarray) -> None:
        # Event-driven: touch outgoing rows only for neurons that fired. Rows
        # are gathered in bounded vectorized batches, avoiding a Python loop
        # per neuron while keeping temporary memory independent of graph size.
        if not len(fired):
            return
        starts = np.asarray(self.connectome.indptr[fired], dtype=np.int64)
        ends = np.asarray(self.connectome.indptr[fired + 1], dtype=np.int64)
        counts = ends - starts
        populated = counts > 0
        starts, counts = starts[populated], counts[populated]
        if not len(starts):
            return
        cumulative = np.cumsum(counts)
        row_start = 0
        edge_budget = 1_000_000
        while row_start < len(starts):
            previous = int(cumulative[row_start - 1]) if row_start else 0
            row_end = int(np.searchsorted(cumulative, previous + edge_budget, side="right"))
            row_end = max(row_start + 1, row_end)
            chunk_starts = starts[row_start:row_end]
            chunk_counts = counts[row_start:row_end]
            local_prefix = np.cumsum(chunk_counts)
            total = int(local_prefix[-1])
            row_offsets = np.repeat(
                chunk_starts - np.concatenate(([0], local_prefix[:-1])),
                chunk_counts,
            )
            positions = np.arange(total, dtype=np.int64) + row_offsets
            targets = self.connectome.indices[positions]
            values = self.connectome.weights[positions]
            contribution = np.bincount(
                targets, weights=values, minlength=self.connectome.neuron_count
            ).astype(np.float32, copy=False)
            self.pending_input += contribution
            row_start = row_end

    def step(self, external_input: np.ndarray | None = None) -> BrainStep:
        current = self.pending_input
        self.pending_input = np.zeros_like(current)
        if external_input is not None:
            current = current + np.asarray(external_input, dtype=np.float32)
        spikes = self.model.step(self.state, current, self.dt_ms, self.step_number)
        fired = np.flatnonzero(spikes)
        self._propagate(fired)
        self.step_number += 1
        return BrainStep(
            step=self.step_number,
            time_ms=self.time_ms,
            spike_count=int(len(fired)),
            active_count=int(np.count_nonzero(np.abs(self.state.voltage) > 0.05) + len(fired)),
            spikes=spikes.copy(),
        )

    def save_state(self, path: str) -> None:
        np.savez_compressed(
            path,
            voltage=self.state.voltage,
            refractory=self.state.refractory,
            spikes=self.state.spikes,
            last_spike_step=self.state.last_spike_step,
            pending_input=self.pending_input,
            step_number=np.asarray(self.step_number, dtype=np.int64),
        )

    def load_state(self, path: str) -> None:
        with np.load(path) as data:
            if len(data["voltage"]) != self.connectome.neuron_count:
                raise ValueError("Snapshot neuron count does not match the connectome")
            self.state.voltage[:] = data["voltage"]
            self.state.refractory[:] = data["refractory"]
            self.state.spikes[:] = data["spikes"]
            self.state.last_spike_step[:] = data["last_spike_step"]
            self.pending_input[:] = data["pending_input"]
            self.step_number = int(data["step_number"])
