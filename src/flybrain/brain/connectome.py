from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synapses import normalize_weights


@dataclass(slots=True)
class Connectome:
    neuron_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    roles: dict[str, np.ndarray]
    metadata_path: Path | None = None
    is_demo: bool = False

    def __post_init__(self) -> None:
        n = len(self.neuron_ids)
        if len(self.indptr) != n + 1:
            raise ValueError("CSR indptr length must equal neuron count + 1")
        if len(self.indices) != len(self.weights) or int(self.indptr[-1]) != len(self.indices):
            raise ValueError("Inconsistent CSR arrays")
        if len(self.indices) and (self.indices.min() < 0 or self.indices.max() >= n):
            raise ValueError("CSR target index outside neuron range")
        self.roles = {key: np.asarray(value, dtype=np.int64) for key, value in self.roles.items()}

    @property
    def neuron_count(self) -> int:
        return int(len(self.neuron_ids))

    @property
    def edge_count(self) -> int:
        return int(len(self.indices))

    def index_for_body(self, body_id: int) -> int:
        matches = np.flatnonzero(self.neuron_ids == int(body_id))
        if not len(matches):
            raise KeyError(f"Unknown body ID: {body_id}")
        return int(matches[0])

    def outgoing(self, neuron_index: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.indptr[neuron_index]), int(self.indptr[neuron_index + 1])
        return self.indices[start:end], self.weights[start:end]

    @classmethod
    def load(cls, directory: str | Path, mmap_mode: str | None = "r") -> "Connectome":
        path = Path(directory)
        required = ["neuron_ids.npy", "graph_indptr.npy", "graph_indices.npy", "graph_weights.npy"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            raise FileNotFoundError(f"Processed connectome is incomplete: {', '.join(missing)}")
        role_path = path / "neuron_roles.json"
        raw_roles = json.loads(role_path.read_text(encoding="utf-8")) if role_path.exists() else {}
        return cls(
            neuron_ids=np.load(path / "neuron_ids.npy", mmap_mode=mmap_mode),
            indptr=np.load(path / "graph_indptr.npy", mmap_mode=mmap_mode),
            indices=np.load(path / "graph_indices.npy", mmap_mode=mmap_mode),
            weights=np.load(path / "graph_weights.npy", mmap_mode=mmap_mode),
            roles={name: np.asarray(values, dtype=np.int64) for name, values in raw_roles.items()},
            metadata_path=(path / "neuron_metadata.parquet") if (path / "neuron_metadata.parquet").exists() else None,
        )

    @classmethod
    def demo(cls, scale: float = 0.22) -> "Connectome":
        """Small explicit sensory→interneuron→descending circuit for smoke tests."""
        ids = np.array([9000001, 9000002, 9000010, 9000011, 9000020, 9000021, 9000030, 9000031], dtype=np.int64)
        edges = [
            # Strong demonstrator edges ensure one sensory event can be traced
            # through every stage. They are not presented as MaleCNS weights.
            (0, 2, 120), (1, 3, 120),
            (2, 4, 120), (3, 5, 120),
            (4, 6, 120), (5, 7, 120),
            (4, 7, 5), (5, 6, 5),
        ]
        indptr = np.zeros(len(ids) + 1, dtype=np.int64)
        for source, _, _ in edges:
            indptr[source + 1] += 1
        np.cumsum(indptr, out=indptr)
        cursor = indptr[:-1].copy()
        indices = np.empty(len(edges), dtype=np.int32)
        raw = np.empty(len(edges), dtype=np.float32)
        for source, target, weight in edges:
            slot = cursor[source]
            indices[slot], raw[slot] = target, weight
            cursor[source] += 1
        return cls(
            ids,
            indptr,
            indices,
            normalize_weights(raw, "log1p", scale),
            {
                "visual": np.array([0, 1]),
                "looming": np.array([0, 1]),
                "central": np.array([2, 3]),
                "descending": np.array([4, 5]),
                "vnc": np.array([6, 7]),
                "motor": np.array([6, 7]),
                "escape": np.array([6, 7]),
                "forward": np.array([6, 7]),
                "turn_left": np.array([6]),
                "turn_right": np.array([7]),
                "stop": np.array([], dtype=np.int64),
            },
            is_demo=True,
        )
