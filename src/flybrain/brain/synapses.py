from __future__ import annotations

import numpy as np


def normalize_weights(
    weights: np.ndarray,
    mode: str = "log1p",
    scale: float = 0.02,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if mode == "log1p":
        values = np.log1p(values)
    elif mode == "linear":
        pass
    elif mode == "sqrt":
        values = np.sqrt(values)
    else:
        raise ValueError(f"Unknown weight normalization: {mode}")
    return (values * np.float32(scale)).astype(np.float32, copy=False)

