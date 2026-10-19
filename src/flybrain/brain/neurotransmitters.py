from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def transmitter_signs(
    neuron_ids: np.ndarray,
    metadata: pd.DataFrame,
    body_column: str,
    config_path: str | Path,
    mode: str = "neutral",
) -> np.ndarray:
    """Return per-presynaptic-neuron multipliers.

    Experimental signs are deliberately configurable approximations; transmitter
    identity alone does not establish a synapse's biological effect.
    """
    if mode == "neutral":
        return np.ones(len(neuron_ids), dtype=np.float32)
    if mode != "experimental":
        raise ValueError(f"Unknown neurotransmitter mode: {mode}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    mapping = {str(key).casefold(): float(value) for key, value in config.get("experimental_signs", {}).items()}
    nt_column = next(
        (name for name in ("consensus_nt", "ground_truth", "predicted_nt") if name in metadata.columns),
        None,
    )
    if nt_column is None:
        return np.full(len(neuron_ids), mapping.get("unknown", 1.0), dtype=np.float32)
    lookup = metadata.drop_duplicates(body_column).set_index(body_column)[nt_column]
    values = lookup.reindex(neuron_ids).fillna("unknown").astype(str).str.casefold()
    return values.map(lambda value: mapping.get(value, mapping.get("unknown", 1.0))).to_numpy(dtype=np.float32)

