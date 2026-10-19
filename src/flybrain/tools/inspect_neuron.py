from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.brain.connectome import Connectome


def metadata_value(value):
    """Convert Arrow/Pandas scalar or list columns into printable values."""
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size else None
    if isinstance(value, (list, tuple)):
        return list(value) if value else None
    missing = pd.isna(value)
    if isinstance(missing, (np.ndarray, list)):
        return value
    return None if bool(missing) else value


def inspect_neuron(directory: Path, body_id: int, top: int = 20) -> str:
    graph = Connectome.load(directory)
    index = graph.index_for_body(body_id)
    targets, outgoing_weights = graph.outgoing(index)
    incoming_sources: list[int] = []
    incoming_weights: list[float] = []
    for source in range(graph.neuron_count):
        row_targets, row_weights = graph.outgoing(source)
        matches = np.flatnonzero(row_targets == index)
        for match in matches:
            incoming_sources.append(source)
            incoming_weights.append(float(row_weights[match]))
    metadata = {}
    if graph.metadata_path:
        frame = pd.read_parquet(graph.metadata_path)
        body_column = next(name for name in ("body", "bodyId", "body_id", "id") if name in frame.columns)
        rows = frame.loc[frame[body_column] == body_id]
        if not rows.empty:
            metadata = {
                key: cleaned
                for key, value in rows.iloc[0].to_dict().items()
                if (cleaned := metadata_value(value)) is not None
            }
    order_out = np.argsort(np.abs(outgoing_weights))[::-1][:top]
    order_in = np.argsort(np.abs(incoming_weights))[::-1][:top]
    lines = [f"Body ID: {body_id}", f"Dense index: {index}"]
    for key in ("type", "cell_type", "class", "cell_class", "superclass", "super_class", "side", "somaSide", "consensus_nt", "predicted_nt"):
        if key in metadata:
            lines.append(f"{key}: {metadata[key]}")
    lines.extend([
        f"Incoming connections: {len(incoming_sources):,}",
        f"Outgoing connections: {len(targets):,}",
        f"Total normalized incoming weight: {sum(incoming_weights):.4f}",
        f"Total normalized outgoing weight: {float(np.sum(outgoing_weights)):.4f}",
        "", "Top inputs:",
    ])
    lines.extend(f"  {int(graph.neuron_ids[incoming_sources[i]])}: {incoming_weights[i]:.4f}" for i in order_in)
    lines.append("\nTop outputs:")
    lines.extend(f"  {int(graph.neuron_ids[targets[i]])}: {float(outgoing_weights[i]):.4f}" for i in order_out)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect one neuron in a processed connectome")
    parser.add_argument("body_id", type=int)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    print(inspect_neuron(args.data, args.body_id, args.top))


if __name__ == "__main__":
    main()
