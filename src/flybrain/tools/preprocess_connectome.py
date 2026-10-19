from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from flybrain.brain.synapses import normalize_weights
from flybrain.brain.neurotransmitters import transmitter_signs
from flybrain.config import PROJECT_ROOT


EDGE_ALIASES = {
    "source": ("body_pre", "pre", "source", "bodyId_pre"),
    "target": ("body_post", "post", "target", "bodyId_post"),
    "weight": ("weight", "count", "synapse_count"),
}
BODY_ALIASES = ("body", "bodyId", "body_id", "id")


def find_column(columns: Iterable[str], aliases: Iterable[str], label: str) -> str:
    available = set(columns)
    for candidate in aliases:
        if candidate in available:
            return candidate
    raise ValueError(f"Cannot find {label} column. Available: {sorted(available)}")


def read_batches(path: Path, columns: list[str]):
    scanner = ds.dataset(path, format="feather").scanner(columns=columns, batch_size=1_000_000)
    yield from scanner.to_batches()


def annotations_text(frame: pd.DataFrame) -> pd.Series:
    preferred = [
        "type", "instance", "cell_type", "class", "cell_class", "super_class", "superclass",
        "subclass", "flow", "nerve", "side", "somaSide", "body_part_sensory",
        "body_part_effector", "status",
    ]
    present = [name for name in preferred if name in frame.columns]
    if not present:
        present = [name for name in frame.columns if frame[name].dtype == object]
    return frame[present].fillna("").astype(str).agg(" ".join, axis=1)


def resolve_group(
    frame: pd.DataFrame,
    body_col: str,
    spec: dict,
    fallback: str | None = None,
    text: pd.Series | None = None,
) -> np.ndarray:
    explicit = {int(value) for value in spec.get("body_ids", [])}
    patterns = list(spec.get("type_patterns", []))
    if fallback:
        patterns.append(fallback)
    if text is None:
        text = annotations_text(frame)
    mask = frame[body_col].isin(explicit)
    if patterns:
        pattern = "|".join(f"(?:{value})" for value in patterns)
        mask = mask | text.str.contains(pattern, case=False, regex=True, na=False)
    where = spec.get("where", {})
    if where:
        filtered = pd.Series(True, index=frame.index)
        for field, condition in where.items():
            if field.endswith("_regex"):
                column = field.removesuffix("_regex")
                if column not in frame.columns:
                    filtered &= False
                else:
                    filtered &= frame[column].fillna("").astype(str).str.contains(
                        str(condition), case=False, regex=True, na=False
                    )
            else:
                values = condition if isinstance(condition, list) else [condition]
                filtered &= frame[field].isin(values) if field in frame.columns else False
        mask = mask | filtered
    return frame.loc[mask, body_col].astype(np.int64).drop_duplicates().head(int(spec.get("max_neurons", 100000))).to_numpy()


def body_ids_to_indices(sorted_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_ids, body_ids)
    valid = (positions < len(sorted_ids)) & (sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == body_ids)
    return positions[valid].astype(np.int64)


def infer_roles(annotations: pd.DataFrame, body_col: str, config_dir: Path) -> dict[str, np.ndarray]:
    sensory = yaml.safe_load((config_dir / "sensory_neurons.yaml").read_text(encoding="utf-8"))
    motor = yaml.safe_load((config_dir / "motor_neurons.yaml").read_text(encoding="utf-8"))
    text = annotations_text(annotations)
    roles_body: dict[str, np.ndarray] = {}
    roles_body["looming"] = resolve_group(annotations, body_col, sensory.get("looming", {}), text=text)
    for name, spec in sensory.get("manual_stimuli", {}).items():
        roles_body[f"stimulus_{name}"] = resolve_group(annotations, body_col, spec, text=text)
    roles_body["visual"] = resolve_group(annotations, body_col, {}, r"visual|optic|photoreceptor|LPLC|\bLC\d", text)
    roles_body["descending"] = resolve_group(annotations, body_col, {}, r"descending|\bDN[a-zA-Z0-9]", text)
    roles_body["motor"] = resolve_group(annotations, body_col, {}, r"motor|motoneuron", text)
    roles_body["vnc"] = resolve_group(annotations, body_col, {}, r"VNC|ventral nerve", text)
    roles_body["central"] = resolve_group(annotations, body_col, {}, r"central brain|interneuron|intrinsic", text)
    for name, spec in motor.items():
        roles_body[name] = resolve_group(annotations, body_col, spec, text=text)
    return roles_body


def preprocess(
    raw_dir: Path,
    output_dir: Path,
    minimum_weight: int = 1,
    normalization: str = "log1p",
    weight_scale: float = 0.02,
    config_dir: Path | None = None,
    neurotransmitter_mode: str = "neutral",
    include_unclassified: bool = False,
) -> dict[str, object]:
    annotations_path = raw_dir / "annotations.feather"
    nt_path = raw_dir / "neurotransmitters.feather"
    edges_path = raw_dir / "connectome-weights.feather"
    for path in (annotations_path, nt_path, edges_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run flybrain-download first")

    annotations = pd.read_feather(annotations_path)
    body_col = find_column(annotations.columns, BODY_ALIASES, "body ID")
    annotations = annotations.drop_duplicates(body_col).copy()
    # MaleCNS v1.0 contains additional annotated segments (glia, orphans,
    # unimportant fragments). An assigned superclass is the release's robust
    # neuron criterion and yields 166,700 nodes for the published v1.0 files.
    superclass_col = next((name for name in ("superclass", "super_class") if name in annotations.columns), None)
    if superclass_col and not include_unclassified:
        annotations = annotations.loc[annotations[superclass_col].notna()].copy()
    neuron_ids = np.sort(annotations[body_col].astype(np.int64).to_numpy())
    if not len(neuron_ids):
        raise ValueError("No annotated neuron IDs found")

    nt = pd.read_feather(nt_path)
    nt_body = find_column(nt.columns, BODY_ALIASES, "neurotransmitter body ID")
    nt_columns = [name for name in ("consensus_nt", "predicted_nt", "ground_truth") if name in nt.columns]
    if nt_columns:
        nt = nt[[nt_body, *nt_columns]].drop_duplicates(nt_body)
        annotations = annotations.merge(nt, how="left", left_on=body_col, right_on=nt_body, suffixes=("", "_nt"))

    schema_names = ds.dataset(edges_path, format="feather").schema.names
    source_col = find_column(schema_names, EDGE_ALIASES["source"], "presynaptic")
    target_col = find_column(schema_names, EDGE_ALIASES["target"], "postsynaptic")
    weight_col = find_column(schema_names, EDGE_ALIASES["weight"], "weight")
    columns = [source_col, target_col, weight_col]

    degree = np.zeros(len(neuron_ids), dtype=np.int64)
    edge_count = 0
    print(f"Pass 1/2: counting valid edges for {len(neuron_ids):,} annotated neurons")
    for batch in read_batches(edges_path, columns):
        source = batch.column(source_col).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        target = batch.column(target_col).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        weight = batch.column(weight_col).to_numpy(zero_copy_only=False)
        source_idx = np.searchsorted(neuron_ids, source)
        target_idx = np.searchsorted(neuron_ids, target)
        valid = (
            (source_idx < len(neuron_ids)) & (target_idx < len(neuron_ids)) &
            (neuron_ids[np.minimum(source_idx, len(neuron_ids) - 1)] == source) &
            (neuron_ids[np.minimum(target_idx, len(neuron_ids) - 1)] == target) &
            (weight >= minimum_weight)
        )
        np.add.at(degree, source_idx[valid], 1)
        edge_count += int(valid.sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "neuron_ids.npy", neuron_ids)
    indptr = np.empty(len(neuron_ids) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    np.save(output_dir / "graph_indptr.npy", indptr)
    indices = np.lib.format.open_memmap(output_dir / "graph_indices.npy", mode="w+", dtype=np.int32, shape=(edge_count,))
    weights = np.lib.format.open_memmap(output_dir / "graph_weights.npy", mode="w+", dtype=np.float32, shape=(edge_count,))
    cursor = indptr[:-1].copy()
    resolved_config_dir = config_dir or PROJECT_ROOT / "config"
    nt_signs = transmitter_signs(
        neuron_ids,
        annotations,
        body_col,
        resolved_config_dir / "neurotransmitters.yaml",
        neurotransmitter_mode,
    )

    print(f"Pass 2/2: writing {edge_count:,} CSR edges")
    for batch in read_batches(edges_path, columns):
        source = batch.column(source_col).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        target = batch.column(target_col).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        raw_weight = batch.column(weight_col).to_numpy(zero_copy_only=False)
        source_idx = np.searchsorted(neuron_ids, source)
        target_idx = np.searchsorted(neuron_ids, target)
        valid = (
            (source_idx < len(neuron_ids)) & (target_idx < len(neuron_ids)) &
            (neuron_ids[np.minimum(source_idx, len(neuron_ids) - 1)] == source) &
            (neuron_ids[np.minimum(target_idx, len(neuron_ids) - 1)] == target) &
            (raw_weight >= minimum_weight)
        )
        valid_source = source_idx[valid]
        if not len(valid_source):
            continue
        valid_target = target_idx[valid]
        valid_weight = raw_weight[valid]
        # Batch-pack CSR rows without a Python loop over potentially 151M edges.
        order = np.argsort(valid_source, kind="stable")
        sorted_source = valid_source[order]
        starts = np.empty(len(sorted_source), dtype=np.bool_)
        starts[0] = True
        starts[1:] = sorted_source[1:] != sorted_source[:-1]
        start_index = np.maximum.accumulate(np.where(starts, np.arange(len(sorted_source)), 0))
        rank_in_row = np.arange(len(sorted_source)) - start_index
        slots = cursor[sorted_source] + rank_in_row
        indices[slots] = valid_target[order]
        normalized = normalize_weights(valid_weight[order], normalization, weight_scale)
        weights[slots] = normalized * nt_signs[sorted_source]
        unique_source, counts = np.unique(sorted_source, return_counts=True)
        cursor[unique_source] += counts
    indices.flush()
    weights.flush()

    annotations.sort_values(body_col).to_parquet(output_dir / "neuron_metadata.parquet", index=False)
    roles_body = infer_roles(annotations, body_col, resolved_config_dir)
    roles = {name: body_ids_to_indices(neuron_ids, values).tolist() for name, values in roles_body.items()}
    (output_dir / "neuron_roles.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")
    summary = {
        "neurons": int(len(neuron_ids)),
        "edges": edge_count,
        "minimum_weight": int(minimum_weight),
        "normalization": normalization,
        "weight_scale": float(weight_scale),
        "neurotransmitter_mode": neurotransmitter_mode,
        "inhibitory_or_modulatory_neurons": int(np.count_nonzero(nt_signs <= 0)),
        **{f"role_{key}": len(value) for key, value in roles.items()},
    }
    (output_dir / "build_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert MaleCNS Feather tables to memory-mapped CSR arrays")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--minimum-weight", type=int, default=1)
    parser.add_argument("--normalization", choices=("log1p", "linear", "sqrt"), default="log1p")
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--nt-mode", choices=("neutral", "experimental"), default="neutral")
    parser.add_argument("--include-unclassified", action="store_true", help="Keep glia/orphans/unclassified segments")
    args = parser.parse_args(argv)
    print(json.dumps(preprocess(
        args.raw, args.output, args.minimum_weight, args.normalization,
        args.weight_scale, neurotransmitter_mode=args.nt_mode,
        include_unclassified=args.include_unclassified,
    ), indent=2))


if __name__ == "__main__":
    main()
