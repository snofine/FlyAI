from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ExperimentLog:
    def __init__(self, directory: str | Path, manifest: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def save(self) -> Path:
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        path = self.directory / "steps.parquet"
        pd.DataFrame(self.rows).to_parquet(path, index=False)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return manifest, pd.read_parquet(path / "steps.parquet")

