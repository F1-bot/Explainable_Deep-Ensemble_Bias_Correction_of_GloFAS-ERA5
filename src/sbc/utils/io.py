"""Lightweight IO helpers (parquet-first, CSV mirror for inspection)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(df: pd.DataFrame, path: str | Path, csv_mirror: bool = False) -> Path:
    """Save a dataframe to parquet; optionally also write a CSV mirror."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    if csv_mirror:
        df.to_csv(path.with_suffix(".csv"), index=False)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
