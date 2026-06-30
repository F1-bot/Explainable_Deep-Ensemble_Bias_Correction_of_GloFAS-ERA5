"""Canonical schema of the modelling table shared by every component.

A *modelling table* is a tidy ``pandas.DataFrame`` with one row per
(gauge, period) and the following guaranteed columns::

    code     str        gauge identifier
    basin    str        basin label (spatial CV grouping)
    domain   str        "core" | "transfer"
    scale    str        "decadal" | "daily"
    date     datetime   period representative date
    q_obs    float      observed discharge            [m3 s-1]
    q_glofas float      raw GloFAS-ERA5 discharge      [m3 s-1]
    log_residual float  ML target = log(q_obs+EPS) - log(q_glofas+EPS)

plus any number of engineered dynamic and static feature columns, and an
optional ``regime`` label.  Feature columns are discovered at run time so the
pipeline never hard-codes a feature list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPS

ID_COLS: list[str] = ["code", "basin", "domain", "scale", "date"]
OBS_COL = "q_obs"
SIM_COL = "q_glofas"
TARGET_COL = "log_residual"
REGIME_COL = "regime"
WEIGHT_COL = "sample_weight"
PRED_COL = "q_pred"

#: columns that are never model inputs
NON_FEATURE: set[str] = set(ID_COLS) | {
    OBS_COL, SIM_COL, TARGET_COL, REGIME_COL, WEIGHT_COL, PRED_COL,
}


def make_target(q_obs, q_glofas) -> np.ndarray:
    """Log-space multiplicative-residual target."""
    return np.log(np.asarray(q_obs, float) + EPS) - np.log(np.asarray(q_glofas, float) + EPS)


def back_transform(q_glofas, log_residual) -> np.ndarray:
    """Reconstruct corrected discharge from GloFAS and a predicted log-residual."""
    q = np.exp(np.log(np.asarray(q_glofas, float) + EPS) + np.asarray(log_residual, float)) - EPS
    return np.clip(q, 0.0, None)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric model-input columns (everything that is not id/target)."""
    cols = []
    for c in df.columns:
        if c in NON_FEATURE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def static_feature_columns(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    """Features that are constant within each gauge (catchment attributes)."""
    features = features or feature_columns(df)
    if df["code"].nunique() <= 1:
        return []
    nun = df.groupby("code")[features].nunique(dropna=False)
    return [c for c in features if int(nun[c].max()) <= 1]


def dynamic_feature_columns(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    features = features or feature_columns(df)
    static = set(static_feature_columns(df, features))
    return [c for c in features if c not in static]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the mandatory columns exist and have sane dtypes; return df."""
    missing = [c for c in ID_COLS + [OBS_COL, SIM_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"modelling table missing columns: {missing}")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        raise ValueError("'date' must be datetime64")
    if TARGET_COL not in df.columns:
        df = df.copy()
        df[TARGET_COL] = make_target(df[OBS_COL], df[SIM_COL])
    return df
