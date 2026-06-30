"""Cross-validation orchestration and result aggregation.

Runs any :class:`~sbc.models.base.BaseCorrector` through the full leakage-safe
validation matrix — per-gauge temporal holdout, leave-one-basin-out (LOBO) and
prediction-in-ungauged-regions (PUR) — and reports per-gauge skill of the
corrected discharge against both the observations and the raw GloFAS benchmark.

Models are supplied as zero-argument *factories* so a fresh, untrained instance
is fitted inside every fold (no state leaks between folds).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..schemas import OBS_COL, PRED_COL, SIM_COL, back_transform
from ..utils import get_logger
from . import metrics as M
from .splits import pur_split, spatial_folds, temporal_split

log = get_logger(__name__)

ModelFactory = Callable[[], "object"]
_METRIC_KEYS = ("kge", "kge_ss", "nse", "lognse", "pbias",
                "kge_r", "kge_beta", "kge_gamma", "peak_timing_err", "crps")


def _per_gauge(model, test: pd.DataFrame, label: str, split: str, fold: str
               ) -> pd.DataFrame:
    """Per-gauge metrics of the corrected series vs obs and vs raw GloFAS."""
    test = test.reset_index(drop=True)
    resid = np.asarray(model.predict_residual(test), float)
    pred = back_transform(test[SIM_COL].to_numpy(float), resid)
    work = test.assign(**{PRED_COL: pred})

    # optional probabilistic skill (CRPS) for models exposing quantiles
    crps_by_code = {}
    if getattr(model, "is_probabilistic", False):
        try:
            q = model.predict_discharge_quantiles(test, (0.05, 0.25, 0.5, 0.75, 0.95))
            work = work.assign(_crps_ens=[None] * len(work))
            for code, idx in work.groupby("code").groups.items():
                ii = np.asarray(idx)
                crps_by_code[code] = M.crps_ensemble(
                    work.loc[ii, OBS_COL].to_numpy(float), q[ii])
        except Exception as exc:  # pragma: no cover
            log.debug("CRPS skipped for %s: %s", label, exc)

    rows = []
    for code, g in work.groupby("code"):
        corr = M.evaluate(g[OBS_COL].values, g[PRED_COL].values, g["date"].values)
        raw = M.evaluate(g[OBS_COL].values, g[SIM_COL].values, g["date"].values)
        row = {"model": label, "split": split, "fold": fold, "code": code,
               "basin": g["basin"].iloc[0], "domain": g["domain"].iloc[0],
               "n": int(corr["n"])}
        for k in _METRIC_KEYS:
            row[k] = corr.get(k, np.nan)
            row[f"{k}_raw"] = raw.get(k, np.nan)
        row["crps"] = crps_by_code.get(code, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def run_split(make_model: ModelFactory, train: pd.DataFrame, test: pd.DataFrame,
              label: str, split: str, fold: str = "") -> pd.DataFrame:
    model = make_model()
    model.fit(train)
    return _per_gauge(model, test, label, split, fold)


def run_matrix(make_model: ModelFactory, df: pd.DataFrame, label: str,
               temporal: bool = True, lobo: bool = True, pur: bool = True,
               test_frac: float = 0.3) -> pd.DataFrame:
    """Evaluate one model across the temporal / LOBO / PUR matrix."""
    df = df.reset_index(drop=True)
    out: list[pd.DataFrame] = []
    if temporal:
        tr, te = temporal_split(df, test_frac)
        if tr.any() and te.any():
            out.append(run_split(make_model, df[tr], df[te], label, "temporal"))
    if lobo:
        for name, tr, te in spatial_folds(df):
            if tr.any() and te.any():
                out.append(run_split(make_model, df[tr], df[te], label, "lobo", name))
    if pur:
        tr, te = pur_split(df)
        if tr.any() and te.any():
            out.append(run_split(make_model, df[tr], df[te], label, "pur"))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not res.empty:
        med = res.groupby("split")["kge"].median().round(3).to_dict()
        log.info("%-14s median KGE' by split: %s", label, med)
    return res


def compare(model_factories: dict[str, ModelFactory], df: pd.DataFrame,
            **kwargs) -> pd.DataFrame:
    """Run several models through the matrix; return the concatenated per-gauge table."""
    frames = []
    for name, mk in model_factories.items():
        try:
            frames.append(run_matrix(mk, df, name, **kwargs))
        except Exception as exc:  # pragma: no cover
            log.warning("model %s failed: %s", name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Median per-(model, split) skill, plus the improvement over raw GloFAS."""
    if results.empty:
        return results
    r = results.copy()
    r["d_kge"] = r["kge"] - r["kge_raw"]
    r["d_nse"] = r["nse"] - r["nse_raw"]
    agg = (r.groupby(["model", "split"])
           .agg(n_gauges=("code", "nunique"),
                kge_raw=("kge_raw", "median"),
                kge=("kge", "median"),
                d_kge=("d_kge", "median"),
                nse=("nse", "median"),
                pbias_raw=("pbias_raw", "median"),
                pbias=("pbias", "median"),
                peak_timing_err=("peak_timing_err", "median"),
                crps=("crps", "median"))
           .reset_index())
    return agg.round(3)
