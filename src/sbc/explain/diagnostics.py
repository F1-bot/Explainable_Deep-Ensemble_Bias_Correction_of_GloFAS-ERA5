"""Deeper hydrological diagnostics for the GloFAS-ERA5 bias-correction results.

A single headline KGE' per (model, split) hides *where* and *when* the learned
correction helps.  This module turns the existing per-gauge result tables and the
assembled modelling table into the process-resolved diagnostics a hydrology
reviewer expects, so the paper can show not just *how much* skill improves but
*which part of the hydrograph* it improves:

``per_regime_skill``
    Corrected-vs-raw KGE' / PBIAS / NSE stratified by the five hydrological
    regimes of :mod:`sbc.features.regimes` (accumulation, melt-freshet,
    rain-on-snow, glacier-melt, recession) -- does the correction fix the damped
    freshet and the rain-on-snow floods, not just the annual volume?
``seasonal_skill``
    The same skill resolved by calendar month or by Central-Asian decade
    (day 1-10 / 11-20 / 21-end), exposing the seasonal signature of the bias.
``fdc_segment_table``
    Percent flow-volume bias in the very-high / high / mid / low segments of the
    flow-duration curve, i.e. the FHV/FMS/FLV story as a compact table.
``peak_timing_distribution``
    The full distribution (not just the median) of the annual peak-timing error,
    per model, read straight from the real per-gauge table -- this one reflects
    the actual model zoo including the flagship.
``plot_example_hydrographs``
    Observed vs raw-GloFAS vs bias-corrected time series for a few illustrative
    gauges, with the temporal-holdout boundary marked.

The skill helpers are pure functions over a tidy modelling table (one row per
gauge/period carrying ``q_obs``, ``q_glofas`` and -- for the corrected curves --
a prediction column or a supplied ``preds`` array); they reuse
:mod:`sbc.validation.metrics` and never hard-code a feature list.

``build_all`` is the cheap entry point.  ``peak_timing_distribution`` is read
directly from ``results/tables/per_gauge_<tag>.parquet`` (the real model zoo,
flagship included).  The time-series diagnostics need a per-timestep corrected
series, which the per-gauge table does not store; rather than re-run the ~4 h
GPU validation, ``build_all`` reconstructs a *leakage-safe reference correction*
(empirical quantile mapping fitted on the temporal-train split only) from the
assembled modelling table and evaluates it on the held-out test period.  These
figures therefore illustrate the *shape* of the correction across regimes /
seasons / FDC segments; the headline per-gauge skill of the full model zoo stays
in the per-gauge tables.  Pass ``corrector=None`` together with a ``model_table``
that already carries a ``q_pred`` column to diagnose any other model's output.

All plotting is Agg-safe (no ``plt.show``); tables go to ``results/tables`` and
figures to ``results/figures`` with a ``diag_`` prefix.  Heavy / optional imports
(matplotlib, models) are kept inside the functions that need them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import PATHS
from ..schemas import OBS_COL, PRED_COL, SIM_COL, REGIME_COL
from ..utils import get_logger
from ..validation.metrics import kge_prime, lognse, nse, pbias

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #
#: flow-duration-curve segments as [low, high) exceedance-probability bands of the
#: *observed* flow (the last band is closed on the right).  Mirrors the FHV / FMS
#: / FLV split used in :mod:`sbc.validation.metrics`.
FDC_SEGMENTS: tuple[tuple[str, float, float], ...] = (
    ("very_high", 0.00, 0.02),
    ("high", 0.02, 0.20),
    ("mid", 0.20, 0.70),
    ("low", 0.70, 1.00),
)

#: preference order when auto-selecting which model's corrected series anchors the
#: illustrative-gauge choice (flagship first, classical baselines last).
_MODEL_PRIORITY: tuple[str, ...] = (
    "regimeprobnet", "stacked", "catboost", "lgbm", "xgb", "ealstm",
    "qmap", "scaling",
)

#: minimum finite paired samples a gauge needs to contribute a skill value
MIN_GAUGE_SAMPLES: int = 5

__all__ = [
    "FDC_SEGMENTS",
    "per_regime_skill",
    "seasonal_skill",
    "fdc_segment_table",
    "peak_timing_distribution",
    "plot_regime_skill",
    "plot_seasonal_skill",
    "plot_fdc_segments",
    "plot_peak_timing",
    "plot_example_hydrographs",
    "build_all",
]


# --------------------------------------------------------------------------- #
#  Small numeric helpers                                                      #
# --------------------------------------------------------------------------- #
def _safe_median(values: Sequence[float]) -> float:
    """Median of the finite entries of ``values`` (NaN if none)."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def _attach_pred(df: pd.DataFrame, preds: Any, pred_col: str) -> tuple[pd.DataFrame, bool]:
    """Return ``(frame, has_pred)`` with the corrected discharge in ``pred_col``.

    ``preds`` may be ``None`` (use an existing ``pred_col`` if present), an array
    / Series aligned to ``df`` rows, or the name of an existing column.
    """
    if preds is None:
        return df, pred_col in df.columns
    if isinstance(preds, str):
        if preds not in df.columns:
            raise KeyError(f"prediction column {preds!r} not in frame")
        out = df if preds == pred_col else df.assign(**{pred_col: df[preds].to_numpy(float)})
        return out, True
    arr = np.asarray(preds, dtype=float).ravel()
    if arr.size != len(df):
        raise ValueError(f"preds length {arr.size} != n_rows {len(df)}")
    return df.assign(**{pred_col: arr}), True


def _per_gauge_skill(g: pd.DataFrame, obs_col: str, sim_col: str,
                     min_n: int) -> dict[str, float] | None:
    """KGE'/NSE/logNSE/PBIAS of ``sim_col`` vs ``obs_col`` for one gauge's rows."""
    obs = g[obs_col].to_numpy(float)
    sim = g[sim_col].to_numpy(float)
    if np.isfinite(obs + sim).sum() < min_n:
        return None
    # zero-variance bins make np.corrcoef emit a benign divide warning; the NaN it
    # returns is filtered downstream by the median aggregation.
    with np.errstate(invalid="ignore", divide="ignore"):
        k = kge_prime(obs, sim)
        return {"kge": k["kge"], "kge_r": k["r"], "kge_beta": k["beta"],
                "kge_gamma": k["gamma"], "nse": nse(obs, sim),
                "lognse": lognse(obs, sim), "pbias": pbias(obs, sim)}


def _grouped_skill(df: pd.DataFrame, bin_col: str, *, obs_col: str, raw_col: str,
                   pred_col: str, has_pred: bool, min_n: int) -> pd.DataFrame:
    """Median-over-gauges corrected-vs-raw skill within each ``bin_col`` value.

    Per (bin, gauge) skill is computed first and then aggregated to the median
    across gauges -- the same per-gauge-then-median convention as the headline
    KGE' the paper reports, so a single gauge cannot dominate a bin.
    """
    metrics = ("kge", "kge_r", "kge_beta", "kge_gamma", "nse", "lognse", "pbias")
    rows = []
    for key, grp in df.groupby(bin_col, observed=True):
        raw_acc: dict[str, list[float]] = {m: [] for m in metrics}
        cor_acc: dict[str, list[float]] = {m: [] for m in metrics}
        n_gauges = 0
        for _, g in grp.groupby("code", observed=True):
            r = _per_gauge_skill(g, obs_col, raw_col, min_n)
            if r is None:
                continue
            n_gauges += 1
            for m in metrics:
                raw_acc[m].append(r[m])
            if has_pred:
                c = _per_gauge_skill(g, obs_col, pred_col, min_n)
                if c is not None:
                    for m in metrics:
                        cor_acc[m].append(c[m])
        rec: dict[str, Any] = {bin_col: key, "n_obs": int(len(grp)),
                               "n_gauges": int(n_gauges)}
        for m in metrics:
            rec[f"{m}_raw"] = _safe_median(raw_acc[m])
            rec[m] = _safe_median(cor_acc[m]) if has_pred else np.nan
        rec["d_kge"] = rec["kge"] - rec["kge_raw"]
        rec["d_pbias_abs"] = abs(rec["pbias_raw"]) - abs(rec["pbias"])
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Regime-stratified skill                                                    #
# --------------------------------------------------------------------------- #
def per_regime_skill(df: pd.DataFrame, preds: Any = None, *,
                     obs_col: str = OBS_COL, raw_col: str = SIM_COL,
                     pred_col: str = PRED_COL, regime_col: str = REGIME_COL,
                     min_n: int = MIN_GAUGE_SAMPLES) -> pd.DataFrame:
    """Corrected-vs-raw skill stratified by hydrological regime.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table carrying ``obs_col``, ``raw_col``, a ``code`` column and
        a ``regime`` label (run :func:`sbc.features.regimes.classify_regimes`
        first if absent).
    preds : array-like, str or None
        Corrected discharge per row, the name of a column holding it, or ``None``
        to use an existing ``pred_col`` (and report raw-only if that is missing).
    obs_col, raw_col, pred_col, regime_col : str
        Column names.
    min_n : int
        Minimum finite paired samples a gauge needs to enter a regime's median.

    Returns
    -------
    pandas.DataFrame
        One row per regime (plus an ``"all"`` row), with median-over-gauges
        ``kge``/``kge_*``/``nse``/``lognse``/``pbias`` for the corrected and raw
        (``*_raw``) series, the improvement ``d_kge`` and ``d_pbias_abs``, and the
        ``n_obs`` / ``n_gauges`` counts.  Regimes are emitted in canonical order.
    """
    if regime_col not in df.columns:
        raise KeyError(f"missing regime column {regime_col!r}; classify regimes first")
    work, has_pred = _attach_pred(df, preds, pred_col)

    out = _grouped_skill(work, regime_col, obs_col=obs_col, raw_col=raw_col,
                         pred_col=pred_col, has_pred=has_pred, min_n=min_n)
    overall = _grouped_skill(work.assign(_all="all"), "_all", obs_col=obs_col,
                             raw_col=raw_col, pred_col=pred_col,
                             has_pred=has_pred, min_n=min_n)
    overall = overall.rename(columns={"_all": regime_col})
    out = pd.concat([out, overall], ignore_index=True)

    # canonical regime ordering (then the "all" row last)
    try:
        from ..features.regimes import REGIMES

        order = {r: i for i, r in enumerate(REGIMES)}
    except Exception:  # pragma: no cover
        order = {}
    out["_ord"] = out[regime_col].map(lambda r: order.get(str(r), 90)).fillna(90)
    out.loc[out[regime_col] == "all", "_ord"] = 99
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
#  Seasonal skill                                                             #
# --------------------------------------------------------------------------- #
def _season_key(dates: pd.Series, by: str) -> tuple[pd.Series, pd.Series]:
    """Return ``(bin_value, label)`` Series for month- or decade-of-year binning."""
    d = pd.to_datetime(dates)
    month = d.dt.month
    if by == "month":
        return month.astype(int), month.astype(int)
    if by == "decade":
        day = d.dt.day
        dec = np.where(day <= 10, 0, np.where(day <= 20, 1, 2))
        idx = (month.to_numpy() - 1) * 3 + dec + 1     # 1..36
        return pd.Series(idx, index=d.index, dtype=int), pd.Series(idx, index=d.index, dtype=int)
    raise ValueError(f"by must be 'month' or 'decade', got {by!r}")


def seasonal_skill(df: pd.DataFrame, preds: Any = None, *, by: str = "month",
                   obs_col: str = OBS_COL, raw_col: str = SIM_COL,
                   pred_col: str = PRED_COL, date_col: str = "date",
                   min_n: int = MIN_GAUGE_SAMPLES) -> pd.DataFrame:
    """Corrected-vs-raw skill resolved by calendar month or by decade-of-year.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table with ``obs_col``, ``raw_col``, ``code`` and ``date_col``.
    preds : array-like, str or None
        Corrected discharge (see :func:`per_regime_skill`).
    by : {"month", "decade"}
        ``"month"`` bins by calendar month (1-12); ``"decade"`` by Central-Asian
        decade-of-year (1-36, day 1-10 / 11-20 / 21-end of each month).
    obs_col, raw_col, pred_col, date_col : str
        Column names.
    min_n : int
        Minimum finite paired samples per gauge within a bin.

    Returns
    -------
    pandas.DataFrame
        One row per season bin with the same skill columns as
        :func:`per_regime_skill`; the bin index is in the ``"period"`` column and
        ``by`` is recorded in ``df.attrs["by"]``.
    """
    work, has_pred = _attach_pred(df, preds, pred_col)
    binv, _ = _season_key(work[date_col], by)
    work = work.assign(period=binv.to_numpy())
    out = _grouped_skill(work, "period", obs_col=obs_col, raw_col=raw_col,
                         pred_col=pred_col, has_pred=has_pred, min_n=min_n)
    out = out.sort_values("period").reset_index(drop=True)
    out.attrs["by"] = by
    return out


# --------------------------------------------------------------------------- #
#  Flow-duration-curve segment bias                                           #
# --------------------------------------------------------------------------- #
def _fdc_gauge_bias(obs: np.ndarray, raw: np.ndarray, cor: np.ndarray | None,
                    segments: Sequence[tuple[str, float, float]]
                    ) -> dict[tuple[str, str], float]:
    """Percent flow-volume bias of raw/corrected within each FDC segment, one gauge.

    Rows are assigned to a segment by the exceedance probability of the *observed*
    flow (Weibull plotting position ``rank/(n+1)``); within a segment the bias is
    ``100 * (sum(x) - sum(obs)) / sum(obs)``.
    """
    m = np.isfinite(obs) & np.isfinite(raw)
    if cor is not None:
        m &= np.isfinite(cor)
    obs, raw = obs[m], raw[m]
    cor = cor[m] if cor is not None else None
    n = obs.size
    if n < MIN_GAUGE_SAMPLES:
        return {}

    order = np.argsort(-obs, kind="mergesort")          # exceedance: largest first
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    p = ranks / (n + 1.0)

    out: dict[tuple[str, str], float] = {}
    last = segments[-1][0]
    for name, lo, hi in segments:
        sel = (p >= lo) & (p <= hi) if name == last else (p >= lo) & (p < hi)
        so = obs[sel].sum()
        if sel.sum() < 1 or so <= 0:
            continue
        out[(name, "raw")] = float(100.0 * (raw[sel].sum() - so) / so)
        if cor is not None:
            out[(name, "cor")] = float(100.0 * (cor[sel].sum() - so) / so)
    return out


def fdc_segment_table(df: pd.DataFrame, preds: Any = None, *,
                      obs_col: str = OBS_COL, raw_col: str = SIM_COL,
                      pred_col: str = PRED_COL,
                      segments: Sequence[tuple[str, float, float]] = FDC_SEGMENTS
                      ) -> pd.DataFrame:
    """Percent flow-volume bias by flow-duration-curve segment, raw vs corrected.

    The bias is computed per gauge (so gauges of different size are comparable)
    and aggregated to the median across gauges.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table with ``obs_col``, ``raw_col`` and ``code``.
    preds : array-like, str or None
        Corrected discharge (see :func:`per_regime_skill`).
    obs_col, raw_col, pred_col : str
        Column names.
    segments : sequence of (name, lo, hi)
        Exceedance-probability bands; defaults to :data:`FDC_SEGMENTS`.

    Returns
    -------
    pandas.DataFrame
        One row per segment with ``exceed_lo`` / ``exceed_hi``, the median
        percent volume bias ``pbias_raw`` and ``pbias`` (corrected), the
        absolute-bias reduction ``d_pbias_abs`` and ``n_gauges``.
    """
    work, has_pred = _attach_pred(df, preds, pred_col)
    raw_acc: dict[str, list[float]] = {s[0]: [] for s in segments}
    cor_acc: dict[str, list[float]] = {s[0]: [] for s in segments}
    seen: dict[str, int] = {s[0]: 0 for s in segments}
    for _, g in work.groupby("code", observed=True):
        cor = g[pred_col].to_numpy(float) if has_pred else None
        bias = _fdc_gauge_bias(g[obs_col].to_numpy(float),
                               g[raw_col].to_numpy(float), cor, segments)
        for (name, kind), val in bias.items():
            if kind == "raw":
                raw_acc[name].append(val)
                seen[name] += 1
            else:
                cor_acc[name].append(val)

    rows = []
    for name, lo, hi in segments:
        praw = _safe_median(raw_acc[name])
        pcor = _safe_median(cor_acc[name]) if has_pred else np.nan
        rows.append({"segment": name, "exceed_lo": lo, "exceed_hi": hi,
                     "n_gauges": int(seen[name]), "pbias_raw": praw,
                     "pbias": pcor, "d_pbias_abs": abs(praw) - abs(pcor)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Peak-timing distribution (from the per-gauge result table)                 #
# --------------------------------------------------------------------------- #
def peak_timing_distribution(per_gauge: pd.DataFrame, *, split: str = "temporal",
                             models: Sequence[str] | None = None,
                             err_col: str = "peak_timing_err",
                             raw_col: str = "peak_timing_err_raw") -> pd.DataFrame:
    """Distribution of the annual peak-timing error per model, from the result table.

    Unlike the time-series helpers above, this reads the real per-gauge result
    table directly, so it reflects the actual model zoo (the flagship included).

    Parameters
    ----------
    per_gauge : pandas.DataFrame
        A ``per_gauge_<tag>`` table with ``model`` / ``split`` and the
        ``peak_timing_err`` (corrected) and ``peak_timing_err_raw`` columns.
    split : str
        Validation split to summarise (``"temporal"``, ``"lobo"`` or ``"pur"``).
    models : sequence of str, optional
        Restrict / order the models shown; defaults to every model present.
    err_col, raw_col : str
        Column names of the corrected and raw peak-timing errors (days).

    Returns
    -------
    pandas.DataFrame
        One row per model with the absolute peak-timing error distribution
        (median / mean / p90) for the corrected and raw (``*_raw``) series, the
        median improvement in days and ``n_gauges``.
    """
    sub = per_gauge[per_gauge["split"] == split].copy()
    if sub.empty:
        log.warning("peak_timing_distribution: no rows for split=%s", split)
        return pd.DataFrame()
    avail = list(models) if models is not None else list(pd.unique(sub["model"]))

    def _stats(values: np.ndarray) -> tuple[float, float, float]:
        v = np.abs(np.asarray(values, float))
        v = v[np.isfinite(v)]
        if not v.size:
            return np.nan, np.nan, np.nan
        return float(np.median(v)), float(np.mean(v)), float(np.percentile(v, 90))

    rows = []
    for model in avail:
        g = sub[sub["model"] == model]
        if g.empty:
            continue
        med_c, mean_c, p90_c = _stats(g[err_col].to_numpy())
        med_r, mean_r, p90_r = _stats(g[raw_col].to_numpy())
        rows.append({
            "model": model, "split": split, "n_gauges": int(g["code"].nunique()),
            "ptd_median": med_c, "ptd_mean": mean_c, "ptd_p90": p90_c,
            "ptd_median_raw": med_r, "ptd_mean_raw": mean_r, "ptd_p90_raw": p90_r,
            "d_median_days": med_r - med_c,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("ptd_median").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
#  Plotting (Agg-safe)                                                        #
# --------------------------------------------------------------------------- #
def _new_axes(figsize: tuple[float, float]):
    """Return a fresh (fig, ax) on the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def _save(fig, path: str | Path) -> Path:
    """Tight-layout, save and close a figure; return its path."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figure -> %s", out)
    return out


def plot_regime_skill(table: pd.DataFrame, path: str | Path, *,
                      regime_col: str = REGIME_COL, label: str = "corrected") -> Path:
    """Grouped bar chart of raw vs corrected median KGE' per regime."""
    t = table[table[regime_col] != "all"]
    regimes = t[regime_col].astype(str).tolist()
    y = np.arange(len(regimes))
    fig, ax = _new_axes((7.0, 0.6 * len(regimes) + 1.8))
    h = 0.4
    ax.barh(y + h / 2, t["kge_raw"].to_numpy(float), height=h,
            label="raw GloFAS", color="#bdbdbd")
    if t["kge"].notna().any():
        ax.barh(y - h / 2, t["kge"].to_numpy(float), height=h,
                label=f"{label}", color="#1f77b4")
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(regimes)
    ax.set_xlabel("median per-gauge KGE'")
    ax.set_title("Skill by hydrological regime")
    ax.legend(fontsize="small")
    return _save(fig, path)


def plot_seasonal_skill(table: pd.DataFrame, path: str | Path, *,
                        label: str = "corrected") -> Path:
    """Two-panel seasonal skill: KGE' (top) and PBIAS (bottom) vs season bin."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    by = table.attrs.get("by", "month")
    x = table["period"].to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(x, table["kge_raw"], "-o", color="#bdbdbd", ms=3, label="raw GloFAS")
    if table["kge"].notna().any():
        axes[0].plot(x, table["kge"], "-o", color="#1f77b4", ms=3, label=label)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_ylabel("median KGE'")
    axes[0].legend(fontsize="small")
    axes[0].set_title(f"Seasonal skill (by {by})")

    axes[1].plot(x, table["pbias_raw"], "-o", color="#bdbdbd", ms=3, label="raw GloFAS")
    if table["pbias"].notna().any():
        axes[1].plot(x, table["pbias"], "-o", color="#1f77b4", ms=3, label=label)
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_ylabel("median PBIAS [%]")
    axes[1].set_xlabel("calendar month" if by == "month" else "decade of year (1-36)")
    return _save(fig, path)


def plot_fdc_segments(table: pd.DataFrame, path: str | Path, *,
                      label: str = "corrected") -> Path:
    """Grouped bar chart of percent volume bias per FDC segment, raw vs corrected."""
    segs = table["segment"].tolist()
    x = np.arange(len(segs))
    fig, ax = _new_axes((7.0, 4.5))
    w = 0.4
    ax.bar(x - w / 2, table["pbias_raw"].to_numpy(float), width=w,
           label="raw GloFAS", color="#bdbdbd")
    if table["pbias"].notna().any():
        ax.bar(x + w / 2, table["pbias"].to_numpy(float), width=w,
               label=label, color="#1f77b4")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n[{lo:.0%}-{hi:.0%}]"
                        for s, lo, hi in zip(segs, table["exceed_lo"], table["exceed_hi"])])
    ax.set_ylabel("flow-volume bias [%]")
    ax.set_title("Flow-duration-curve segment bias")
    ax.legend(fontsize="small")
    return _save(fig, path)


def plot_peak_timing(table: pd.DataFrame, path: str | Path) -> Path:
    """Bar chart of median |peak-timing error| (days) per model vs the raw value."""
    if table.empty:
        fig, ax = _new_axes((6.0, 3.0))
        ax.set_title("No peak-timing data")
        return _save(fig, path)
    models = table["model"].tolist()
    y = np.arange(len(models))
    fig, ax = _new_axes((7.0, 0.5 * len(models) + 1.8))
    h = 0.4
    ax.barh(y + h / 2, table["ptd_median_raw"].to_numpy(float), height=h,
            label="raw GloFAS", color="#bdbdbd")
    ax.barh(y - h / 2, table["ptd_median"].to_numpy(float), height=h,
            label="corrected", color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("median |annual peak-timing error| [days]")
    ax.set_title("Peak-timing error by model")
    ax.legend(fontsize="small")
    return _save(fig, path)


def plot_example_hydrographs(df: pd.DataFrame, codes: Sequence[str], path: str | Path, *,
                             preds: Any = None, obs_col: str = OBS_COL,
                             raw_col: str = SIM_COL, pred_col: str = PRED_COL,
                             date_col: str = "date", test_start=None,
                             label: str = "corrected") -> Path:
    """Stacked observed / raw / corrected hydrographs for several gauges.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table with ``obs_col``, ``raw_col``, ``code`` and ``date_col``.
    codes : sequence of str
        Gauges to plot, one panel each (in order).
    path : str or Path
        Output PNG path.
    preds : array-like, str or None
        Corrected discharge for the whole table (see :func:`per_regime_skill`).
    test_start : datetime-like, optional
        If given, a dashed vertical line marks the temporal-holdout boundary.
    label : str
        Legend label for the corrected series.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    work, has_pred = _attach_pred(df, preds, pred_col)
    codes = [c for c in codes if c in set(work["code"].astype(str))]
    if not codes:
        fig, ax = _new_axes((8.0, 3.0))
        ax.set_title("No gauges available for hydrograph plot")
        return _save(fig, path)

    n = len(codes)
    fig, axes = plt.subplots(n, 1, figsize=(10.0, 2.4 * n + 0.5), squeeze=False)
    for ax, code in zip(axes[:, 0], codes):
        g = work[work["code"].astype(str) == str(code)].sort_values(date_col)
        d = pd.to_datetime(g[date_col])
        ax.plot(d, g[obs_col], color="black", lw=1.1, label="observed")
        ax.plot(d, g[raw_col], color="#ff7f0e", lw=0.9, alpha=0.85, label="raw GloFAS")
        title_extra = ""
        if has_pred:
            ax.plot(d, g[pred_col], color="#1f77b4", lw=0.9, alpha=0.85, label=label)
            mask = (d >= test_start) if test_start is not None else slice(None)
            gg = g[mask] if test_start is not None else g
            with np.errstate(invalid="ignore", divide="ignore"):
                kr = kge_prime(gg[obs_col].to_numpy(float), gg[raw_col].to_numpy(float))["kge"]
                kc = kge_prime(gg[obs_col].to_numpy(float), gg[pred_col].to_numpy(float))["kge"]
            title_extra = f"  |  KGE' raw {kr:+.2f} -> {label} {kc:+.2f}"
        if test_start is not None:
            ax.axvline(pd.to_datetime(test_start), color="0.5", ls="--", lw=0.8)
        basin = g["basin"].iloc[0] if "basin" in g.columns else ""
        ax.set_title(f"gauge {code} ({basin}){title_extra}", fontsize="small")
        ax.set_ylabel("Q [m3 s-1]")
        ax.margins(x=0.01)
    axes[0, 0].legend(fontsize="small", ncol=3, loc="upper right")
    axes[-1, 0].set_xlabel("date")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def _scale_from_tag(tag: str) -> str:
    """Infer the temporal scale ('daily' / 'decadal') from a result tag."""
    return "daily" if "daily" in tag else "decadal"


def _reference_correction(df: pd.DataFrame, corrector: str | None,
                          test_frac: float = 0.3, seed: int = 0):
    """Fit a leakage-safe reference corrector on the temporal-train split.

    Returns ``(preds, label, test_mask)``.  ``preds`` is ``None`` when
    ``corrector`` is ``None`` / unavailable (the diagnostics then report raw
    only).  The split mirrors the paper's temporal holdout so the corrected
    series is evaluated out of sample.
    """
    from ..validation.splits import temporal_split

    tr, te = temporal_split(df, test_frac)
    if corrector in (None, "none"):
        return None, "raw", te
    try:
        if corrector == "qmap":
            from ..models.quantile_mapping import QuantileMappingCorrector

            model = QuantileMappingCorrector().fit(df[tr])
        elif corrector == "scaling":
            from ..models.quantile_mapping import LinearScalingCorrector

            model = LinearScalingCorrector().fit(df[tr])
        elif corrector == "lgbm":
            from ..models.boosting import LightGBMCorrector

            model = LightGBMCorrector(n_optuna_trials=0).fit(df[tr])
        else:
            raise ValueError(f"unknown corrector {corrector!r}")
        preds = np.asarray(model.predict(df), dtype=float)
        return preds, corrector, te
    except Exception as exc:  # pragma: no cover - degrade gracefully
        log.warning("reference correction (%s) failed: %s; reporting raw only",
                    corrector, exc)
        return None, "raw", te


def _select_gauges(per_gauge: pd.DataFrame, available: set[str], n: int) -> list[str]:
    """Pick illustrative gauges: largest KGE' gain first, spread across basins."""
    sub = per_gauge[(per_gauge["split"] == "temporal")
                    & per_gauge["code"].astype(str).isin(available)].copy()
    if sub.empty:
        return sorted(available)[:n]
    model = next((m for m in _MODEL_PRIORITY if m in set(sub["model"])),
                 sub["model"].iloc[0])
    sub = sub[sub["model"] == model].copy()
    sub["d_kge"] = sub["kge"] - sub["kge_raw"]
    sub = sub.sort_values("d_kge", ascending=False)
    picked: list[str] = []
    seen_basins: set[str] = set()
    for _, r in sub.iterrows():                      # one per basin first
        b = r.get("basin")
        if b not in seen_basins:
            picked.append(str(r["code"]))
            seen_basins.add(b)
        if len(picked) >= n:
            return picked
    for _, r in sub.iterrows():                      # then fill remaining slots
        c = str(r["code"])
        if c not in picked:
            picked.append(c)
        if len(picked) >= n:
            break
    return picked


def build_all(tag: str = "real_decadal", *, model_table: pd.DataFrame | None = None,
              per_gauge: pd.DataFrame | None = None, corrector: str | None = "qmap",
              n_hydrographs: int = 6, figures_dir: str | Path | None = None,
              tables_dir: str | Path | None = None, scale: str | None = None,
              min_n: int = MIN_GAUGE_SAMPLES) -> dict[str, Any]:
    """Compute every diagnostic for ``tag`` and write the tables and figures.

    The per-gauge result table powers :func:`peak_timing_distribution` (the real
    model zoo).  The assembled modelling table -- plus a leakage-safe reference
    correction (see module docstring) -- powers the regime / seasonal / FDC /
    hydrograph diagnostics, evaluated on the temporal-holdout test period.

    Parameters
    ----------
    tag : str
        Result tag, e.g. ``"real_decadal"`` / ``"real_daily"``.  Reads
        ``results/tables/per_gauge_<tag>.parquet`` unless ``per_gauge`` is given.
    model_table : pandas.DataFrame, optional
        Pre-loaded modelling table (with ``date`` / ``q_obs`` / ``q_glofas`` and
        optionally a ``q_pred`` column).  Defaults to
        ``datasets/processed/model_table_<scale>.parquet``.
    per_gauge : pandas.DataFrame, optional
        Pre-loaded per-gauge result table (overrides the on-disk read).
    corrector : {"qmap", "scaling", "lgbm", None}
        Reference corrector for the time-series diagnostics.  ``None`` reports
        raw only (or uses a ``q_pred`` column already present in ``model_table``).
    n_hydrographs : int
        Number of illustrative gauges to draw.
    figures_dir, tables_dir : path-like, optional
        Output directories (default ``results/figures`` and ``results/tables``).
    scale : str, optional
        Override the scale inferred from ``tag``.
    min_n : int
        Minimum per-gauge samples for the skill helpers.

    Returns
    -------
    dict
        ``{"tables": {name: path}, "figures": {name: path}, "label": str,
        "regime_skill": DataFrame, "seasonal_skill": DataFrame,
        "fdc_segments": DataFrame, "peak_timing": DataFrame}``.
    """
    from ..utils import save_table

    scale = scale or _scale_from_tag(tag)
    figures_dir = Path(figures_dir) if figures_dir is not None else PATHS.figures
    tables_dir = Path(tables_dir) if tables_dir is not None else PATHS.tables
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables_out: dict[str, Path] = {}
    figures_out: dict[str, Path] = {}

    def _emit(name: str, frame: pd.DataFrame) -> None:
        tables_out[name] = save_table(frame, tables_dir / f"diag_{name}_{tag}.parquet",
                                      csv_mirror=True)

    # -- 1. peak-timing distribution from the real per-gauge table ----------
    if per_gauge is None:
        pg_path = PATHS.tables / f"per_gauge_{tag}.parquet"
        per_gauge = pd.read_parquet(pg_path)
        log.info("loaded per-gauge table %s (%d rows)", pg_path.name, len(per_gauge))
    ptd = peak_timing_distribution(per_gauge, split="temporal")
    if not ptd.empty:
        _emit("peak_timing", ptd)
        figures_out["peak_timing"] = plot_peak_timing(
            ptd, figures_dir / f"diag_peak_timing_{tag}.png")

    # -- 2. modelling table + reference correction --------------------------
    if model_table is None:
        mt_path = PATHS.processed / f"model_table_{scale}.parquet"
        if not mt_path.exists():
            log.warning("modelling table %s absent; only peak-timing produced", mt_path)
            return {"tables": tables_out, "figures": figures_out, "label": "raw",
                    "regime_skill": pd.DataFrame(), "seasonal_skill": pd.DataFrame(),
                    "fdc_segments": pd.DataFrame(), "peak_timing": ptd}
        model_table = pd.read_parquet(mt_path)
        log.info("loaded modelling table %s (%d rows, %d gauges)",
                 mt_path.name, len(model_table), model_table["code"].nunique())

    from ..features.regimes import classify_regimes

    df = model_table.reset_index(drop=True)
    if REGIME_COL not in df.columns:
        df = classify_regimes(df)

    preds, label, test_mask = _reference_correction(df, corrector)
    if preds is not None:
        df = df.assign(**{PRED_COL: preds})
    test = df[test_mask.to_numpy()] if hasattr(test_mask, "to_numpy") else df[test_mask]
    test_start = pd.to_datetime(test["date"]).min() if len(test) else None
    pred_arg = PRED_COL if preds is not None else None

    # -- 3. regime / seasonal / FDC skill on the held-out test period -------
    reg = per_regime_skill(test, pred_arg, min_n=min_n)
    _emit("regime_skill", reg)
    figures_out["regime_skill"] = plot_regime_skill(
        reg, figures_dir / f"diag_regime_skill_{tag}.png", label=label)

    sea_m = seasonal_skill(test, pred_arg, by="month", min_n=min_n)
    _emit("seasonal_month", sea_m)
    figures_out["seasonal_month"] = plot_seasonal_skill(
        sea_m, figures_dir / f"diag_seasonal_month_{tag}.png", label=label)

    sea_d = seasonal_skill(test, pred_arg, by="decade", min_n=min_n)
    _emit("seasonal_decade", sea_d)
    figures_out["seasonal_decade"] = plot_seasonal_skill(
        sea_d, figures_dir / f"diag_seasonal_decade_{tag}.png", label=label)

    fdc = fdc_segment_table(test, pred_arg)
    _emit("fdc_segments", fdc)
    figures_out["fdc_segments"] = plot_fdc_segments(
        fdc, figures_dir / f"diag_fdc_segments_{tag}.png", label=label)

    # -- 4. illustrative hydrographs (full record, test boundary marked) ----
    codes = _select_gauges(per_gauge, set(df["code"].astype(str)), n_hydrographs)
    figures_out["hydrographs"] = plot_example_hydrographs(
        df, codes, figures_dir / f"diag_hydrographs_{tag}.png",
        preds=pred_arg, test_start=test_start, label=label)

    log.info("build_all(%s): %d tables, %d figures (corrected=%s)",
             tag, len(tables_out), len(figures_out), label)
    return {"tables": tables_out, "figures": figures_out, "label": label,
            "regime_skill": reg, "seasonal_skill": sea_m,
            "fdc_segments": fdc, "peak_timing": ptd}


# --------------------------------------------------------------------------- #
#  Self-test (tiny synthetic table; writes to a scratch dir)                  #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.quantile_mapping import QuantileMappingCorrector
    from ..schemas import validate
    from ..synthetic import generate
    from ..validation.splits import temporal_split

    df = validate(generate(scale="decadal", years=8, n_basins=3, seed=7))
    df = classify_regimes(build_features(df, scale="decadal")).reset_index(drop=True)

    tr, te = temporal_split(df, 0.3)
    model = QuantileMappingCorrector().fit(df[tr])
    preds = model.predict(df)

    reg = per_regime_skill(df, preds, min_n=3)
    sea = seasonal_skill(df, preds, by="month", min_n=3)
    dec = seasonal_skill(df, preds, by="decade", min_n=3)
    fdc = fdc_segment_table(df, preds)

    # peak-timing from any existing real/synthetic per-gauge table if present
    ptd = pd.DataFrame()
    pg_path = PATHS.tables / "per_gauge_synthetic_decadal_quick.parquet"
    if pg_path.exists():
        ptd = peak_timing_distribution(pd.read_parquet(pg_path))

    assert {"kge", "kge_raw", "pbias", "pbias_raw"}.issubset(reg.columns)
    assert (reg["regime"] == "all").sum() == 1, "missing the 'all' aggregate row"
    assert set(sea["period"]).issubset(set(range(1, 13))), "month bins out of range"
    assert set(dec["period"]).issubset(set(range(1, 37))), "decade bins out of range"
    assert list(fdc["segment"]) == [s[0] for s in FDC_SEGMENTS], "fdc segments"
    assert fdc["pbias_raw"].notna().any(), "no raw FDC bias computed"

    with tempfile.TemporaryDirectory() as tmp:
        out = build_all("synthetic_decadal_quick", model_table=df,
                        figures_dir=Path(tmp) / "fig", tables_dir=Path(tmp) / "tab",
                        n_hydrographs=3)
        n_fig = len(list((Path(tmp) / "fig").glob("*.png")))
        n_tab = len(list((Path(tmp) / "tab").glob("*.parquet")))

    all_kge_raw = float(reg.loc[reg["regime"] == "all", "kge_raw"].iloc[0])
    all_kge_cor = float(reg.loc[reg["regime"] == "all", "kge"].iloc[0])
    print(f"[diagnostics] regimes={reg['regime'].nunique()} "
          f"seasonal_months={len(sea)} decades={len(dec)} fdc_rows={len(fdc)} "
          f"| all-regime KGE' raw={all_kge_raw:+.3f} -> qmap={all_kge_cor:+.3f}")
    print(f"[diagnostics] build_all wrote {n_fig} figures + {n_tab} tables to scratch "
          f"| figures={list(out['figures'])}")
    print(f"[diagnostics] peak_timing models={'n/a' if ptd.empty else list(ptd['model'])}")


if __name__ == "__main__":
    _selftest()
