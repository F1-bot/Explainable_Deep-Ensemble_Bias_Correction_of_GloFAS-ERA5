"""Leakage-safe dynamic feature engineering and static-attribute curation.

The corrector models predict the **log-space residual** between observed and
GloFAS-ERA5 discharge.  Because the modelling table is a time series, the single
most important correctness property of the feature layer is *causality*: every
temporal predictor must be computable from information available **at or before**
the period being predicted, and none may be derived from the prediction target
(``q_obs`` / ``log_residual``).  Violating this leaks the answer and inflates
skill — a classic and silent failure mode in hydrological post-processing.

This module therefore builds memory features strictly from the **exogenous**
state available at run time (the raw GloFAS discharge and the meteorological
forcing), using only past rows for any aggregate.  The physical motivation for
each family follows the snow-influenced bias literature:

* **GloFAS memory** — the raw model's own recent behaviour (log discharge, its
  short lags, causal rolling moments and rate of change) carries most of the
  predictable structure of its error.
* **Snow memory** — snow-water-equivalent lags / means, recent melt totals and
  cumulative positive degree-days (reset at the 1 October hydrological-year
  boundary) encode the storage and release that GloFAS mis-times in nival
  basins.
* **Rain-on-snow & melt season** — rain falling on an existing, melting snowpack
  is a known driver of GloFAS error; a melt-season flag separates the freshet.
* **Temperature** — causal air-temperature means and freeze/thaw crossings.
* **Seasonality** — smooth sine/cosine encodings of day-of-year and of the
  within-year decade index.

Static catchment attributes are curated separately: when the real CA-discharge
attribute table contributes monthly climatology groups (``scf_*`` snow-cover
fraction, ``pr_*`` precipitation, ``tas_*`` air temperature), each twelve-month
group is collapsed into four compact, physically meaningful seasonal descriptors
and the raw monthly columns are dropped.  Terrain, glacier, area and aridity
attributes are kept verbatim.  Synthetic tables carry no monthly groups, so the
statics pass through unchanged.

All engineered columns are prefixed ``f_`` so that :func:`engineered_feature_names`
can recover them without a hard-coded list, while the canonical
:mod:`sbc.schemas` static/dynamic discovery continues to work on their per-gauge
variance.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import EPS
from ..utils import get_logger

log = get_logger(__name__)

# Causal rolling-window sizes, in number of periods, per temporal scale.
_ROLL_WINDOWS: dict[str, tuple[int, int, int]] = {
    "decadal": (3, 6, 9),
    "daily": (7, 30, 90),
}
# Snow-water-equivalent memory lags, in number of periods, per scale.
_SWE_LAGS: dict[str, tuple[int, int, int]] = {
    "decadal": (3, 6, 9),
    "daily": (30, 60, 90),
}
_GLOFAS_LAGS: tuple[int, ...] = (1, 2, 3)

# Snowmelt freshet months (Apr-Jul) for the dynamic melt-season flag.
_MELT_MONTHS: tuple[int, ...] = (4, 5, 6, 7)

# Static monthly-climatology groups and the seasons used to summarise them.
_MONTHLY_GROUPS: tuple[str, ...] = ("scf", "pr", "tas")
_COLD_MONTHS: tuple[int, ...] = (12, 1, 2, 3, 4, 5)   # DJF + MAM
_AMJ_MONTHS: tuple[int, ...] = (4, 5, 6)              # melt-season AMJ

__all__ = ["build_features", "engineered_feature_names"]


# --------------------------------------------------------------------------- #
#  Causal transform primitives (operate within a single gauge's series)        #
# --------------------------------------------------------------------------- #
def _lag(k: int):
    """Return a transform that shifts a series ``k`` periods into the past."""
    return lambda s: s.shift(k)


def _roc():
    """First difference (current minus previous) — uses no future row."""
    return lambda s: s - s.shift(1)


def _causal_mean(window: int):
    """Mean over the ``window`` rows strictly *before* the current one."""
    return lambda s: s.shift(1).rolling(window, min_periods=1).mean()


def _causal_std(window: int):
    """Std over the ``window`` rows strictly *before* the current one."""
    return lambda s: s.shift(1).rolling(window, min_periods=2).std()


def _causal_sum(window: int):
    """Sum over the ``window`` rows strictly *before* the current one."""
    return lambda s: s.shift(1).rolling(window, min_periods=1).sum()


def _gt(df: pd.DataFrame, col: str, func) -> pd.Series:
    """Apply a per-gauge transform to ``col`` (groupby('code'), date-ordered)."""
    return df.groupby("code", sort=False)[col].transform(func)


# --------------------------------------------------------------------------- #
#  Static curation: collapse monthly climatology groups                        #
# --------------------------------------------------------------------------- #
def _month_of(col: str, stem: str) -> int | None:
    """Parse a 1..12 month index from a ``<stem>_<month>`` column name."""
    m = re.search(r"(\d+)$", col)
    if not m:
        return None
    mi = int(m.group(1))
    return mi if 1 <= mi <= 12 else None


def _collapse_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``scf_*``/``pr_*``/``tas_*`` monthly groups with seasonal stats.

    For every group with at least two parseable monthly columns, add four
    ``f_<stem>_*`` descriptors (annual mean, cold-season DJF/MAM mean,
    melt-season AMJ mean, seasonal amplitude) and drop the raw monthly columns.
    Groups that are absent or unparseable are left untouched.
    """
    out = df
    for stem in _MONTHLY_GROUPS:
        month_map = {}
        for c in out.columns:
            if re.match(rf"^{stem}_", c):
                mi = _month_of(c, stem)
                if mi is not None:
                    month_map[c] = mi
        if len(month_map) < 2:
            continue
        mcols = list(month_map)
        vals = out[mcols].apply(pd.to_numeric, errors="coerce")
        out = out.copy()
        out[f"f_{stem}_ann"] = vals.mean(axis=1)
        out[f"f_{stem}_amp"] = vals.max(axis=1) - vals.min(axis=1)
        cold = [c for c, mi in month_map.items() if mi in _COLD_MONTHS]
        if cold:
            out[f"f_{stem}_cold"] = out[cold].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        melt = [c for c, mi in month_map.items() if mi in _AMJ_MONTHS]
        if melt:
            out[f"f_{stem}_melt"] = out[melt].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        out = out.drop(columns=mcols)
        log.debug("collapsed %d '%s' monthly columns into seasonal descriptors",
                  len(mcols), stem)
    return out


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
def engineered_feature_names(df: pd.DataFrame) -> list[str]:
    """Names of the columns added by :func:`build_features` (the ``f_`` family)."""
    return [c for c in df.columns if c.startswith("f_")]


def build_features(df: pd.DataFrame, scale: str = "decadal", *,
                   p_thr: float = 10.0, s_thr: float = 10.0) -> pd.DataFrame:
    """Add leakage-safe dynamic features and curate static attributes.

    Parameters
    ----------
    df : pandas.DataFrame
        A modelling table (see :mod:`sbc.schemas`).  Must contain ``code`` and
        ``date``; engineered families are added only for the raw columns that
        are present, so the function degrades gracefully on partial tables.
    scale : str, default "decadal"
        Temporal scale selecting the rolling-window and snow-lag sizes
        (``"decadal"`` or ``"daily"``); unknown scales fall back to decadal.
    p_thr, s_thr : float
        Precipitation [mm] and snow-water-equivalent [mm] thresholds for the
        rain-on-snow indicator.

    Returns
    -------
    pandas.DataFrame
        A new, date-ordered table (the input is never mutated) with all
        ``f_``-prefixed engineered columns added and raw monthly-climatology
        groups, if any, collapsed.  Warm-up NaNs from lagging are filled per
        gauge by back-filling the first valid value; columns with no usable
        history remain NaN (models tolerate NaN).

    Notes
    -----
    No engineered column is derived from ``q_obs`` or ``log_residual``; every
    temporal aggregate is strictly causal (uses only rows at or before the
    current one).  The per-gauge back-fill of the short warm-up region is the
    one deliberate, documented exception and touches only the leading periods.
    """
    if "code" not in df.columns or "date" not in df.columns:
        raise ValueError("build_features requires 'code' and 'date' columns")

    windows = _ROLL_WINDOWS.get(scale, _ROLL_WINDOWS["decadal"])
    swe_lags = _SWE_LAGS.get(scale, _SWE_LAGS["decadal"])
    mid_window = windows[1]

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # Stable date order within each gauge underpins every causal aggregate.
    out = out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)

    fcols: list[str] = []

    def add(name: str, values) -> None:
        out[name] = np.asarray(values, dtype=float)
        fcols.append(name)

    month = out["date"].dt.month
    day = out["date"].dt.day
    doy = out["date"].dt.dayofyear.to_numpy(float)

    # -- GloFAS memory ------------------------------------------------------ #
    if "q_glofas" in out.columns:
        add("f_log_qglofas", np.log(out["q_glofas"].to_numpy(float) + EPS))
        for k in _GLOFAS_LAGS:
            add(f"f_log_qglofas_lag{k}", _gt(out, "f_log_qglofas", _lag(k)))
        for w in windows:
            add(f"f_log_qglofas_rmean{w}", _gt(out, "f_log_qglofas", _causal_mean(w)))
            add(f"f_log_qglofas_rstd{w}", _gt(out, "f_log_qglofas", _causal_std(w)))
        add("f_log_qglofas_roc", _gt(out, "f_log_qglofas", _roc()))

    # -- Snow memory: SWE lags / means, recent melt, cumulative PDD --------- #
    if "swe" in out.columns:
        for L in swe_lags:
            add(f"f_swe_lag{L}", _gt(out, "swe", _lag(L)))
        for w in windows:
            add(f"f_swe_rmean{w}", _gt(out, "swe", _causal_mean(w)))
    if "smlt" in out.columns:
        for w in windows:
            add(f"f_smlt_rsum{w}", _gt(out, "smlt", _causal_sum(w)))
    if "t2m_mean" in out.columns:
        # Cumulative positive degree-days within the hydrological year (1 Oct).
        hydro_year = out["date"].dt.year + (month >= 10).astype(int)
        out["_pdd"] = np.maximum(out["t2m_mean"].to_numpy(float), 0.0)
        add("f_pdd_cum", out.groupby(["code", hydro_year], sort=False)["_pdd"].cumsum())
        out = out.drop(columns="_pdd")

    # -- Rain-on-snow indicator and melt-season flag ------------------------ #
    if {"tp", "swe", "t2m_mean"} <= set(out.columns):
        ros = ((out["tp"] > p_thr) & (out["swe"] > s_thr) & (out["t2m_mean"] > 0.0))
        add("f_rain_on_snow", ros.to_numpy(float))
    add("f_melt_season", month.isin(_MELT_MONTHS).to_numpy(float))

    # -- Temperature: causal means and freeze/thaw crossings ---------------- #
    if "t2m_mean" in out.columns:
        for w in windows:
            add(f"f_t2m_rmean{w}", _gt(out, "t2m_mean", _causal_mean(w)))
        if {"t2m_min", "t2m_max"} <= set(out.columns):
            ft = (out["t2m_min"] < 0.0) & (out["t2m_max"] > 0.0)
        else:
            prev = _gt(out, "t2m_mean", _lag(1))
            cur = out["t2m_mean"]
            ft = ((cur > 0.0) & (prev < 0.0)) | ((cur < 0.0) & (prev > 0.0))
        add("f_freeze_thaw", ft.to_numpy(float))
        out["_ft"] = ft.to_numpy(float)
        add("f_t2m_ft_crossings", _gt(out, "_ft", _causal_sum(mid_window)))
        out = out.drop(columns="_ft")

    # -- Seasonality: smooth day-of-year and decade-index encodings --------- #
    ang_doy = 2.0 * np.pi * doy / 365.25
    add("f_doy_sin", np.sin(ang_doy))
    add("f_doy_cos", np.cos(ang_doy))
    third = np.where(day.to_numpy() <= 10, 0, np.where(day.to_numpy() <= 20, 1, 2))
    decade_idx = (month.to_numpy() - 1) * 3 + third      # 0..35 within the year
    ang_dec = 2.0 * np.pi * decade_idx / 36.0
    add("f_decade_sin", np.sin(ang_dec))
    add("f_decade_cos", np.cos(ang_dec))

    # -- Fill warm-up NaNs from lagging (per-gauge back-fill) --------------- #
    if fcols:
        out[fcols] = out.groupby("code", sort=False)[fcols].bfill()

    # -- Static curation: collapse monthly climatology groups --------------- #
    out = _collapse_monthly(out)

    log.info("build_features(scale=%s): +%d engineered columns over %d rows",
             scale, len(engineered_feature_names(out)), len(out))
    return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sbc.synthetic import generate

    df = generate(scale="decadal")
    feat = build_features(df, scale="decadal")
    added = engineered_feature_names(feat)

    # 1) no rows gained or lost
    assert len(feat) == len(df), "row count changed"

    # 2) q_obs / log_residual were never used: rebuilding without them is identical
    drop = [c for c in ("q_obs", "log_residual") if c in df.columns]
    feat_no_obs = build_features(df.drop(columns=drop), scale="decadal")
    assert engineered_feature_names(feat_no_obs) == added, "feature set depends on target"
    q_obs_independent = all(
        np.allclose(feat[c].to_numpy(float), feat_no_obs[c].to_numpy(float), equal_nan=True)
        for c in added
    )
    assert q_obs_independent, "an engineered feature changed when q_obs was removed"

    # 3) the monthly-climatology collapse path (synthetic data has no such groups)
    inj = df.copy()
    for stem in ("scf", "pr", "tas"):
        for mth in range(1, 13):
            inj[f"{stem}_{mth}"] = float(mth)            # static, per-gauge constant
    coll = build_features(inj, scale="decadal")
    collapsed_ok = (
        not any(re.match(r"^(scf|pr|tas)_\d+$", c) for c in coll.columns)
        and {"f_scf_ann", "f_pr_amp", "f_tas_melt"} <= set(coll.columns)
    )

    print(f"OK rows={len(feat)} features_added={len(added)} "
          f"q_obs_used=False(independent={q_obs_independent}) "
          f"monthly_collapse={collapsed_ok}")
