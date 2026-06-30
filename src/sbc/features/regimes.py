"""Rule-based hydrological-regime classifier for snow-influenced basins.

Snow-/glacier-fed catchments cycle through a small set of physically distinct
runoff-generation regimes over the water year, and GloFAS-ERA5 mis-represents
them in characteristic, regime-dependent ways (a damped, early-shifted freshet;
unresolved rain-on-snow floods; missing glacier melt).  Labelling every
(gauge, period) row with the regime that physically dominates therefore lets the
downstream models stratify their bias correction and lets the evaluation report
skill *by process* rather than as a single basin-average number.

This module assigns one of five mutually exclusive regimes to each row using
transparent, physically-motivated rules on the ERA5-Land forcings carried in the
modelling table (air temperature ``t2m_mean``, snow-water-equivalent ``swe``,
snowmelt ``smlt``, total precipitation ``tp``), the day-of-year, and the static
catchment attributes ``glacier_frac`` / ``snow_frac`` when present:

``accumulation``
    Cold season: sub-freezing air temperature with a building snowpack
    (rising SWE or precipitation falling as snow).
``melt_freshet``
    Spring / early-summer snowmelt peak: high melt flux while a seasonal
    snowpack is still present.
``rain_on_snow``
    Liquid precipitation onto an existing snowpack at above-freezing
    temperature -- a distinct, flood-prone regime.
``glacier_melt``
    Mid/late-summer ice melt: snowpack exhausted, warm conditions and a
    non-negligible glacier fraction.
``recession``
    None of the above -- baseflow recession / low-flow.

All thresholds are exposed as documented module constants.  The classifier is
deterministic and fully vectorised (group-wise, no per-row Python loops); the
two "high flux" tests adapt to each gauge and to the temporal scale by combining
an absolute physical floor with a per-gauge quantile, so the same rules apply
unchanged to daily and decadal tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import REGIME_COL
from ..utils import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Regime vocabulary                                                          #
# --------------------------------------------------------------------------- #
#: canonical, ordered regime labels; ``regime_id`` is the index into this list
REGIMES: list[str] = [
    "accumulation",
    "melt_freshet",
    "rain_on_snow",
    "glacier_melt",
    "recession",
]

#: label -> integer id (stable across runs and tables)
REGIME_IDS: dict[str, int] = {r: i for i, r in enumerate(REGIMES)}
REGIME_ID_COL = "regime_id"

# --------------------------------------------------------------------------- #
#  Physical thresholds (documented module constants)                          #
# --------------------------------------------------------------------------- #
# Air-temperature thresholds [degC] -----------------------------------------
T_FREEZE_C: float = 0.0     # below: cold season, precip as snow, melt suppressed
T_MELT_C: float = 0.0       # above: liquid water present / active melt
T_GLACIER_C: float = 1.0    # above (snow gone): exposed-ice degree-day melt

# Snow-water-equivalent thresholds [mm] -------------------------------------
SWE_PRESENT_MM: float = 10.0  # snowpack present (freshet / rain-on-snow possible)
SWE_TRACE_MM: float = 5.0     # snowpack effectively exhausted (glacier season)
SWE_RISE_MM: float = 0.5      # minimum SWE increment to call the pack "building"

# "High flux" tests: per-gauge quantile of the *positive* values, floored by an
# absolute physical minimum so trivially small fluxes never count as high.
# Units are mm per period (mm/day at daily scale, mean mm/day at decadal scale).
# Melt uses a low quantile so the whole freshet pulse is captured (not just its
# peak); precipitation uses a high quantile because rain-on-snow is event-like.
SMLT_HIGH_QUANTILE: float = 0.25
TP_HIGH_QUANTILE: float = 0.80
SMLT_FLOOR_MM: float = 1.0
TP_FLOOR_MM: float = 2.0

# Static catchment-attribute thresholds [fraction 0-1] ----------------------
GLACIER_FRAC_MIN: float = 0.02  # min glacier cover to admit a glacier-melt regime
SNOW_FRAC_MIN: float = 0.05     # min snow influence to admit snow regimes

# Warm-season day-of-year window (northern hemisphere) gating glacier melt ---
WARM_SEASON_DOY: tuple[int, int] = (121, 305)  # ~1 May .. 31 Oct

__all__ = [
    "REGIMES",
    "REGIME_IDS",
    "REGIME_ID_COL",
    "classify_regimes",
    "regime_onehot",
]


# --------------------------------------------------------------------------- #
#  Small vectorised helpers                                                   #
# --------------------------------------------------------------------------- #
def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return ``df[col]`` coerced to float (NaN->default), or a constant Series.

    A missing column degrades gracefully to a constant ``default`` so the
    classifier runs on tables that lack an optional forcing or attribute.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(float(default), index=df.index, dtype=float)


def _swe_delta(df: pd.DataFrame, swe_col: str = "swe") -> pd.Series:
    """Per-gauge SWE increment between consecutive periods (chronological).

    Computed on a date-sorted copy and realigned to the original row order, so
    the output preserves the caller's index without reordering the table.
    """
    if swe_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    if {"code", "date"}.issubset(df.columns):
        s = df[["code", "date", swe_col]].sort_values(["code", "date"])
        d = s.groupby("code")[swe_col].diff()
        return d.reindex(df.index).fillna(0.0)
    return df[swe_col].diff().fillna(0.0)


def _per_gauge_high(df: pd.DataFrame, col: str, quantile: float, floor: float) -> pd.Series:
    """Boolean mask: ``col`` is "high" for its gauge.

    The threshold is ``max(floor, q-quantile of the gauge's positive values)``.
    Gauges with no positive values get an infinite threshold (never high).
    """
    x = _num(df, col, 0.0)
    pos = x.where(x > 0)
    if "code" in df.columns:
        thr = pos.groupby(df["code"]).transform(lambda s: s.quantile(quantile))
    else:
        thr = pd.Series(pos.quantile(quantile), index=df.index)
    thr = thr.fillna(np.inf).clip(lower=floor)
    return x >= thr


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Label every row with its dominant hydrological regime.

    Parameters
    ----------
    df : pandas.DataFrame
        A modelling table (see :mod:`sbc.schemas`).  Uses ``t2m_mean``, ``swe``,
        ``smlt``, ``tp`` and ``date`` when available, plus the static attributes
        ``glacier_frac`` / ``snow_frac``; any missing column degrades gracefully.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with two added columns: ``regime`` (an ordered
        categorical over :data:`REGIMES`) and ``regime_id`` (its integer code).

    Notes
    -----
    Rules are evaluated in priority order ``accumulation -> rain_on_snow ->
    melt_freshet -> glacier_melt``, defaulting to ``recession``.  ``accumulation``
    (cold) and the warm-season regimes are mutually exclusive by temperature;
    ``rain_on_snow`` (snowpack present) and ``glacier_melt`` (snowpack gone) are
    mutually exclusive by SWE.  Rain-on-snow is ranked above the generic freshet
    because it is the more specific, flood-relevant condition.
    """
    t2m = _num(df, "t2m_mean", 0.0)
    swe = _num(df, "swe", 0.0)
    tp = _num(df, "tp", 0.0)
    glacier_frac = _num(df, "glacier_frac", 0.0)
    snow_frac = _num(df, "snow_frac", 1.0)   # unknown -> assume snow-influenced
    dswe = _swe_delta(df, "swe")

    if "date" in df.columns:
        doy = df["date"].dt.dayofyear.astype(float)
    else:
        doy = pd.Series(0.0, index=df.index)

    # primitive physical conditions ----------------------------------------
    cold = t2m < T_FREEZE_C
    warm = t2m > T_MELT_C
    warm_ice = t2m > T_GLACIER_C
    swe_present = swe > SWE_PRESENT_MM       # substantial seasonal snowpack
    snow_on_ground = swe > SWE_TRACE_MM      # any non-trivial pack present
    swe_gone = swe <= SWE_TRACE_MM
    snowy = snow_frac >= SNOW_FRAC_MIN
    glaciated = glacier_frac > GLACIER_FRAC_MIN
    warm_season = (doy >= WARM_SEASON_DOY[0]) & (doy <= WARM_SEASON_DOY[1])
    building = (dswe > SWE_RISE_MM) | (tp > TP_FLOOR_MM)

    melt_high = _per_gauge_high(df, "smlt", SMLT_HIGH_QUANTILE, SMLT_FLOOR_MM)
    rain_high = _per_gauge_high(df, "tp", TP_HIGH_QUANTILE, TP_FLOOR_MM)

    # regime masks ----------------------------------------------------------
    # accumulation is a cold-season *state*: sub-freezing with a snowpack on the
    # ground or actively forming (fresh snowfall / rising SWE).
    accumulation = cold & snowy & (snow_on_ground | building)
    rain_on_snow = warm & swe_present & rain_high & snowy
    melt_freshet = warm & swe_present & melt_high & snowy
    glacier_melt = warm_ice & swe_gone & glaciated & warm_season

    labels = np.select(
        [accumulation.to_numpy(), rain_on_snow.to_numpy(),
         melt_freshet.to_numpy(), glacier_melt.to_numpy()],
        ["accumulation", "rain_on_snow", "melt_freshet", "glacier_melt"],
        default="recession",
    )

    cat = pd.Categorical(labels, categories=REGIMES, ordered=True)
    out = df.copy()
    out[REGIME_COL] = pd.Series(cat, index=df.index)
    out[REGIME_ID_COL] = cat.codes.astype("int16")

    if log.isEnabledFor(10):  # DEBUG
        frac = out[REGIME_COL].value_counts(normalize=True).round(3).to_dict()
        log.debug("regime fractions: %s", frac)
    return out


def regime_onehot(df: pd.DataFrame, prefix: str = "regime_") -> pd.DataFrame:
    """One-hot indicator columns for the regime label, one per :data:`REGIMES`.

    Parameters
    ----------
    df : pandas.DataFrame
        Table that already carries a ``regime`` column, or any modelling table
        (in which case :func:`classify_regimes` is applied first).
    prefix : str, default ``"regime_"``
        Prefix for the indicator column names.

    Returns
    -------
    pandas.DataFrame
        Integer 0/1 indicators with columns ``{prefix}{regime}`` in the order of
        :data:`REGIMES`, aligned to ``df.index``.  Every regime gets a column
        even if absent from the data, and the rows sum to exactly one.
    """
    reg = df[REGIME_COL] if REGIME_COL in df.columns else classify_regimes(df)[REGIME_COL]
    reg = reg.astype(str).to_numpy()
    data = {f"{prefix}{r}": (reg == r).astype("int8") for r in REGIMES}
    return pd.DataFrame(data, index=df.index)


if __name__ == "__main__":
    from sbc.synthetic import generate

    df = generate(n_basins=3, gauges_per_basin=(2, 3), years=4, seed=7)
    out = classify_regimes(df)

    counts = out[REGIME_COL].value_counts()
    print("regime value_counts:")
    print(counts.to_string())

    oh = regime_onehot(out)
    assert out[REGIME_COL].notna().all(), "some rows were left unlabelled"
    assert out[REGIME_ID_COL].between(0, len(REGIMES) - 1).all(), "bad regime_id"
    assert list(oh.columns) == [f"regime_{r}" for r in REGIMES], "onehot columns"
    assert (oh.sum(axis=1) == 1).all(), "one-hot is not mutually exclusive"

    print(
        f"OK: {len(out)} rows | {out[REGIME_COL].nunique()} regimes present | "
        f"all rows labelled | one-hot exclusive"
    )
