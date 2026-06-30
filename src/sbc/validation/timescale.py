"""Skill-by-timescale decomposition and ALE attribution of GloFAS error.

A basin-average KGE' hides *where in the frequency spectrum* a bias correction
actually helps.  GloFAS-ERA5 is known to track the seasonal snowmelt-freshet
volume well while mistiming and damping the sub-monthly hydrograph, so a single
skill number conflates a strong low-frequency component with a weak
high-frequency one.  Following the time-frequency decomposition evaluation of
Rasiya Koya & Roy (HESS 28:3597, 2024) -- a discrete-wavelet multi-resolution
analysis of observed and reanalysis discharge into dyadic timescale bands, scored
band-by-band, complemented by an Accumulated-Local-Effects (ALE) attribution of
the residual error onto catchment attributes -- this module turns that intuition
into evidence.  Their headline (median KGE rising from ~0.02 at the 2-day scale to
~0.47 at the 128-day scale) is exactly the *GloFAS strong at seasonal, weak at
sub-monthly* signature we report; here we additionally show, per band, **where the
correction adds skill** by scoring the corrected series against the raw one.

Method
------
1. :func:`multiresolution_decompose` performs an additive multi-resolution
   analysis (MRA) of a 1-D series into ``J`` band-pass *detail* components
   ``D_1 .. D_J`` (capturing the dyadic timescales ``~2^1 .. 2^J`` sampling
   periods) plus a final low-pass *approximation* ``A_J`` (the seasonal/trend
   band).  The components sum *exactly* back to the input (telescoping), so no
   variance is lost.  A discrete wavelet transform (``pywt.mra``, Haar by
   default) is used when PyWavelets is installed; otherwise a dependency-free
   *a-trous* Haar / centered-moving-average MRA is used -- the two give the same
   additive band structure.

2. :func:`skill_by_timescale` decomposes the observed and the simulated
   (raw and/or corrected) discharge per gauge, scores each band with the
   project's :mod:`sbc.validation.metrics` (KGE', NSE, correlation; volume bias
   is neutralised per band by re-referencing both series to the observed mean so
   the score reflects timing/amplitude skill *at that scale*), and reports the
   across-gauge median.  Passing several ``sim_col`` columns scores raw and
   corrected side by side, exposing the frequency bands in which the correction
   helps.

3. :func:`ale_by_attribute` wraps the existing PUR attribution
   (:mod:`sbc.explain.pur_attribution`) to compute the 1-D Accumulated Local
   Effect (Apley & Zhu, 2020) of a single static catchment attribute on a
   per-gauge skill/failure target -- the marginal "effect curve" that
   complements the rank-correlation / ridge ranking already in that module.

All heavy / optional imports (``pywt``, ``scipy``, ``matplotlib``) are deferred
into the functions that use them; figures are written on the headless ``Agg``
backend to :pyattr:`sbc.config.Paths.figures` (``results/figures``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import COL_DATE, COL_GAUGE, PATHS
from ..schemas import OBS_COL, SIM_COL
from ..utils import get_logger
from . import metrics as M

log = get_logger(__name__)

#: Largest dyadic scale targeted by default, in days (matches HESS 28:3597's
#: 2-day .. 128-day ladder); the number of MRA levels is derived from this and
#: the table's sampling period.
DEFAULT_MAX_SCALE_DAYS: float = 128.0
#: Hard cap on the number of MRA levels (2^7 = 128 sampling periods).
MAX_LEVELS: int = 7
#: Tiny positive floor for the observed-mean re-reference offset.
_EPS: float = 1e-9

__all__ = [
    "multiresolution_decompose",
    "band_scales",
    "skill_by_timescale",
    "ale_by_attribute",
    "save_timescale_plot",
    "save_ale_plot",
]


# --------------------------------------------------------------------------- #
#  Multi-resolution decomposition                                             #
# --------------------------------------------------------------------------- #
def _feasible_levels(n: int, requested: int | None) -> int:
    """Number of dyadic levels supported by a length-``n`` series."""
    cap = max(1, int(np.floor(np.log2(max(n, 2)))) - 1)
    cap = min(cap, MAX_LEVELS)
    if requested is None:
        return cap
    return int(np.clip(int(requested), 1, cap))


def _centered_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Length-preserving centered moving average with shrinking edge windows.

    Implemented with a prefix sum so it is O(n); at the boundaries the window is
    truncated to the available samples (a "nearest"-style edge), which keeps the
    output the same length as the input and the MRA telescoping exact.
    """
    n = x.size
    if window <= 1 or n == 0:
        return x.astype(float).copy()
    k = window // 2
    csum = np.concatenate([[0.0], np.cumsum(x.astype(float))])
    idx = np.arange(n)
    lo = np.maximum(0, idx - k)
    hi = np.minimum(n, idx + k + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _atrous_mra(x: np.ndarray, levels: int) -> np.ndarray:
    """Dependency-free a-trous Haar MRA: detail bands ``D_1..D_J`` then ``A_J``.

    ``s_0 = x``; ``s_j = MA(s_{j-1}, 2^j)``; ``D_j = s_{j-1} - s_j``;
    ``A_J = s_J``.  By construction ``sum_j D_j + A_J == x`` exactly.
    """
    n = x.size
    comps = np.empty((levels + 1, n), float)
    prev = x.astype(float).copy()
    for j in range(1, levels + 1):
        smooth = _centered_moving_average(prev, 2 ** j)
        comps[j - 1] = prev - smooth
        prev = smooth
    comps[levels] = prev
    return comps


def _pywt_mra(x: np.ndarray, levels: int, wavelet: str) -> np.ndarray:
    """Wavelet MRA via :func:`pywt.mra`, reordered to ``[D_1..D_J, A_J]``.

    Raises if PyWavelets (or its ``mra`` helper) is unavailable, so the caller
    can fall back to :func:`_atrous_mra`.
    """
    import pywt  # noqa: F401  (optional dependency)

    if not hasattr(pywt, "mra"):
        raise RuntimeError("pywt.mra not available in this PyWavelets build")
    n = x.size
    pad = (-n) % (2 ** levels)  # length must be a multiple of 2^levels for SWT
    xp = np.pad(x.astype(float), (0, pad), mode="reflect") if pad else x.astype(float)
    parts = pywt.mra(xp, wavelet, level=levels, transform="swt")  # [A_J, D_J..D_1]
    approx = np.asarray(parts[0], float)[:n]
    details = [np.asarray(p, float)[:n] for p in parts[1:]][::-1]  # -> D_1..D_J
    return np.vstack(details + [approx])


def multiresolution_decompose(series, levels: int | None = None, *,
                              wavelet: str = "haar") -> np.ndarray:
    """Additive multi-resolution decomposition of a 1-D series into dyadic bands.

    Parameters
    ----------
    series : array_like, shape (n,)
        The signal to decompose (e.g. a single gauge's discharge time series,
        already date-ordered).  Non-finite samples are linearly interpolated
        before decomposition and do not corrupt neighbouring bands.
    levels : int, optional
        Number of dyadic *detail* levels ``J``.  Defaults to the largest value
        the series length supports (capped at :data:`MAX_LEVELS` = 7, i.e. a
        128-period coarsest scale).  Always clipped to a feasible value.
    wavelet : str, default ``"haar"``
        Wavelet passed to :func:`pywt.mra` when PyWavelets is installed; ignored
        by the moving-average fallback (which is Haar-equivalent).

    Returns
    -------
    numpy.ndarray, shape (J + 1, n)
        Row ``j`` (``0 <= j < J``) is the band-pass detail ``D_{j+1}`` isolating
        the dyadic timescale ``~2^{j+1}`` sampling periods; the final row is the
        low-pass approximation ``A_J`` (the seasonal/trend band).  The rows sum
        back to the (gap-filled) input.

    Notes
    -----
    Uses :func:`pywt.mra` (discrete-wavelet MRA) when available and otherwise a
    dependency-free a-trous Haar / centered-moving-average MRA; both yield an
    exact additive band decomposition.
    """
    x = np.asarray(series, float).ravel()
    n = x.size
    if n == 0:
        return np.zeros((1, 0), float)

    # gap-fill so the smoother does not propagate NaNs across the record
    finite = np.isfinite(x)
    if not finite.all():
        if not finite.any():
            return np.zeros((_feasible_levels(n, levels) + 1, n), float)
        xi = np.arange(n, dtype=float)
        x = np.interp(xi, xi[finite], x[finite])

    J = _feasible_levels(n, levels)
    try:
        comps = _pywt_mra(x, J, wavelet)
    except Exception as exc:  # pragma: no cover - exercised only without pywt
        log.debug("multiresolution_decompose: wavelet path unavailable (%s); "
                  "using moving-average MRA", exc)
        comps = _atrous_mra(x, J)
    return comps


def band_scales(levels: int, dt_days: float = 1.0) -> pd.DataFrame:
    """Nominal timescale label for each band of a ``levels``-deep MRA.

    Parameters
    ----------
    levels : int
        Number of detail levels ``J`` (so ``J + 1`` bands including the trend).
    dt_days : float, default 1.0
        Sampling period in days (e.g. ``~10`` for decadal, ``1`` for daily),
        used to convert dyadic *period* scales into approximate day scales.

    Returns
    -------
    pandas.DataFrame
        Columns ``band`` (0-based index), ``scale_periods`` (``2^{band+1}`` for
        detail bands, ``inf`` for the trend), ``scale_days`` and a human-readable
        ``label`` (``"~N d"`` / ``"trend (>~N d)"``).
    """
    rows = []
    for j in range(1, levels + 1):
        sp = float(2 ** j)
        rows.append({"band": j - 1, "scale_periods": sp,
                     "scale_days": sp * dt_days, "label": f"~{sp * dt_days:g} d"})
    trend_days = float(2 ** levels) * dt_days
    rows.append({"band": levels, "scale_periods": float("inf"),
                 "scale_days": float("inf"), "label": f"trend (>~{trend_days:g} d)"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Skill by timescale                                                         #
# --------------------------------------------------------------------------- #
def _median_dt_days(dates: pd.Series) -> float:
    """Median spacing between consecutive *distinct* dates, in days (>=1).

    Uses unique dates so that pooling several gauges (which share the same period
    calendar) does not inject spurious zero gaps and collapse the estimate.
    """
    u = pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values()
    d = u.diff().dropna()
    if d.empty:
        return 1.0
    days = float(np.median(d.dt.total_seconds()) / 86400.0)
    return max(days, 1.0)


def _band_skill_one(obs: np.ndarray, sim: np.ndarray, levels: int) -> pd.DataFrame:
    """Per-band KGE'/NSE/r of ``sim`` vs ``obs`` for one (already-clean) series.

    Each band is scored after re-referencing both the observed and the simulated
    band-pass component to the observed mean flow, so the per-band score reflects
    timing/amplitude agreement at that scale rather than the overall volume bias
    (which is a level-0 quantity, not a frequency band).  NSE is invariant to the
    common shift; KGE' becomes well defined on the otherwise zero-mean band.
    """
    co = multiresolution_decompose(obs, levels)
    cs = multiresolution_decompose(sim, levels)
    offset = max(float(np.mean(obs)), _EPS)
    var_o = co.var(axis=1)
    tot = float(var_o.sum()) or np.nan
    rows = []
    for b in range(co.shape[0]):
        ob = co[b] + offset
        sb = cs[b] + offset
        k = M.kge_prime(ob, sb)
        rows.append({
            "band": b,
            "kge": k["kge"], "r": k["r"], "nse": M.nse(ob, sb),
            "var_frac_obs": float(var_o[b] / tot) if np.isfinite(tot) else np.nan,
        })
    return pd.DataFrame(rows)


def skill_by_timescale(df: pd.DataFrame, obs_col: str = OBS_COL,
                       sim_col: str | Sequence[str] = SIM_COL, by: str | None = COL_GAUGE,
                       *, levels: int | None = None, date_col: str = COL_DATE,
                       max_scale_days: float = DEFAULT_MAX_SCALE_DAYS) -> pd.DataFrame:
    """KGE'/NSE of simulated vs observed discharge, resolved by timescale band.

    Decomposes the observed and the simulated discharge of every gauge into
    dyadic timescale bands and scores each band, then reports the across-gauge
    median -- revealing *where in frequency* the simulation (raw GloFAS, a
    corrected series, or both) tracks the observations.  Passing several
    ``sim_col`` columns scores them side by side, so one can read off the bands
    in which a correction adds skill.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table with ``date``, the gauge id ``by`` (when grouping) and
        the discharge columns.
    obs_col : str, default :data:`sbc.schemas.OBS_COL`
        Observed-discharge column.
    sim_col : str or sequence of str, default :data:`sbc.schemas.SIM_COL`
        One or more simulated-discharge columns to score (e.g. raw ``q_glofas``
        and a corrected ``q_pred``).
    by : str or None, default ``"code"``
        Gauge-grouping column; per-gauge band skills are aggregated by median.
        ``None`` scores the table as a single pooled series.
    levels : int, optional
        Number of dyadic detail levels.  Defaults to the value whose coarsest
        scale is closest to ``max_scale_days`` given the sampling period,
        clipped to a feasible range.
    date_col : str, default ``"date"``
        Datetime column used to order each series and infer the sampling period.
    max_scale_days : float, default 128
        Target coarsest detail scale in days (drives the default ``levels``).

    Returns
    -------
    pandas.DataFrame
        One row per (``sim``, band) with the across-gauge median ``kge`` / ``nse``
        / ``r``, the median observed-variance fraction ``var_frac_obs`` in that
        band, the band ``label`` / ``scale_days`` / ``scale_periods`` and the
        number of gauges scored ``n_gauges``.  Sorted by simulation then ascending
        timescale (the seasonal *trend* band last).  ``df.attrs["dt_days"]`` and
        ``df.attrs["levels"]`` record the resolved sampling period and depth.
    """
    sims = [sim_col] if isinstance(sim_col, str) else list(sim_col)
    missing = [c for c in [obs_col, *sims] if c not in df.columns]
    if missing:
        raise KeyError(f"skill_by_timescale: missing column(s) {missing}")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    dt_days = _median_dt_days(work[date_col])

    # resolve the MRA depth from the target coarsest scale and the data length
    desired = int(np.clip(round(np.log2(max(max_scale_days / dt_days, 2.0))), 1, MAX_LEVELS))
    if by is not None and by in work.columns:
        groups = [g for _, g in work.groupby(by, sort=False)]
    else:
        groups = [work]
    min_len = min((int(np.isfinite(g[obs_col].to_numpy(float)).sum()) for g in groups),
                  default=0)
    used = _feasible_levels(max(min_len, 2), min(desired, levels) if levels else desired)
    if levels is not None:
        used = _feasible_levels(max(min_len, 2), levels)

    per_gauge: list[pd.DataFrame] = []
    for g in groups:
        gg = g.sort_values(date_col)
        obs = gg[obs_col].to_numpy(float)
        for sc in sims:
            sim = gg[sc].to_numpy(float)
            mask = np.isfinite(obs) & np.isfinite(sim)
            if int(mask.sum()) <= 2 ** used:
                continue
            tab = _band_skill_one(obs[mask], sim[mask], used)
            tab["sim"] = sc
            per_gauge.append(tab)

    scales = band_scales(used, dt_days)
    if not per_gauge:
        log.warning("skill_by_timescale: no gauge had enough data for %d levels", used)
        out = scales.assign(sim=sims[0], kge=np.nan, nse=np.nan, r=np.nan,
                            var_frac_obs=np.nan, n_gauges=0)
    else:
        long = pd.concat(per_gauge, ignore_index=True)
        agg = (long.groupby(["sim", "band"], as_index=False)
               .agg(kge=("kge", "median"), nse=("nse", "median"), r=("r", "median"),
                    var_frac_obs=("var_frac_obs", "median"),
                    n_gauges=("kge", lambda s: int(np.isfinite(s).sum()))))
        out = agg.merge(scales, on="band", how="left")

    out = out.sort_values(["sim", "scale_periods"], kind="stable").reset_index(drop=True)
    cols = ["sim", "band", "label", "scale_periods", "scale_days",
            "kge", "nse", "r", "var_frac_obs", "n_gauges"]
    out = out[[c for c in cols if c in out.columns]]
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].round(4)
    out.attrs["dt_days"] = dt_days
    out.attrs["levels"] = used
    log.info("skill_by_timescale: %d sims x %d bands (dt~%.1f d, J=%d)",
             len(sims), used + 1, dt_days, used)
    return out


# --------------------------------------------------------------------------- #
#  ALE attribution (wraps sbc.explain.pur_attribution)                        #
# --------------------------------------------------------------------------- #
def _default_target(table: pd.DataFrame) -> str:
    """Pick a sensible per-gauge skill/failure target column."""
    for c in ("delta_kge", "kge", "d_abs_pbias", "nse"):
        if c in table.columns:
            return c
    num = [c for c in table.columns
           if c != "code" and pd.api.types.is_numeric_dtype(table[c])]
    if not num:
        raise KeyError("ale_by_attribute: no numeric target column found")
    return num[0]


def ale_by_attribute(per_gauge_table: pd.DataFrame, static: pd.DataFrame,
                     attribute: str, *, target: str | None = None,
                     n_bins: int = 8, code_col: str = "code") -> pd.DataFrame:
    """1-D Accumulated Local Effect of a catchment attribute on per-gauge skill.

    Wraps the attribute resolution of :mod:`sbc.explain.pur_attribution`
    (so the synthetic ``glacier_frac`` / ``elev_m`` and the real HydroATLAS
    aliases both work unchanged) and computes the Accumulated Local Effect
    (Apley & Zhu, 2020) of ``attribute`` on a per-gauge ``target``.  For a single
    feature the ALE main effect reduces to the centred conditional mean
    ``E[target | attribute]``; here it is estimated over quantile bins and
    obtained by accumulating the local (bin-to-bin) effects, then centring on the
    sample.  This is the marginal *effect curve* that complements the
    rank-correlation / ridge ranking already produced by
    :func:`sbc.explain.pur_attribution.attribute_regression`.

    Parameters
    ----------
    per_gauge_table : pandas.DataFrame
        One row per gauge with a ``code`` column and the ``target`` skill/failure
        column (e.g. the ``per_gauge`` / failure table of the PUR attribution, or
        any per-gauge metric table).
    static : pandas.DataFrame
        Static-attribute table passed to
        :func:`sbc.explain.pur_attribution.resolve_attributes`.
    attribute : str
        Canonical attribute name (resolved through its documented aliases).
    target : str, optional
        Per-gauge target column; defaults to the first of ``delta_kge`` / ``kge``
        / ``d_abs_pbias`` / ``nse`` present, else the first numeric column.
    n_bins : int, default 8
        Number of quantile bins along the attribute (reduced automatically when
        the gauges or distinct values are few).
    code_col : str, default ``"code"``
        Gauge-id column used for the join.

    Returns
    -------
    pandas.DataFrame
        One row per occupied bin with ``x_left`` / ``x_right`` / ``x_center``
        (attribute edges/centre), ``n`` gauges, the per-bin ``local_effect``
        (first differences of the conditional mean; ``NaN`` for the first bin)
        and the centred ``ale``.  ``df.attrs`` records ``attribute`` / ``source``
        / ``label`` / ``target`` / ``n_gauges``.  Empty when fewer than three
        gauges join or the attribute does not resolve.
    """
    from ..explain.pur_attribution import resolve_attributes  # deferred (avoids cycle)

    attr = resolve_attributes(static, [attribute], code_col=code_col)
    if attribute not in attr.columns:
        log.warning("ale_by_attribute: attribute %r did not resolve in static table",
                    attribute)
        return pd.DataFrame()
    resolved = attr.attrs.get("resolved", {}).get(attribute, {})

    pg = per_gauge_table.copy()
    pg[code_col] = pg[code_col].astype(str)
    tgt = target or _default_target(pg)
    if tgt not in pg.columns:
        raise KeyError(f"ale_by_attribute: target {tgt!r} not in per_gauge_table")

    merged = pg[[code_col, tgt]].merge(attr[[code_col, attribute]], on=code_col, how="inner")
    merged = merged[np.isfinite(merged[attribute]) & np.isfinite(merged[tgt])]
    x = merged[attribute].to_numpy(float)
    y = merged[tgt].to_numpy(float)
    if x.size < 3 or np.unique(x).size < 2:
        log.warning("ale_by_attribute: too few finite gauges (%d) for %r", x.size, attribute)
        return pd.DataFrame()

    # quantile bin edges (deduplicated); cap bins by distinct values and sample size
    k = int(np.clip(n_bins, 2, min(max(2, x.size // 2), np.unique(x).size)))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, k + 1)))
    if edges.size < 3:
        edges = np.array([x.min(), np.median(x), x.max()])
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)

    rows = []
    for b in range(edges.size - 1):
        sel = idx == b
        n_b = int(sel.sum())
        rows.append({"bin": b, "x_left": float(edges[b]), "x_right": float(edges[b + 1]),
                     "x_center": float(0.5 * (edges[b] + edges[b + 1])), "n": n_b,
                     "mean": float(np.mean(y[sel])) if n_b else np.nan})
    out = pd.DataFrame(rows)
    occupied = out["n"] > 0
    out = out[occupied].reset_index(drop=True)

    means = out["mean"].to_numpy(float)
    local = np.diff(means, prepend=means[0])      # local effects; first bin -> 0
    acc = np.cumsum(local)                          # accumulated effect
    weights = out["n"].to_numpy(float)
    centre = float(np.average(acc, weights=weights))
    out["local_effect"] = local
    out.loc[0, "local_effect"] = np.nan
    out["ale"] = acc - centre
    out = out.drop(columns="mean")
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].round(5)

    out.attrs.update({"attribute": attribute, "target": tgt,
                      "source": resolved.get("source", attribute),
                      "label": resolved.get("label", attribute),
                      "n_gauges": int(x.size)})
    log.info("ale_by_attribute: %r on %r over %d gauges, %d bins (range %.3g)",
             attribute, tgt, x.size, len(out),
             float(out["ale"].max() - out["ale"].min()) if len(out) else 0.0)
    return out


# --------------------------------------------------------------------------- #
#  Plotting (Agg-safe, writes to results/figures)                              #
# --------------------------------------------------------------------------- #
def _new_axes(figsize: tuple[float, float]):
    """Return a fresh ``(fig, ax)`` on the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def _resolve_path(path: str | Path | None, default_name: str) -> Path:
    """Return a writable PNG path, defaulting under ``results/figures``."""
    if path is None:
        PATHS.figures.mkdir(parents=True, exist_ok=True)
        return PATHS.figures / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _finite_scale(table: pd.DataFrame) -> np.ndarray:
    """X positions for plotting: replace the trend band's inf scale with 2x max."""
    sd = table["scale_days"].to_numpy(float)
    fin = sd[np.isfinite(sd)]
    cap = (fin.max() * 2.0) if fin.size else 1.0
    return np.where(np.isfinite(sd), sd, cap)


def save_timescale_plot(table: pd.DataFrame, *, path: str | Path | None = None,
                        metric: str = "kge", title: str | None = None) -> Path:
    """Plot a per-timescale skill metric versus scale for each simulation.

    Draws ``metric`` (default KGE') against the band timescale on a log-scale
    x-axis, one line per ``sim`` column, with the seasonal *trend* band placed at
    the right edge.  This is the figure that makes the *strong-at-seasonal,
    weak-at-sub-monthly* signature -- and the band(s) where a correction helps --
    legible at a glance.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`skill_by_timescale`.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to ``results/figures/timescale_skill.png``).
    metric : str, default ``"kge"``
        Column to plot (``"kge"``, ``"nse"``, ``"r"`` or ``"var_frac_obs"``).
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    if metric not in table.columns:
        raise KeyError(f"save_timescale_plot: metric {metric!r} not in table")
    out = _resolve_path(path, f"timescale_{metric}.png")
    fig, ax = _new_axes((7.0, 4.6))
    labels = (table.drop_duplicates("band").sort_values("scale_periods")["label"].tolist())
    for sim, sub in table.groupby("sim"):
        sub = sub.sort_values("scale_periods")
        ax.plot(_finite_scale(sub), sub[metric].to_numpy(float),
                marker="o", ms=5, lw=1.6, label=str(sim))
    ax.set_xscale("log")
    xt = _finite_scale(table.drop_duplicates("band").sort_values("scale_periods"))
    ax.set_xticks(xt)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize="small")
    if metric in ("kge", "nse", "r"):
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("timescale band (approx. days; rightmost = seasonal trend)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"GloFAS skill by timescale ({metric.upper()})")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(fontsize="small", title="series")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote timescale-skill figure -> %s", out)
    return out


def save_ale_plot(ale_table: pd.DataFrame, *, path: str | Path | None = None,
                  title: str | None = None) -> Path:
    """Plot the 1-D ALE effect curve from :func:`ale_by_attribute`.

    Parameters
    ----------
    ale_table : pandas.DataFrame
        Output of :func:`ale_by_attribute` (its ``attrs`` carry the axis labels).
    path : str or pathlib.Path, optional
        Destination PNG (defaults to ``results/figures/ale_<attribute>.png``).
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    attribute = ale_table.attrs.get("attribute", "attribute")
    label = ale_table.attrs.get("label", attribute)
    target = ale_table.attrs.get("target", "target")
    out = _resolve_path(path, f"ale_{attribute}.png")
    fig, ax = _new_axes((6.2, 4.4))
    x = ale_table["x_center"].to_numpy(float)
    ax.plot(x, ale_table["ale"].to_numpy(float), marker="o", ms=5, lw=1.6, color="C0")
    ax.scatter(x, ale_table["ale"].to_numpy(float),
               s=12 + 10 * ale_table["n"].to_numpy(float), color="C0", alpha=0.4, zorder=2)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(label)
    ax.set_ylabel(f"ALE on {target}")
    ax.set_title(title or f"Accumulated local effect of {label} on {target}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote ALE figure -> %s", out)
    return out


# --------------------------------------------------------------------------- #
#  Self-test (small synthetic decadal; < 3 min, no real matrix)               #
# --------------------------------------------------------------------------- #
def _selftest() -> None:  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.quantile_mapping import LinearScalingCorrector
    from ..schemas import PRED_COL, validate
    from ..synthetic import generate
    from ..validation.splits import temporal_split

    # --- small synthetic decadal table -> features -> regimes -------------- #
    raw = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(validate(raw), scale="decadal")).reset_index(drop=True)
    print(f"[timescale] table: {len(df)} rows | {df['code'].nunique()} gauges")

    # --- a simple per-gauge linear-scaling correction (leakage-safe fit) --- #
    tr, _ = temporal_split(df, test_frac=0.3)
    model = LinearScalingCorrector().fit(df[tr].reset_index(drop=True))
    df[PRED_COL] = model.predict(df)

    # --- decomposition sanity: bands sum back to the signal ---------------- #
    s = df[df["code"] == df["code"].iloc[0]].sort_values("date")[OBS_COL].to_numpy(float)
    comps = multiresolution_decompose(s, levels=4)
    recon_err = float(np.max(np.abs(comps.sum(axis=0) - np.interp(
        np.arange(s.size, dtype=float),
        np.flatnonzero(np.isfinite(s)).astype(float), s[np.isfinite(s)]))))
    print(f"[timescale] MRA bands={comps.shape[0]} (J=4)  reconstruction max|err|={recon_err:.2e}")
    assert comps.shape[0] == 5, "expected J+1 = 5 bands"
    assert recon_err < 1e-6, "MRA components must sum back to the signal"

    # --- skill by timescale: raw vs scaling-corrected ---------------------- #
    sk = skill_by_timescale(df, OBS_COL, [SIM_COL, PRED_COL], by="code")
    print(f"[timescale] dt~{sk.attrs['dt_days']:.1f} d | J={sk.attrs['levels']} | rows={len(sk)}")
    print(sk.to_string(index=False))

    raw_b = sk[sk["sim"] == SIM_COL].sort_values("scale_periods")
    cor_b = sk[sk["sim"] == PRED_COL].sort_values("scale_periods")
    fine_raw = float(raw_b["kge"].iloc[0])
    coarse_raw = float(raw_b["kge"].iloc[-1])
    print(f"[timescale] raw KGE': finest band={fine_raw:+.3f} -> trend band={coarse_raw:+.3f} "
          f"(strong-at-seasonal, weak-at-sub-monthly)")
    print(f"[timescale] mean KGE' across bands: raw={raw_b['kge'].mean():+.3f} "
          f"corrected={cor_b['kge'].mean():+.3f}")

    assert set(sk["sim"]) == {SIM_COL, PRED_COL}, "both series must be scored"
    assert sk["kge"].notna().any() and sk["nse"].notna().any(), "no band skill computed"
    assert (sk["n_gauges"] > 0).any(), "no gauges scored"
    # GloFAS should track the seasonal trend better than the finest band
    assert coarse_raw >= fine_raw - 1e-6, "expected coarser bands to be at least as skilful"

    # --- ALE attribution wrapping the existing PUR attribution ------------- #
    static_cols = ["area_km2", "elev_m", "slope_deg", "snow_frac", "glacier_frac", "aridity"]
    static = df.groupby("code", as_index=False)[static_cols].first()
    pg = M.evaluate_by_group(df.assign(**{PRED_COL: df[PRED_COL]}), OBS_COL, PRED_COL)
    raw_pg = M.evaluate_by_group(df, OBS_COL, SIM_COL)
    pg["delta_kge"] = pg["kge"].to_numpy() - raw_pg["kge"].to_numpy()
    ale = ale_by_attribute(pg, static, "elev_m", target="delta_kge", n_bins=5)
    print(f"[timescale] ALE(elev_m -> delta_kge): {len(ale)} bins, "
          f"range={float(ale['ale'].max() - ale['ale'].min()):.3f} over "
          f"{ale.attrs['n_gauges']} gauges")
    assert not ale.empty and {"x_center", "ale", "n"}.issubset(ale.columns), "ALE malformed"
    # data-weighted mean of the ALE effect is ~0 by construction (rounding aside)
    assert abs(float(np.average(ale['ale'], weights=ale['n']))) < 1e-3, "ALE not centred"

    # --- figures (Agg-safe) ------------------------------------------------ #
    f1 = save_timescale_plot(sk, path=PATHS.figures / "timescale_skill_selftest.png",
                             title="Skill by timescale (synthetic decadal)")
    f2 = save_ale_plot(ale, path=PATHS.figures / "ale_elev_selftest.png")
    assert f1.exists() and f2.exists(), "figures not written"
    print(f"[timescale] figures: {f1.name}, {f2.name}")
    print("[timescale] SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
