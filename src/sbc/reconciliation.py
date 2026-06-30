"""CRPS-coherent cross-temporal probabilistic reconciliation of the
10-daily -> 1-dekada discharge hierarchy.

The paper advertises a *multi-scale* bias-correction framework that operates on
both the native **daily** GloFAS-ERA5 stream and the Central-Asian **decadal**
(10-day "dekada") discharge bulletins.  :mod:`sbc.multiscale` already quantifies
how skill transfers across the two resolutions and *diagnoses* their aggregation
incoherence, but a diagnostic is not a model: a daily-corrected hydrograph and a
separately-corrected decadal bulletin remain two independent products whose
decade means disagree (~0.2 % in the CA discharge record).  This module closes
that gap, turning "multi-scale" from a label / consistency *check* into a single
**reconciled** model whose daily-corrected series aggregates *exactly* to the
decadal-corrected bulletin -- in both point and full predictive-distribution
form, and without sacrificing daily skill.

The temporal hierarchy and the aggregation map S
-------------------------------------------------
A dekada is the mean of the ~10 daily discharges that fall inside it, so the
hierarchy is the temporal aggregation studied by Athanasopoulos et al. (2017,
*EJOR* "Forecasting with temporal hierarchies").  Stacking the bottom-level daily
series ``b`` (one block per gauge-decade) the constraint is the linear map

    decadal_mean = S @ b ,          S in R^{D x N},

where row ``d`` of ``S`` holds the averaging weights ``1 / n_d`` over exactly the
``n_d`` daily cells of decade ``d`` (``S`` is block-diagonal: every daily cell
belongs to one decade).  :func:`aggregate_constraint` builds ``S`` and the
*coherence residual* ``decadal_pred - S @ daily_pred`` -- the volume-accounting
mismatch reconciliation must drive to zero.

Point reconciliation (MinT / OLS)
---------------------------------
:func:`reconcile_point` adjusts the daily-corrected series so that it aggregates
*exactly* to the decadal-corrected bulletin while moving as little as possible,
in the trace-minimising sense of Wickramasuriya et al. (2019, *JASA*
"Optimal forecast reconciliation ... (MinT)").  Treating the operational decadal
bulletin as the authoritative aggregate, the reconciled daily vector solves the
equality-constrained generalised least squares

    min_b (b - daily_pred)' W^{-1} (b - daily_pred)   s.t.  S b = decadal_pred ,

whose closed form is the MinT projection

    b~ = daily_pred + W S' (S W S')^{-1} (decadal_pred - S daily_pred) .

``W = I`` recovers ordinary least squares (every day shifted equally to close the
per-decade gap); ``W = diag(var)`` is the diagonal-MinT / WLS variant that lets
the more uncertain days absorb more of the adjustment.  Because ``S`` is
block-diagonal the projection decouples per gauge-decade and is evaluated in
``O(N)`` with ``numpy.bincount`` -- no dense ``N x N`` weight or ``D x D`` solve.

Probabilistic reconciliation (draw-level)
-----------------------------------------
:func:`reconcile_probabilistic` realises coherent predictive distributions by
applying the *same* linear projection to every draw of the incoherent base
ensemble -- the sample-path reconciliation of Panagiotelis et al. (2023, *EJOR*
"Probabilistic forecast reconciliation: properties, evaluation and score
optimisation") and Rangapuram et al. (2023).  Pushing each joint draw through the
projection lands it on the coherent subspace, so the reconciled daily ensemble
aggregates draw-for-draw to the decadal ensemble; the aggregated-daily and
decadal predictive CRPS therefore coincide and every distributional coherence
discrepancy collapses to ~0, while each daily cell keeps its predictive spread
(under OLS the per-decade adjustment is a single per-draw shift).

All reconciliation is performed in **discharge space** (m3 s-1), the space in
which the dekada-mean constraint is linear and water volume is conserved; the
log-residual predictive samples are mapped through :func:`sbc.schemas.back_transform`
first.  This complements the multi-resolution learning of Gauch et al. (2021,
*HESS* MTS-LSTM) with a forecast-side coherence guarantee.

House style: heavy / optional imports (``scipy``) are deferred into the functions
that use them; everything else is pure NumPy / pandas and Colab-safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .schemas import OBS_COL, SIM_COL, back_transform
from .utils import get_logger
from .validation import metrics as M

log = get_logger(__name__)

GAUGE_COL = "code"
DATE_COL = "date"

__all__ = [
    "ConstraintMap",
    "PointReconResult",
    "ProbReconResult",
    "aggregate_constraint",
    "coherence_error",
    "reconcile_point",
    "reconcile_probabilistic",
]


# --------------------------------------------------------------------------- #
#  Decade representative date (mirrors sbc.synthetic.decadal_aggregate)        #
# --------------------------------------------------------------------------- #
def _decade_date(dates: pd.Series | Sequence) -> pd.Series:
    """Map daily timestamps to their Central-Asian decade representative date.

    Each calendar month is split into three dekads -- days 1-10, 11-20 and
    21-end -- represented by day 5, 15 and 25, exactly matching
    :func:`sbc.synthetic.decadal_aggregate` and :mod:`sbc.multiscale`.

    Parameters
    ----------
    dates : pandas.Series or sequence
        Datetime-like daily timestamps.

    Returns
    -------
    pandas.Series
        Decade representative date per input timestamp (index-aligned).
    """
    dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    d = dt.dt.day
    dec = np.where(d <= 10, 5, np.where(d <= 20, 15, 25))
    return pd.to_datetime(dict(year=dt.dt.year, month=dt.dt.month, day=dec))


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


# --------------------------------------------------------------------------- #
#  The aggregation constraint (the linear map S)                              #
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintMap:
    """The 10-daily -> 1-dekada aggregation map ``S`` and its row/column index.

    The map encodes ``decadal = S @ daily`` for a *temporal hierarchy* keyed on
    ``(gauge, decade)``: every daily cell (column) belongs to exactly one decade
    (row), with averaging weight :attr:`agg_w`.  ``S`` is therefore block-diagonal
    and is represented compactly by the per-column decade index :attr:`group` and
    weights :attr:`agg_w`; :meth:`matrix` materialises the explicit sparse ``S``.

    Attributes
    ----------
    group : numpy.ndarray, shape (n_daily,)
        Decade-row index (``0 .. n_decadal-1``) of each daily column.
    agg_w : numpy.ndarray, shape (n_daily,)
        Aggregation weight ``S[group[i], i]`` of each daily column (mean: ``1/n_d``).
    daily_pos : numpy.ndarray, shape (n_daily,)
        Positional indices, into the *daily* frame passed to
        :func:`aggregate_constraint`, of the columns used (those whose decade has
        a matching decadal row).
    decadal_pos : numpy.ndarray, shape (n_decadal,)
        Positional indices, into the *decadal* frame, of the rows used, ordered to
        match :attr:`group`.
    decadal_keys : list of tuple
        The ``(gauge, decade_date)`` key of each decadal row.
    n_daily, n_decadal : int
        Number of daily columns / decadal rows in the (aligned) hierarchy.
    """

    group: np.ndarray
    agg_w: np.ndarray
    daily_pos: np.ndarray
    decadal_pos: np.ndarray
    decadal_keys: list = field(default_factory=list)
    n_daily: int = 0
    n_decadal: int = 0
    _csr: object = field(default=None, repr=False)

    # -- aggregation -------------------------------------------------------- #
    def aggregate(self, daily_values: np.ndarray) -> np.ndarray:
        """Apply ``S``: aggregate daily column values to decade rows.

        Parameters
        ----------
        daily_values : numpy.ndarray
            Either ``(n_daily,)`` or ``(n_daily, M)`` (e.g. ``M`` ensemble draws),
            aligned to :attr:`group`.

        Returns
        -------
        numpy.ndarray
            ``(n_decadal,)`` or ``(n_decadal, M)`` aggregate ``S @ daily_values``.
        """
        x = np.asarray(daily_values, float)
        if x.ndim == 1:
            return np.bincount(self.group, weights=self.agg_w * x,
                               minlength=self.n_decadal)
        return self.matrix() @ x

    def coherence_residual(self, daily_values: np.ndarray,
                           decadal_values: np.ndarray) -> np.ndarray:
        """Coherence residual ``decadal_values - S @ daily_values``."""
        return np.asarray(decadal_values, float) - self.aggregate(daily_values)

    def take_daily(self, full: np.ndarray) -> np.ndarray:
        """Subset a full-daily-frame array to the columns used by this map."""
        return np.asarray(full, float).ravel()[self.daily_pos]

    def take_decadal(self, full: np.ndarray) -> np.ndarray:
        """Subset a full-decadal-frame array to the rows used, in row order."""
        return np.asarray(full, float).ravel()[self.decadal_pos]

    def matrix(self):
        """Materialise the explicit sparse aggregation matrix ``S`` (CSR)."""
        if self._csr is None:
            from scipy import sparse

            cols = np.arange(self.n_daily)
            self._csr = sparse.csr_matrix(
                (self.agg_w, (self.group, cols)),
                shape=(self.n_decadal, self.n_daily),
            )
        return self._csr


def aggregate_constraint(daily: pd.DataFrame, decadal: pd.DataFrame, *,
                         gauge_col: str = GAUGE_COL, date_col: str = DATE_COL,
                         weights: str = "mean") -> ConstraintMap:
    """Build the 10-daily -> 1-dekada aggregation map ``S`` for a gauge network.

    Aligns a daily modelling table to a decadal one on the ``(gauge, decade)``
    temporal hierarchy and returns the linear map ``S`` such that
    ``decadal_mean = S @ daily`` (see :class:`ConstraintMap`).  Only decades
    present in *both* tables are retained, so the map is immediately usable for
    reconciliation on a temporal holdout where the daily and decadal evaluation
    windows overlap only partially.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily table with ``gauge_col`` and a daily ``date_col``.
    decadal : pandas.DataFrame
        Decadal table for the same gauges; ``date_col`` holds the decade
        representative date (day 5/15/25, as produced by
        :func:`sbc.synthetic.decadal_aggregate`).
    gauge_col, date_col : str
        Identifier columns (default ``"code"`` / ``"date"``).
    weights : {"mean"}, default "mean"
        Aggregation weighting.  Only the dekada **mean** is currently defined
        (each daily cell weighted ``1/n_d``); the argument exists so a future
        volume-weighted aggregation can slot in.

    Returns
    -------
    ConstraintMap
        The aggregation map together with the positional index needed to align
        daily / decadal prediction arrays (:meth:`ConstraintMap.take_daily` /
        :meth:`ConstraintMap.take_decadal`).

    Raises
    ------
    ValueError
        If the two tables share no ``(gauge, decade)`` cell.
    """
    if weights != "mean":
        raise ValueError(f"unsupported aggregation weighting {weights!r}; "
                         "only 'mean' (dekada mean) is defined")

    dly = daily.reset_index(drop=True)
    dec = decadal.reset_index(drop=True)

    dly_key = list(zip(dly[gauge_col].to_numpy(),
                       _decade_date(dly[date_col]).to_numpy()))
    dec_key = list(zip(dec[gauge_col].to_numpy(), pd.to_datetime(dec[date_col]).to_numpy()))

    # decadal rows that actually have daily cells, in stable appearance order
    dec_index = {k: j for j, k in enumerate(dec_key)}
    daily_pos: list[int] = []
    group_keys: list = []
    for i, k in enumerate(dly_key):
        if k in dec_index:
            daily_pos.append(i)
            group_keys.append(k)
    if not daily_pos:
        raise ValueError("aggregate_constraint: daily and decadal tables share no "
                         "(gauge, decade) cell")

    # compact, contiguous decade-row ids over the *used* decades only
    used_keys = sorted(set(group_keys), key=lambda k: (str(k[0]), k[1]))
    key_to_row = {k: r for r, k in enumerate(used_keys)}
    group = np.fromiter((key_to_row[k] for k in group_keys), dtype=np.int64,
                        count=len(group_keys))
    daily_pos_arr = np.asarray(daily_pos, dtype=np.int64)
    decadal_pos = np.asarray([dec_index[k] for k in used_keys], dtype=np.int64)

    n_decadal = len(used_keys)
    counts = np.bincount(group, minlength=n_decadal).astype(float)
    agg_w = 1.0 / counts[group]  # dekada mean weight 1/n_d

    log.info("aggregate_constraint: %d daily cells -> %d dekads "
             "(%d gauges); mean dekada length %.1f days",
             daily_pos_arr.size, n_decadal,
             pd.Series([k[0] for k in used_keys]).nunique(),
             float(counts.mean()))
    return ConstraintMap(group=group, agg_w=agg_w, daily_pos=daily_pos_arr,
                         decadal_pos=decadal_pos, decadal_keys=used_keys,
                         n_daily=daily_pos_arr.size, n_decadal=n_decadal)


# --------------------------------------------------------------------------- #
#  Coherence metrics                                                          #
# --------------------------------------------------------------------------- #
def _coherence_stats(residual: np.ndarray, decadal_ref: np.ndarray) -> dict[str, float]:
    """Scalar summaries of a coherence residual vector (NaN-aware)."""
    r = _as_1d(residual)
    ref = np.abs(_as_1d(decadal_ref))
    finite = np.isfinite(r)
    r = r[finite]
    scale = float(np.mean(ref[np.isfinite(ref)])) if np.isfinite(ref).any() else np.nan
    if r.size == 0:
        return {"max_abs": np.nan, "rms": np.nan, "mean_abs": np.nan, "rel_pct": np.nan}
    mean_abs = float(np.mean(np.abs(r)))
    return {
        "max_abs": float(np.max(np.abs(r))),
        "rms": float(np.sqrt(np.mean(r ** 2))),
        "mean_abs": mean_abs,
        "rel_pct": float(100.0 * mean_abs / scale) if scale and np.isfinite(scale) else np.nan,
    }


def coherence_error(cmap: ConstraintMap, daily_values: np.ndarray,
                    decadal_values: np.ndarray) -> dict[str, float]:
    """Aggregation-coherence error of a daily/decadal forecast pair.

    Parameters
    ----------
    cmap : ConstraintMap
        The aggregation map from :func:`aggregate_constraint`.
    daily_values, decadal_values : numpy.ndarray
        Daily (``n_daily``) and decadal (``n_decadal``) forecasts, already aligned
        to ``cmap`` (use :meth:`ConstraintMap.take_daily` / ``take_decadal``).

    Returns
    -------
    dict
        ``max_abs`` / ``rms`` / ``mean_abs`` coherence residual [m3 s-1] and
        ``rel_pct`` (mean absolute residual as a percentage of the mean decadal
        magnitude).
    """
    resid = cmap.coherence_residual(daily_values, decadal_values)
    return _coherence_stats(resid, decadal_values)


def _energy_distance_rows(x: np.ndarray, y: np.ndarray,
                          max_draws: int = 120, seed: int = 0) -> float:
    """Mean per-row univariate energy distance between two ensembles.

    ``x`` / ``y`` are ``(n_rows, M)`` empirical samples; the energy distance per
    row is ``2 E|X-Y| - E|X-X'| - E|Y-Y'|`` (0 iff the two predictive
    distributions coincide).  Draws are subsampled to ``max_draws`` for the
    ``O(M^2)`` term.
    """
    x = np.atleast_2d(np.asarray(x, float))
    y = np.atleast_2d(np.asarray(y, float))
    rng = np.random.default_rng(seed)
    if x.shape[1] > max_draws:
        x = x[:, rng.choice(x.shape[1], max_draws, replace=False)]
    if y.shape[1] > max_draws:
        y = y[:, rng.choice(y.shape[1], max_draws, replace=False)]

    def _mean_abs(a, b):
        return np.abs(a[:, :, None] - b[:, None, :]).mean(axis=(1, 2))

    ed = 2.0 * _mean_abs(x, y) - _mean_abs(x, x) - _mean_abs(y, y)
    ed = ed[np.isfinite(ed)]
    return float(np.mean(ed)) if ed.size else np.nan


# --------------------------------------------------------------------------- #
#  Core projection (equality-constrained MinT / OLS)                          #
# --------------------------------------------------------------------------- #
def _trust_weights(method: str, n_daily: int,
                   weights: np.ndarray | None) -> np.ndarray:
    """Resolve the per-daily-cell trust weights ``W_ii`` of the MinT projection."""
    if weights is not None:
        w = _as_1d(weights)
        if w.shape[0] != n_daily:
            raise ValueError("explicit weights length != number of daily cells")
        return np.clip(w, 1e-12, None)
    if method in ("ols", None):
        return np.ones(n_daily, float)
    raise ValueError(f"method {method!r} needs per-cell weights "
                     "(pass weights=... for 'wls'/'mint')")


def _top_variance(anchor: str, n_decadal: int,
                  top_weights: np.ndarray | None) -> np.ndarray:
    """Resolve the per-decadal-row top-level variance ``sigma2_top``.

    ``anchor="decadal"`` returns zeros (the decadal bulletin is authoritative and
    held fixed -> exact aggregation); ``anchor="mint"`` returns ``top_weights``
    (unit if omitted) so both hierarchy levels are blended.
    """
    if anchor == "decadal":
        return np.zeros(n_decadal, float)
    if anchor == "mint":
        if top_weights is None:
            return np.ones(n_decadal, float)
        w = _as_1d(top_weights)
        if w.shape[0] != n_decadal:
            raise ValueError("top_weights length != number of decadal rows")
        return np.clip(w, 0.0, None)
    raise ValueError(f"unknown anchor {anchor!r}; use 'decadal' or 'mint'")


def _project(daily_pred: np.ndarray, decadal_pred: np.ndarray,
             cmap: ConstraintMap, trust_w: np.ndarray,
             sigma2_top: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MinT projection of a daily/decadal forecast pair onto the coherent subspace.

    Solves, per gauge-decade, the generalised-least-squares blend of the two base
    forecasts that both levels of the temporal hierarchy
    ``y_full = [decadal; daily]`` agree on, exploiting the block-diagonal ``S``::

        lambda_d = (c_d - (S y)_d) / (sigma2_top_d + sum_i a_i^2 W_ii)
        daily~_i = y_i + W_ii a_i lambda_d
        decadal~_d = c_d - sigma2_top_d * lambda_d      ( == (S daily~)_d )

    The reconciled daily and decadal forecasts are therefore mutually coherent by
    construction.  ``sigma2_top -> 0`` recovers the *equality-constrained* (hard
    top-anchor) MinT used by :func:`reconcile_point`: every day is moved so the
    decade mean hits ``c_d`` exactly (``decadal~ == c``).  A finite ``sigma2_top``
    (the decadal predictive variance) lets both levels move -- a variance
    contraction that retains probabilistic skill (Wickramasuriya 2019;
    Panagiotelis 2023).

    Accepts 1-D ``(n_daily,)`` or 2-D ``(n_daily, M)`` forecasts (with matching
    ``(n_decadal[, M])`` ``decadal_pred``); ``trust_w`` (per daily cell) and
    ``sigma2_top`` (per decadal row) are shared across draws.

    Returns ``(reconciled_daily, reconciled_decadal, residual)`` where
    ``residual = c - S y`` is the *pre*-reconciliation coherence residual.
    """
    group = cmap.group
    agg_w = cmap.agg_w
    y = np.asarray(daily_pred, float)
    c = np.asarray(decadal_pred, float)

    # (S W S')_dd  -- diagonal because S is block-diagonal in the decades
    denom = np.bincount(group, weights=agg_w * agg_w * trust_w, minlength=cmap.n_decadal)
    denom = denom + np.asarray(sigma2_top, float)              # + top-level variance
    inv_denom = np.where(denom > 0, 1.0 / denom, 0.0)

    sy = cmap.aggregate(y)            # S y
    resid = c - sy                    # c - S y
    if y.ndim == 1:
        lam = resid * inv_denom                       # (D,)
        adj = trust_w * agg_w * lam[group]            # (N,)
        recon_daily = y + adj
        recon_decadal = c - np.asarray(sigma2_top, float) * lam
    else:
        lam = resid * inv_denom[:, None]              # (D, M)
        adj = (trust_w * agg_w)[:, None] * lam[group, :]
        recon_daily = y + adj
        recon_decadal = c - np.asarray(sigma2_top, float)[:, None] * lam
    return recon_daily, recon_decadal, resid


# --------------------------------------------------------------------------- #
#  Point reconciliation                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class PointReconResult:
    """Result of :func:`reconcile_point`.

    Attributes
    ----------
    reconciled : numpy.ndarray, shape (n_daily,)
        Reconciled daily forecast.  Under the default hard top-anchor it
        aggregates *exactly* to ``decadal_pred``.
    reconciled_decadal : numpy.ndarray, shape (n_decadal,)
        ``S @ reconciled`` -- equal to ``decadal_pred`` under the hard anchor, or
        the optimally-blended decadal forecast under ``anchor="mint"``.
    residual : numpy.ndarray, shape (n_decadal,)
        Pre-reconciliation coherence residual ``decadal_pred - S @ daily_pred``.
    coherence_before, coherence_after : dict
        :func:`_coherence_stats` summaries before / after reconciliation
        (``after`` measures ``reconciled_decadal - S @ reconciled``, ~0).
    """

    reconciled: np.ndarray
    reconciled_decadal: np.ndarray
    residual: np.ndarray
    coherence_before: dict
    coherence_after: dict


def reconcile_point(daily_pred: np.ndarray, decadal_pred: np.ndarray,
                    S: ConstraintMap, *, method: str = "ols",
                    weights: np.ndarray | None = None, anchor: str = "decadal",
                    top_weights: np.ndarray | None = None,
                    non_negative: bool = False) -> PointReconResult:
    """MinT / OLS optimal point reconciliation of a daily/decadal forecast pair.

    Adjusts the daily-corrected discharge so it aggregates *exactly* to the
    decadal-corrected bulletin, by the trace-minimising MinT projection of
    Wickramasuriya et al. (2019); see the module docstring for the closed form.
    With ``method="ols"`` (``W = I``) the per-decade gap is shared equally across
    the days; ``method="wls"`` with ``weights`` lets uncertain days absorb more.

    Parameters
    ----------
    daily_pred : numpy.ndarray, shape (n_daily,)
        Daily-corrected discharge [m3 s-1], aligned to ``S`` (use
        :meth:`ConstraintMap.take_daily`).
    decadal_pred : numpy.ndarray, shape (n_decadal,)
        Decadal-corrected discharge [m3 s-1] -- the authoritative aggregate the
        daily series is reconciled to (use ``S.take_decadal``).
    S : ConstraintMap
        Aggregation map from :func:`aggregate_constraint`.
    method : {"ols", "wls", "mint"}, default "ols"
        ``"ols"`` uses unit trust weights; ``"wls"``/``"mint"`` require per-cell
        ``weights`` (e.g. predictive variances) for the diagonal-MinT projection.
    weights : numpy.ndarray, optional
        Per-daily-cell trust weights ``W_ii`` (required for non-OLS).
    anchor : {"decadal", "mint"}, default "decadal"
        ``"decadal"`` treats the decadal bulletin as authoritative and forces the
        daily series to aggregate to it *exactly* (the task default).  ``"mint"``
        relaxes the top with ``top_weights`` so both levels are optimally blended.
    top_weights : numpy.ndarray, optional
        Per-decadal-row variances for ``anchor="mint"`` (unit if omitted).
    non_negative : bool, default False
        Clip the reconciled discharge at 0.  Off by default because clipping can
        re-introduce a small coherence error; daily corrections are positive and
        the per-decade shift is tiny, so this rarely triggers.

    Returns
    -------
    PointReconResult
        Reconciled daily / decadal series plus coherence summaries.
    """
    if not isinstance(S, ConstraintMap):
        raise TypeError("S must be a ConstraintMap from aggregate_constraint()")
    y = _as_1d(daily_pred)
    c = _as_1d(decadal_pred)
    if y.shape[0] != S.n_daily or c.shape[0] != S.n_decadal:
        raise ValueError(f"shape mismatch: daily_pred {y.shape} (expected "
                         f"{S.n_daily}), decadal_pred {c.shape} (expected {S.n_decadal})")

    trust_w = _trust_weights(method, S.n_daily, weights)
    sigma2_top = _top_variance(anchor, S.n_decadal, top_weights)
    recon, recon_dec, resid = _project(y, c, S, trust_w, sigma2_top)
    if non_negative:
        recon = np.clip(recon, 0.0, None)
        recon_dec = S.aggregate(recon)

    before = _coherence_stats(resid, c)
    after = _coherence_stats(recon_dec - S.aggregate(recon), c)
    log.info("reconcile_point[%s/%s]: coherence rel %.4f%% -> %.2e%% "
             "(max abs %.3g -> %.3g m3/s)", method, anchor,
             before["rel_pct"], after["rel_pct"], before["max_abs"], after["max_abs"])
    return PointReconResult(reconciled=recon, reconciled_decadal=recon_dec,
                            residual=resid, coherence_before=before, coherence_after=after)


# --------------------------------------------------------------------------- #
#  Probabilistic reconciliation                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ProbReconResult:
    """Result of :func:`reconcile_probabilistic`.

    Attributes
    ----------
    reconciled_daily : numpy.ndarray, shape (n_daily, M)
        Coherent daily ensemble: every draw aggregates to the reconciled decadal
        draw, so the aggregated-daily and decadal predictive distributions
        coincide.
    reconciled_decadal : numpy.ndarray, shape (n_decadal, M)
        ``S @ reconciled_daily`` -- the coherent decadal ensemble.  Equal to the
        input ``decadal_samples`` under ``anchor="decadal"``; the optimal blend of
        both base ensembles under the default ``anchor="mint"``.
    coherence_before, coherence_after : dict
        Distributional coherence between the aggregated-daily and the
        (reconciled) decadal predictive ensembles, before / after
        (``energy_distance`` and mean-residual stats); ``after`` collapses to ~0.
    crps : dict or None
        Present when observations are supplied: aggregated-daily and decadal CRPS
        before / after and the daily-level CRPS retention.
    """

    reconciled_daily: np.ndarray
    reconciled_decadal: np.ndarray
    coherence_before: dict
    coherence_after: dict
    crps: dict | None = None


def reconcile_probabilistic(daily_samples: np.ndarray, decadal_samples: np.ndarray,
                            S: ConstraintMap, *, method: str = "wls",
                            weights: np.ndarray | None = None,
                            anchor: str = "mint",
                            top_weights: np.ndarray | None = None,
                            decadal_obs: np.ndarray | None = None,
                            daily_obs: np.ndarray | None = None,
                            non_negative: bool = False,
                            seed: int = 0) -> ProbReconResult:
    """Draw-level cross-temporal reconciliation of predictive ensembles.

    Applies the MinT projection of :func:`_project` to *every draw* of the base
    ensembles -- the sample-path probabilistic reconciliation of Panagiotelis et
    al. (2023) and Rangapuram et al. (2023).  Each reconciled draw is coherent
    (its aggregated-daily and decadal parts agree), so the aggregated-daily and
    decadal predictive distributions become identical and every distributional
    coherence discrepancy vanishes.

    Because the two marginal base ensembles are typically drawn *independently*,
    fixing the decadal draws (``anchor="decadal"``) would inject their variance
    into the daily draws.  The default ``anchor="mint"`` instead blends both
    levels with the **full** MinT projection -- a variance contraction onto the
    coherent subspace (Wickramasuriya 2019; Panagiotelis 2023) -- which preserves
    (typically improves) the predictive CRPS at both scales.  The per-cell trust
    weights default to the ensembles' own sample variances (``method="wls"``).

    Parameters
    ----------
    daily_samples : numpy.ndarray, shape (n_daily, M)
        Base daily discharge ensemble [m3 s-1], aligned to ``S``.
    decadal_samples : numpy.ndarray, shape (n_decadal, M)
        Base decadal discharge ensemble (same ``M`` draws).
    S : ConstraintMap
        Aggregation map from :func:`aggregate_constraint`.
    method : {"wls", "ols", "mint"}, default "wls"
        Daily trust weighting.  ``"wls"`` (default) uses each daily cell's sample
        variance; ``"ols"`` uses unit weights; explicit ``weights`` override both.
    weights : numpy.ndarray, optional
        Per-daily-cell trust weights ``W_ii`` (overrides ``method``).
    anchor : {"mint", "decadal"}, default "mint"
        ``"mint"`` blends both levels (decadal variance from ``top_weights`` or the
        decadal sample variance); ``"decadal"`` holds the decadal draws fixed so
        the daily draws aggregate to them exactly.
    top_weights : numpy.ndarray, optional
        Per-decadal-row variances for ``anchor="mint"`` (decadal sample variance
        if omitted).
    decadal_obs : numpy.ndarray, shape (n_decadal,), optional
        Decadal observations; enables the aggregated-daily / decadal CRPS report.
    daily_obs : numpy.ndarray, shape (n_daily,), optional
        Daily observations; enables the daily-level CRPS retention report.
    non_negative : bool, default False
        Clip reconciled discharge draws at 0 (see :func:`reconcile_point`).
    seed : int, default 0
        Seed for the energy-distance draw subsampling.

    Returns
    -------
    ProbReconResult
        Coherent daily / decadal ensembles, coherence summaries and (when
        observations are given) the CRPS report.
    """
    if not isinstance(S, ConstraintMap):
        raise TypeError("S must be a ConstraintMap from aggregate_constraint()")
    dly = np.atleast_2d(np.asarray(daily_samples, float))
    dec = np.atleast_2d(np.asarray(decadal_samples, float))
    if dly.shape[0] != S.n_daily or dec.shape[0] != S.n_decadal:
        raise ValueError(f"shape mismatch: daily_samples {dly.shape} (rows "
                         f"{S.n_daily}), decadal_samples {dec.shape} (rows {S.n_decadal})")
    if dly.shape[1] != dec.shape[1]:
        raise ValueError("daily_samples and decadal_samples must share the draw "
                         f"axis; got {dly.shape[1]} vs {dec.shape[1]}")

    if weights is None and method in ("wls", "mint"):
        trust_w = np.clip(dly.var(axis=1), 1e-12, None)   # MinT(Sample): cell variance
    else:
        trust_w = _trust_weights(method, S.n_daily, weights)
    if anchor == "mint" and top_weights is None:
        sigma2_top = np.clip(dec.var(axis=1), 1e-12, None)
    else:
        sigma2_top = _top_variance(anchor, S.n_decadal, top_weights)

    recon_daily, recon_decadal, _ = _project(dly, dec, S, trust_w, sigma2_top)
    if non_negative:
        recon_daily = np.clip(recon_daily, 0.0, None)
        recon_decadal = S.aggregate(recon_daily)

    # distributional coherence: aggregated-daily ensemble vs the decadal ensemble
    agg_before = S.aggregate(dly)
    agg_after = S.aggregate(recon_daily)
    before = _coherence_stats((dec - agg_before).mean(axis=1), dec.mean(axis=1))
    before["energy_distance"] = _energy_distance_rows(agg_before, dec, seed=seed)
    after = _coherence_stats((recon_decadal - agg_after).mean(axis=1),
                             recon_decadal.mean(axis=1))
    after["energy_distance"] = _energy_distance_rows(agg_after, recon_decadal, seed=seed)

    crps = None
    if decadal_obs is not None:
        c_obs = _as_1d(decadal_obs)
        crps = {
            "decadal_base": M.crps_ensemble(c_obs, dec),
            "decadal_reconciled": M.crps_ensemble(c_obs, recon_decadal),
            "agg_daily_before": M.crps_ensemble(c_obs, agg_before),
            "agg_daily_after": M.crps_ensemble(c_obs, agg_after),
        }
        if daily_obs is not None:
            d_obs = _as_1d(daily_obs)
            crps["daily_before"] = M.crps_ensemble(d_obs, dly)
            crps["daily_after"] = M.crps_ensemble(d_obs, recon_daily)

    log.info("reconcile_probabilistic[%s/%s]: energy-distance(agg-daily, decadal) "
             "%.3g -> %.3g", method, anchor,
             before["energy_distance"], after["energy_distance"])
    return ProbReconResult(reconciled_daily=recon_daily, reconciled_decadal=recon_decadal,
                           coherence_before=before, coherence_after=after, crps=crps)


# --------------------------------------------------------------------------- #
#  Self-test / runnable demo (synthetic, small, < 3 min)                      #
# --------------------------------------------------------------------------- #
def _median_kge(frame: pd.DataFrame, pred_col: str) -> float:
    """Across-gauge median KGE' of (obs, pred_col) -- the house reduction."""
    pg = M.evaluate_by_group(frame, OBS_COL, pred_col, group=GAUGE_COL, date_col=DATE_COL)
    return float(pg["kge"].median(skipna=True))


def _gauss_discharge_ensemble(model, train: pd.DataFrame, test: pd.DataFrame,
                              n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Plug-in additive-Gaussian predictive ensemble around a point corrector.

    Reconciliation is *model-agnostic*: it post-processes any forecast.  For the
    demo a deterministic corrector is wrapped in a per-gauge additive Gaussian
    predictive law **in discharge space**, whose spread is the corrector's
    in-sample residual standard deviation per gauge.  (A native probabilistic
    model such as ``RegimeProbNet`` plugs in the same way via its ``sample``
    method; the additive discharge-space form is used here only so the synthetic
    ensemble has well-behaved, non-explosive per-row variance.)

    Returns ``(point_discharge, discharge_samples[n_rows, n])``.
    """
    pt_tr = np.asarray(model.predict(train), float)
    pt_te = np.asarray(model.predict(test), float)
    err = np.asarray(train[OBS_COL], float) - pt_tr
    by_gauge = (pd.DataFrame({"g": train[GAUGE_COL].to_numpy(), "e": err})
                .groupby("g")["e"].std())
    glob = float(np.clip(np.nanstd(err), 1e-6, None))
    sig = np.clip(test[GAUGE_COL].map(by_gauge).to_numpy(float), 1e-6, None)
    sig = np.where(np.isfinite(sig), sig, glob)
    rng = np.random.default_rng(seed)
    draws = pt_te[:, None] + sig[:, None] * rng.standard_normal((pt_te.shape[0], n))
    return pt_te, np.clip(draws, 0.0, None)


def _demo() -> None:  # pragma: no cover - manual smoke test
    from .features.engineering import build_features
    from .features.regimes import classify_regimes
    from .models.boosting import LightGBMCorrector
    from .schemas import TARGET_COL, make_target, validate
    from .synthetic import decadal_aggregate, generate
    from .validation.splits import temporal_split

    pd.set_option("display.width", 200)

    # --- BOTH scales describe the SAME gauges / truth (shared seed) -------- #
    raw_daily = generate(scale="daily", years=6, n_basins=2,
                         gauges_per_basin=(1, 2), seed=0)
    raw_dec = decadal_aggregate(raw_daily)
    daily = classify_regimes(build_features(raw_daily, scale="daily"))
    decadal = classify_regimes(build_features(raw_dec, scale="decadal"))
    for d in (daily, decadal):
        d[TARGET_COL] = make_target(d[OBS_COL].to_numpy(), d[SIM_COL].to_numpy())
    daily, decadal = validate(daily), validate(decadal)
    print(f"[reconciliation] daily  : {len(daily):6d} rows, {daily['code'].nunique()} gauges")
    print(f"[reconciliation] decadal: {len(decadal):6d} rows, {decadal['code'].nunique()} gauges")

    # --- leakage-safe temporal holdout at each scale ----------------------- #
    dtr, dte = temporal_split(daily, test_frac=0.3)
    ctr, cte = temporal_split(decadal, test_frac=0.3)
    daily_tr, daily_te = daily[dtr].copy(), daily[dte].copy()
    dec_tr, dec_te = decadal[ctr].copy(), decadal[cte].copy()

    # --- well-behaved point correctors at both scales (house default) ------ #
    # reconciliation is model-agnostic; the untuned LightGBM corrector of
    # sbc.multiscale is used here for a fast, deterministic smoke run.
    m_daily = LightGBMCorrector(n_optuna_trials=0, seed=0).fit(daily_tr)
    m_dec = LightGBMCorrector(n_optuna_trials=0, seed=0).fit(dec_tr)

    # point + Gaussian predictive ensembles on the posterior test windows --- #
    M_DRAWS = 100
    daily_pt, daily_s = _gauss_discharge_ensemble(m_daily, daily_tr, daily_te, M_DRAWS, seed=1)
    dec_pt, dec_s = _gauss_discharge_ensemble(m_dec, dec_tr, dec_te, M_DRAWS, seed=1)

    # --- the aggregation constraint S (10-daily -> 1-dekada) --------------- #
    cmap = aggregate_constraint(daily_te, dec_te)
    daily_pt_a = cmap.take_daily(daily_pt)
    dec_pt_a = cmap.take_decadal(dec_pt)
    daily_s_a = daily_s[cmap.daily_pos]
    dec_s_a = dec_s[cmap.decadal_pos]
    daily_obs_a = cmap.take_daily(daily_te[OBS_COL].to_numpy())
    dec_obs_a = cmap.take_decadal(dec_te[OBS_COL].to_numpy())
    print(f"[reconciliation] aligned hierarchy: {cmap.n_daily} daily cells "
          f"-> {cmap.n_decadal} dekads")

    # === POINT reconciliation (MinT/OLS) ================================== #
    pr = reconcile_point(daily_pt_a, dec_pt_a, cmap)
    print("\n=== Point reconciliation (MinT/OLS) ===")
    print(f"  coherence rel error  before={pr.coherence_before['rel_pct']:.4f}%  "
          f"after={pr.coherence_after['rel_pct']:.2e}%")
    print(f"  coherence max |resid| before={pr.coherence_before['max_abs']:.4g}  "
          f"after={pr.coherence_after['max_abs']:.2e} m3/s")

    # daily KGE' retention (raw vs corrected vs reconciled) ----------------- #
    fr = pd.DataFrame({GAUGE_COL: daily_te[GAUGE_COL].to_numpy()[cmap.daily_pos],
                       DATE_COL: daily_te[DATE_COL].to_numpy()[cmap.daily_pos],
                       OBS_COL: daily_obs_a, SIM_COL: cmap.take_daily(daily_te[SIM_COL].to_numpy()),
                       "q_corr": daily_pt_a, "q_recon": pr.reconciled})
    kge_raw = _median_kge(fr, SIM_COL)
    kge_cor = _median_kge(fr, "q_corr")
    kge_rec = _median_kge(fr, "q_recon")
    print(f"  daily   KGE' (median gauge): raw={kge_raw:+.3f}  corrected={kge_cor:+.3f}  "
          f"reconciled={kge_rec:+.3f}  (retention={kge_rec / kge_cor if kge_cor else float('nan'):.3f})")

    # decadal-scale skill: the reconciled daily, re-aggregated, now matches the
    # (more accurate) decadal bulletin instead of the incoherent raw aggregate.
    agg_before = cmap.aggregate(daily_pt_a)
    agg_after = cmap.aggregate(pr.reconciled)
    kge_dec_before = M.kge_prime(dec_obs_a, agg_before)["kge"]
    kge_dec_after = M.kge_prime(dec_obs_a, agg_after)["kge"]
    print(f"  decadal KGE' (aggregated daily): before={kge_dec_before:+.3f}  "
          f"after={kge_dec_after:+.3f}  (decadal bulletin={M.kge_prime(dec_obs_a, dec_pt_a)['kge']:+.3f})")

    # === PROBABILISTIC reconciliation (draw-level, full MinT) ============= #
    pp = reconcile_probabilistic(daily_s_a, dec_s_a, cmap,
                                 decadal_obs=dec_obs_a, daily_obs=daily_obs_a, seed=1)
    print("\n=== Probabilistic reconciliation (draw-level; Panagiotelis/Rangapuram) ===")
    print(f"  distributional coherence (energy dist agg-daily vs decadal): "
          f"before={pp.coherence_before['energy_distance']:.4g}  "
          f"after={pp.coherence_after['energy_distance']:.2e}")
    print(f"  decadal CRPS         base={pp.crps['decadal_base']:.4g}  "
          f"reconciled={pp.crps['decadal_reconciled']:.4g}")
    print(f"  aggregated-daily CRPS  before={pp.crps['agg_daily_before']:.4g}  "
          f"after={pp.crps['agg_daily_after']:.4g}  (== reconciled decadal: coherent)")
    print(f"  daily   CRPS           before={pp.crps['daily_before']:.4g}  "
          f"after={pp.crps['daily_after']:.4g}  (retained/improved by contraction)")

    # --- assertions: incoherence ~ 0 while skill is retained --------------- #
    assert pr.coherence_before["max_abs"] > 0.0, "no incoherence to reconcile?"
    assert pr.coherence_after["max_abs"] < 1e-6 * max(1.0, pr.coherence_before["max_abs"]) \
        or pr.coherence_after["max_abs"] < 1e-6, "point reconciliation left residual coherence error"
    assert np.allclose(cmap.aggregate(pr.reconciled), dec_pt_a, atol=1e-6), \
        "reconciled daily does not aggregate to the decadal anchor"
    # probabilistic: the reconciled daily ensemble aggregates draw-for-draw to the
    # reconciled decadal ensemble (coherent), so their CRPS coincide exactly.
    assert pp.coherence_after["energy_distance"] < 1e-6, \
        "probabilistic reconciliation left distributional incoherence"
    assert np.allclose(pp.reconciled_decadal, cmap.aggregate(pp.reconciled_daily), atol=1e-6), \
        "reconciled daily draws do not aggregate to the reconciled decadal draws"
    assert abs(pp.crps["agg_daily_after"] - pp.crps["decadal_reconciled"]) < 1e-6, \
        "aggregated-daily CRPS must equal the reconciled decadal CRPS (coherent)"
    # skill retained: reconciled daily still beats raw GloFAS at the daily scale,
    # the re-aggregated daily lifts (>=) decadal-scale skill, and the full-MinT
    # contraction does not degrade the daily predictive CRPS.
    assert np.isfinite(kge_rec), "reconciled daily KGE' is not finite"
    assert kge_rec > kge_raw + 1e-9, "reconciliation destroyed daily skill vs raw GloFAS"
    assert kge_dec_after >= kge_dec_before - 1e-9, \
        "reconciliation worsened decadal-scale agreement"
    assert pp.crps["daily_after"] <= 1.05 * pp.crps["daily_before"], \
        "full-MinT reconciliation materially degraded the daily predictive CRPS"

    print("\nOK: cross-temporal reconciliation drove incoherence -> 0 (point & "
          "probabilistic) while retaining daily KGE'/CRPS and lifting decadal-scale skill.")


if __name__ == "__main__":  # pragma: no cover
    _demo()
