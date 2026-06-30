"""Decision-relevant uncertainty-quantification (UQ) scores for the flagship.

:mod:`sbc.validation.calibration` answers *"are the predictive distributions
reliable?"* (PIT, reliability curve, coverage/sharpness).  This module completes
the probabilistic-evaluation suite that reviewers in the streamflow-postprocessing
niche now expect, following the two reference studies in the repository:

* **El Ouahabi et al. (2026, HESS 30:3549)** -- the multi-site QRF uncertainty
  study whose headline distributional + interval metrics are the *Alpha
  reliability index* (Renard et al., 2010), the *dispersion* (sharpness) score,
  the *coverage ratio* / *average width*, and the **Winkler score**, each
  reported as a skill score relative to the climatological distribution.
* **Zhang et al. (2023, HESS 27:4529)** -- the GloFAS QRF post-processor whose
  probabilistic suite pairs CRPS with the **threshold-weighted CRPS** of
  Gneiting & Ranjan (2011) to emphasise high-flow performance, plus
  coverage/sharpness of the 50 % and 90 % prediction intervals.

Implemented here
----------------
* :func:`winkler_interval_score` -- the Winkler / interval score of a central
  ``(1 - alpha)`` prediction interval ``= width + (2/alpha) x undercoverage``
  (Gneiting & Raftery, 2007; Winkler, 1972), with :func:`winkler_skill_score`
  giving ``1 - WS_model / WS_ref`` versus a reference forecast (WSS).
* :func:`alpha_reliability` -- the El Ouahabi / Renard **Alpha index**
  ``1 - 2 * mean|observed_freq - nominal|`` aggregating reliability across all
  predicted quantile levels (``1`` perfect, ``0`` worst).
* :func:`dispersion_ratio` -- predictive sharpness (dispersion) relative to the
  observed climatological spread; the core of El Ouahabi's dispersion score.
* :func:`pi_coverage_width` -- prediction-interval coverage probability (PICP)
  and mean prediction-interval width (MPIW).
* :func:`twcrps` -- the threshold-weighted CRPS for high flows (Gneiting &
  Ranjan, 2011), via the chaining function ``v(z) = max(z, t)`` applied to the
  exact ensemble CRPS.
* :func:`uq_score_summary` -- a compact scalar bundle for the paper's tables.

Design
------
Pure, NaN-aware NumPy/pandas; the deterministic CRPS estimator
(:func:`sbc.validation.metrics.crps_ensemble`) and the reliability /
coverage / sharpness primitives (:mod:`sbc.validation.calibration`) are reused
rather than re-derived.  Every score operates on whatever quantity the analysis
evaluates (the log-residual target, or back-transformed discharge -- the
threshold-weighted CRPS for *high flows* is most naturally read in discharge
space).  SciPy is not required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger
from .calibration import coverage as _coverage
from .calibration import reliability_curve, sharpness
from .metrics import crps_ensemble

log = get_logger(__name__)

__all__ = [
    "winkler_interval_score",
    "winkler_skill_score",
    "alpha_reliability",
    "dispersion_ratio",
    "pi_coverage_width",
    "twcrps",
    "twcrps_skill_score",
    "uq_score_summary",
]


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
def _row_quantile(quant: np.ndarray, levels: np.ndarray, target_p: float) -> np.ndarray:
    """Per-row predictive quantile at ``target_p`` from a monotone grid."""
    order = np.argsort(levels)
    p = np.asarray(levels, float)[order]
    quant = np.atleast_2d(np.asarray(quant, float))[:, order]
    out = np.empty(quant.shape[0], float)
    for i in range(quant.shape[0]):
        qi = np.maximum.accumulate(quant[i])          # enforce non-decreasing
        out[i] = float(np.interp(target_p, p, qi))
    return out


# --------------------------------------------------------------------------- #
#  Winkler / interval score                                                  #
# --------------------------------------------------------------------------- #
def winkler_interval_score(obs, lower, upper, alpha: float, *, reduce: str = "mean"):
    """Winkler (interval) score of a central ``(1 - alpha)`` prediction interval.

    For a nominal ``(1 - alpha)`` interval ``[l, u]`` and observation ``y`` the
    interval score (Gneiting & Raftery, 2007, eq. 43; Winkler, 1972) is

    ``S = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}`` ,

    i.e. the interval width plus a ``2/alpha`` penalty for each unit the
    observation falls *outside* the band.  It is **negatively oriented** (lower
    is better) and is the proper score that operationalises Gneiting's
    "maximise sharpness subject to reliability": a forecaster cannot win by
    shrinking the width without paying for the coverage it loses.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations (residual or discharge -- consistent with the bounds).
    lower, upper : array_like, shape (n,)
        Lower / upper bounds of the central ``(1 - alpha)`` interval, e.g. the
        ``alpha/2`` and ``1 - alpha/2`` predictive quantiles.
    alpha : float
        Interval miscoverage in ``(0, 1)`` (``0.1`` for a 90 % interval).
    reduce : {"mean", "none"}, default "mean"
        ``"mean"`` returns the mean score over finite rows; ``"none"`` returns
        the per-observation score (``NaN`` on non-finite rows).

    Returns
    -------
    float or numpy.ndarray
        Mean Winkler score, or the per-observation array.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    obs = np.asarray(obs, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    width = hi - lo
    below = obs < lo
    above = obs > hi
    score = width + (2.0 / alpha) * (lo - obs) * below + (2.0 / alpha) * (obs - hi) * above
    m = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if reduce == "none":
        return np.where(m, score, np.nan)
    if reduce != "mean":
        raise ValueError("reduce must be 'mean' or 'none'")
    return float(np.mean(score[m])) if m.any() else float("nan")


def winkler_skill_score(obs, lower, upper, ref_lower, ref_upper, alpha: float) -> float:
    """Winkler skill score ``1 - WS_model / WS_ref`` versus a reference interval.

    Both interval pairs are scored at the same ``alpha`` (El Ouahabi et al.,
    2026, report ``WSS`` relative to the climatological distribution).  Positive
    means the model's intervals beat the reference; ``0`` ties it; ``NaN`` if
    the reference score is non-positive or non-finite.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    lower, upper : array_like, shape (n,)
        Model interval bounds.
    ref_lower, ref_upper : array_like, shape (n,)
        Reference (e.g. climatological / raw-GloFAS) interval bounds.
    alpha : float
        Shared interval miscoverage in ``(0, 1)``.

    Returns
    -------
    float
        ``1 - WS_model / WS_ref`` (skill in ``(-inf, 1]``).
    """
    ws_m = winkler_interval_score(obs, lower, upper, alpha)
    ws_r = winkler_interval_score(obs, ref_lower, ref_upper, alpha)
    if not np.isfinite(ws_r) or ws_r <= 0.0:
        return float("nan")
    return float(1.0 - ws_m / ws_r)


# --------------------------------------------------------------------------- #
#  Alpha reliability index (El Ouahabi 2026 / Renard 2010)                    #
# --------------------------------------------------------------------------- #
def alpha_reliability(obs, quantile_preds, levels) -> float:
    """El Ouahabi / Renard Alpha reliability index over the quantile grid.

    The Alpha index (Renard et al., 2010; El Ouahabi et al., 2026) measures how
    close the predicted predictive distributions are to being *reliable*: if the
    forecasts are calibrated, the observation's non-exceedance frequency at each
    nominal level ``p`` equals ``p`` (a uniform PIT).  Aggregating the deviation
    across all levels,

    ``Alpha = 1 - 2 * mean_j |observed_freq_j - p_j|`` ,

    where ``observed_freq_j = P(y <= q_{p_j})`` is read from the quantile-
    reliability curve.  It scores ``1`` for perfect reliability and ``0`` for the
    worst case; it is the level-aggregated companion of the coverage ratio,
    which reports reliability at a single confidence level.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles, column ``j`` at cumulative probability ``levels[j]``.
    levels : array_like, shape (m,)
        Nominal cumulative probabilities of the quantile columns.

    Returns
    -------
    float
        Alpha index in ``[0, 1]`` (clipped at ``0`` to match the paper's range).
    """
    curve = reliability_curve(obs, quantile_preds, levels)
    if curve.empty:
        return float("nan")
    mad = float((curve["empirical"] - curve["nominal"]).abs().mean())
    return float(max(0.0, 1.0 - 2.0 * mad))


# --------------------------------------------------------------------------- #
#  Dispersion (sharpness vs observed spread)                                  #
# --------------------------------------------------------------------------- #
def dispersion_ratio(obs, quantile_preds, levels) -> float:
    """Predictive dispersion (sharpness) relative to the observed spread.

    The dispersion score of El Ouahabi et al. (2026) (after Bontron, 2004)
    measures the *magnitude* of the predictive distributions -- their spread
    about their own median -- relative to the climatological spread.  Here the
    ratio is

    ``D = mean_i  E_F|X_i - median(F_i)|  /  mean|y - median(y)|`` ,

    the average predictive mean-absolute-deviation-from-the-median (integrated
    over the probability grid, the spread of each forecast) divided by the
    observations' climatological mean absolute deviation.  ``D < 1`` means the
    forecasts are sharper than the raw observed variability and ``D ~ 1`` means
    they match it; sharpness is only desirable *subject to* reliability (read it
    next to :func:`alpha_reliability` and :func:`pi_coverage_width`).

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles aligned to ``levels``.
    levels : array_like, shape (m,)
        Cumulative probabilities of the quantile columns.

    Returns
    -------
    float
        Dispersion ratio (predictive spread / observed spread); ``NaN`` when the
        observed spread is zero.
    """
    obs = np.asarray(obs, float)
    q = np.atleast_2d(np.asarray(quantile_preds, float))
    p = np.asarray(levels, float)
    order = np.argsort(p)
    p = p[order]
    q = q[:, order]

    med = _row_quantile(q, p, 0.5)
    dev = np.abs(q - med[:, None])                    # (n, m) abs dev from median
    span = float(p[-1] - p[0])
    if span <= 0:
        pred = np.nanmean(dev, axis=1)
    else:                                             # average |q(p) - m| over p
        pred = np.trapz(dev, p, axis=1) / span
    pred_disp = float(np.nanmean(pred[np.isfinite(pred)])) if pred.size else float("nan")

    yo = obs[np.isfinite(obs)]
    if yo.size == 0:
        return float("nan")
    obs_disp = float(np.mean(np.abs(yo - np.median(yo))))
    if obs_disp <= 0:
        return float("nan")
    return float(pred_disp / obs_disp)


# --------------------------------------------------------------------------- #
#  Prediction-interval coverage probability & mean width                      #
# --------------------------------------------------------------------------- #
def pi_coverage_width(obs, lower, upper, level: float | None = None) -> dict[str, float]:
    """Prediction-interval coverage probability (PICP) and mean width (MPIW).

    PICP is the fraction of observations inside ``[lower, upper]`` (the coverage
    ratio of El Ouahabi et al., 2026; ``CO`` of Zhang et al., 2023) and MPIW is
    the mean interval width (the average-width sharpness metric).  A reliable,
    sharp forecast has ``PICP`` close to the nominal ``level`` with the smallest
    ``MPIW``.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    lower, upper : array_like, shape (n,)
        Interval bounds.
    level : float, optional
        Nominal coverage (e.g. ``0.9``); when given, ``nominal`` and the signed
        ``picp_error = PICP - level`` are added to the result.

    Returns
    -------
    dict
        ``picp``, ``mpiw`` and ``n`` (finite-row count), plus ``nominal`` /
        ``picp_error`` when ``level`` is supplied.
    """
    picp = _coverage(obs, lower, upper)               # scalar empirical coverage
    mpiw = sharpness(lower, upper)
    obs = np.asarray(obs, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    n = int((np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)).sum())
    out: dict[str, float] = {"picp": float(picp), "mpiw": float(mpiw), "n": n}
    if level is not None:
        out["nominal"] = float(level)
        out["picp_error"] = float(picp - level)
    return out


# --------------------------------------------------------------------------- #
#  Threshold-weighted CRPS for high flows (Gneiting & Ranjan 2011)            #
# --------------------------------------------------------------------------- #
def _resolve_threshold(obs: np.ndarray, threshold, q_level: float) -> float:
    """Return the high-flow threshold ``t`` (explicit, or the ``q_level`` quantile)."""
    if threshold is not None:
        return float(threshold)
    yo = obs[np.isfinite(obs)]
    if yo.size == 0:
        return float("nan")
    return float(np.quantile(yo, q_level))


def twcrps(obs, ensemble, threshold: float | None = None, q_level: float = 0.9) -> float:
    """Threshold-weighted CRPS emphasising high flows (Gneiting & Ranjan, 2011).

    The threshold-weighted CRPS

    ``twCRPS(F, y) = integral (F(z) - 1{z >= y})^2 w(z) dz`` ,

    with the upper-tail weight ``w(z) = 1{z >= t}``, restricts the CRPS to the
    high-flow region above the threshold ``t`` so that errors on extreme events
    dominate the score (Zhang et al., 2023, eq. 3).  Using the chaining function
    ``v(z) = max(z, t)`` (an antiderivative of ``w``) it has the exact kernel
    form

    ``twCRPS = E|v(X) - v(y)| - 0.5 E|v(X) - v(X')|`` ,

    i.e. the ordinary ensemble CRPS evaluated on the ``v``-transformed
    ensemble/observations -- so it reuses
    :func:`sbc.validation.metrics.crps_ensemble`.  It is negatively oriented
    (lower is better).

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations (most naturally discharge for a high-flow threshold).
    ensemble : array_like, shape (n, m)
        Predictive members per observation -- e.g. a dense predictive-quantile
        grid or posterior samples of the same quantity as ``obs``.
    threshold : float, optional
        High-flow threshold ``t``.  If ``None`` it is the ``q_level`` quantile of
        the finite observations.
    q_level : float, default 0.9
        Observed-flow quantile used to derive ``threshold`` when it is not given
        (Zhang et al. use the 80/90/95 % percentiles of observed streamflow).

    Returns
    -------
    float
        Mean threshold-weighted CRPS over the finite observations.
    """
    obs = np.asarray(obs, float)
    ens = np.atleast_2d(np.asarray(ensemble, float))
    if ens.shape[0] == 1 and obs.shape[0] != 1:
        ens = np.repeat(ens, obs.shape[0], axis=0)
    t = _resolve_threshold(obs, threshold, q_level)
    if not np.isfinite(t):
        return float("nan")
    vy = np.maximum(obs, t)
    vens = np.maximum(ens, t)
    return crps_ensemble(vy, vens)


def twcrps_skill_score(obs, ensemble, ref_ensemble, threshold: float | None = None,
                       q_level: float = 0.9) -> float:
    """Threshold-weighted CRPS skill score ``1 - twCRPS_model / twCRPS_ref``.

    The same high-flow threshold is applied to both forecasts (derived from the
    observations when ``threshold`` is ``None``), so the model and reference are
    scored on an identical weight.  Positive means the model improves on the
    reference over high flows; ``NaN`` when the reference score is non-positive.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    ensemble, ref_ensemble : array_like, shape (n, m)
        Model and reference predictive members.
    threshold : float, optional
        High-flow threshold; defaults to the ``q_level`` quantile of ``obs``.
    q_level : float, default 0.9
        Quantile used to derive ``threshold`` when it is not given.

    Returns
    -------
    float
        ``1 - twCRPS_model / twCRPS_ref``.
    """
    t = _resolve_threshold(np.asarray(obs, float), threshold, q_level)
    tw_m = twcrps(obs, ensemble, threshold=t)
    tw_r = twcrps(obs, ref_ensemble, threshold=t)
    if not np.isfinite(tw_r) or tw_r <= 0.0:
        return float("nan")
    return float(1.0 - tw_m / tw_r)


# --------------------------------------------------------------------------- #
#  Compact bundle for the paper's tables                                       #
# --------------------------------------------------------------------------- #
def uq_score_summary(obs, quantile_preds, levels, *,
                     interval_levels=(0.9, 0.95),
                     tw_q_level: float = 0.9) -> dict[str, float]:
    """Scalar UQ-score bundle complementing :func:`calibration_summary`.

    Computes the Alpha reliability index, the dispersion ratio, the
    threshold-weighted CRPS for high flows, and -- for each requested central
    confidence level -- the Winkler score, PICP and MPIW.  The full predictive
    quantile grid doubles as the ensemble for :func:`twcrps`.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles aligned to ``levels`` (the predictive distribution).
    levels : array_like, shape (m,)
        Cumulative probabilities of the quantile columns; must span the tails of
        every requested ``interval_levels``.
    interval_levels : tuple of float
        Central confidence levels for the Winkler / PICP / MPIW reporting.
    tw_q_level : float
        Observed-flow quantile defining the high-flow threshold of :func:`twcrps`.

    Returns
    -------
    dict
        ``alpha_reliability``, ``dispersion_ratio``, ``twcrps`` and ``n`` plus
        ``winkler_{c}`` / ``picp_{c}`` / ``mpiw_{c}`` per central level ``c``.
    """
    obs = np.asarray(obs, float)
    q = np.atleast_2d(np.asarray(quantile_preds, float))
    p = np.asarray(levels, float)
    out: dict[str, float] = {
        "alpha_reliability": alpha_reliability(obs, q, p),
        "dispersion_ratio": dispersion_ratio(obs, q, p),
        "twcrps": twcrps(obs, q, q_level=tw_q_level),
        "n": int(np.isfinite(obs).sum()),
    }
    for c in interval_levels:
        alpha = 1.0 - c
        lo = _row_quantile(q, p, alpha / 2.0)
        hi = _row_quantile(q, p, 1.0 - alpha / 2.0)
        out[f"winkler_{c:g}"] = winkler_interval_score(obs, lo, hi, alpha)
        cw = pi_coverage_width(obs, lo, hi, level=c)
        out[f"picp_{c:g}"] = cw["picp"]
        out[f"mpiw_{c:g}"] = cw["mpiw"]
    return out


# --------------------------------------------------------------------------- #
#  Self-test: tiny QRF fit on synthetic; evaluate in discharge space          #
# --------------------------------------------------------------------------- #
def _selftest() -> None:  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.probabilistic_baselines import QRFCorrector
    from ..schemas import OBS_COL, validate
    from ..synthetic import generate
    from ..validation.splits import temporal_split

    df = validate(classify_regimes(build_features(
        generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3),
                 seed=13), scale="decadal"))).reset_index(drop=True)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    print(f"[uq_scores] gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    # tiny gradient-boosted quantile-regression forest -- train a grid that
    # genuinely spans the 0.025/0.975 tails so the 90 % and 95 % intervals differ
    qrf = QRFCorrector(method="gbr", n_estimators=80, learning_rate=0.08,
                       max_depth=2, min_samples_leaf=15, seed=0,
                       quantile_levels=(0.025, 0.05, 0.1, 0.25, 0.5,
                                        0.75, 0.9, 0.95, 0.975)).fit(train)

    levels = np.round(np.linspace(0.025, 0.975, 39), 4)
    # evaluate in DISCHARGE space (high-flow twCRPS is most meaningful there)
    qd = qrf.predict_discharge_quantiles(test, tuple(levels))   # (n, 39) discharge grid
    y = test[OBS_COL].to_numpy(float)

    # central 90 % prediction interval (alpha = 0.10) ---------------------- #
    lo90 = _row_quantile(qd, levels, 0.05)
    hi90 = _row_quantile(qd, levels, 0.95)

    ws = winkler_interval_score(y, lo90, hi90, alpha=0.10)
    # reference: a wide climatological band from the training observations
    yo_tr = train[OBS_COL].to_numpy(float)
    ref_lo = np.full_like(y, float(np.nanquantile(yo_tr, 0.05)))
    ref_hi = np.full_like(y, float(np.nanquantile(yo_tr, 0.95)))
    wss = winkler_skill_score(y, lo90, hi90, ref_lo, ref_hi, alpha=0.10)

    alpha_idx = alpha_reliability(y, qd, levels)
    disp = dispersion_ratio(y, qd, levels)
    cw = pi_coverage_width(y, lo90, hi90, level=0.90)
    tw = twcrps(y, qd, q_level=0.90)
    ref_ens = np.repeat(np.nanquantile(yo_tr, levels)[None, :], len(y), axis=0)
    tw_ss = twcrps_skill_score(y, qd, ref_ens, q_level=0.90)
    summary = uq_score_summary(y, qd, levels)

    # validity checks ------------------------------------------------------ #
    ws_each = winkler_interval_score(y, lo90, hi90, alpha=0.10, reduce="none")
    width90 = sharpness(lo90, hi90)
    assert np.isfinite(ws) and ws >= width90 - 1e-6, "Winkler below interval width"
    assert np.all(ws_each[np.isfinite(ws_each)] >= -1e-9), "negative interval score"
    assert 0.0 <= alpha_idx <= 1.0, "alpha index out of [0,1]"
    assert np.isfinite(disp) and disp > 0, "bad dispersion ratio"
    assert 0.0 <= cw["picp"] <= 1.0 and cw["mpiw"] > 0, "bad PICP/MPIW"
    assert np.isfinite(tw) and tw >= 0.0, "bad twCRPS"
    assert tw <= crps_ensemble(y, qd) + 1e-6, "twCRPS exceeds unweighted CRPS"
    assert set(summary) >= {"alpha_reliability", "dispersion_ratio", "twcrps",
                            "winkler_0.9", "picp_0.9", "mpiw_0.9"}, "summary keys"

    print(f"[uq_scores] Winkler(90%)={ws:.3f}  WSS_vs_clim={wss:+.3f}  "
          f"(MPIW={cw['mpiw']:.3f})")
    print(f"[uq_scores] Alpha-reliability={alpha_idx:.3f}  dispersion_ratio={disp:.3f}")
    print(f"[uq_scores] PICP(90%)={cw['picp']:.3f} (err={cw['picp_error']:+.3f})  "
          f"MPIW(90%)={cw['mpiw']:.3f}")
    print(f"[uq_scores] twCRPS(>=Q90)={tw:.4f}  twCRPSS_vs_clim={tw_ss:+.3f}  "
          f"| CRPS={crps_ensemble(y, qd):.4f}")
    print(f"[uq_scores] summary={ {k: round(v, 3) for k, v in summary.items()} }")
    print("[uq_scores] SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
