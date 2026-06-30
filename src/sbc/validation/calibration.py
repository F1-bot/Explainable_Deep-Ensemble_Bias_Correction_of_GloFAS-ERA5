"""Probabilistic-forecast calibration diagnostics for the flagship.

A *probabilistic* claim ("RegimeProbNet emits calibrated predictive
distributions of the GloFAS log-residual") is only credible if the predictive
distributions are shown to be *reliable*: a nominal 90 % interval should cover
the truth ~90 % of the time, and the probability-integral-transform (PIT) of the
observations through the predictive CDF should be uniform.  This module supplies
exactly that evidence -- the standard uncertainty-calibration toolkit reviewers
expect alongside CRPS -- as pure, NaN-aware NumPy/pandas functions plus two
Agg-safe figure helpers.

Diagnostics
-----------
* :func:`pit_values` -- the probability integral transform ``F_t(y_t)`` from
  either a Gaussian ``(mu, sigma)`` forecast, a grid of predictive quantiles, or
  an ensemble.  A *calibrated* forecast yields PIT ~ Uniform(0, 1); a U-shaped
  histogram signals under-dispersion (over-confidence), a hump over-dispersion.
* :func:`reliability_curve` -- nominal cumulative probability versus the observed
  exceedance frequency ``P(y <= q_p)``; the diagonal is perfect calibration.
* :func:`coverage` / :func:`sharpness` -- empirical coverage of central
  prediction intervals against their nominal level, and the mean interval width
  (sharpness: subject to calibration, narrower is better).
* :func:`crps_skill_score` -- ``1 - CRPS_model / CRPS_ref``; positive means the
  probabilistic model beats the reference (raw GloFAS / climatology).
* :func:`pit_ks_statistic`, :func:`calibration_error`, :func:`calibration_summary`
  -- compact scalar summaries for the paper's tables.

The deterministic CRPS estimators live in :mod:`sbc.validation.metrics`
(``crps_gaussian``, ``crps_ensemble``) and the exact mixture CRPS in
:mod:`sbc.models.regime_prob_net` (``mixture_crps``); this module consumes their
outputs rather than re-deriving them.

All heavy / optional imports (``scipy``, ``matplotlib``) are deferred into the
functions that need them; figures are written with the headless ``Agg`` backend
to :pyattr:`sbc.config.Paths.figures` (``results/figures``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger

log = get_logger(__name__)

__all__ = [
    "pit_values",
    "reliability_curve",
    "coverage",
    "sharpness",
    "crps_skill_score",
    "pit_ks_statistic",
    "calibration_error",
    "calibration_summary",
    "save_reliability_diagram",
    "save_pit_histogram",
]


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
def _resolve_path(path: str | Path | None, default_name: str) -> Path:
    """Return a writable PNG path, defaulting under ``PATHS.figures``."""
    if path is None:
        PATHS.figures.mkdir(parents=True, exist_ok=True)
        return PATHS.figures / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _gaussian_pit(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """PIT under a Gaussian predictive distribution ``N(mu, sigma^2)``."""
    from scipy.stats import norm

    sigma = np.clip(np.asarray(sigma, float), 1e-12, None)
    return norm.cdf((np.asarray(obs, float) - np.asarray(mu, float)) / sigma)


def _quantile_pit(obs: np.ndarray, quant: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """PIT by linear interpolation of a per-row predictive quantile grid.

    For each observation ``y_i`` the predictive CDF is reconstructed from the
    quantile pairs ``(q_{i,j}, p_j)`` and evaluated at ``y_i``.  Observations
    below the smallest / above the largest predicted quantile are mapped to the
    closed-interval endpoints ``0`` / ``1`` (flat extrapolation).
    """
    obs = np.asarray(obs, float)
    quant = np.atleast_2d(np.asarray(quant, float))
    p = np.asarray(levels, float)
    order = np.argsort(p)
    p = p[order]
    quant = quant[:, order]
    if quant.shape[0] == 1 and obs.shape[0] != 1:  # shared grid -> broadcast
        quant = np.repeat(quant, obs.shape[0], axis=0)

    pit = np.empty(obs.shape[0], float)
    for i in range(obs.shape[0]):
        qi = np.maximum.accumulate(quant[i])  # enforce monotone non-decreasing
        pit[i] = float(np.interp(obs[i], qi, p, left=0.0, right=1.0))
    return pit


def _ensemble_pit(obs: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """Empirical-CDF PIT from an ensemble: ``mean(member <= obs)`` per row."""
    obs = np.asarray(obs, float)
    ens = np.asarray(ens, float)
    return (ens <= obs[:, None]).mean(axis=1)


def _interval_coverage(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of finite observations falling in the closed band ``[lo, hi]``."""
    obs = np.asarray(obs, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return float("nan")
    return float(((obs[m] >= lo[m]) & (obs[m] <= hi[m])).mean())


# --------------------------------------------------------------------------- #
#  Probability integral transform                                            #
# --------------------------------------------------------------------------- #
def pit_values(obs, pred, levels=None) -> np.ndarray:
    """Probability-integral transform ``F_t(y_t)`` of observations.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations (e.g. the log-residual target, or back-transformed
        discharge -- PIT is invariant under the monotone back-transform).
    pred : tuple or array_like
        The predictive distribution per observation, in one of three forms:

        * ``(mu, sigma)`` -- a length-2 tuple of arrays -> Gaussian CDF;
        * 2-D array ``(n, m)`` **with** ``levels`` -> predictive quantile grid,
          interpolated to the predictive CDF;
        * 2-D array ``(n, m)`` **without** ``levels`` -> ensemble members,
          giving the empirical-CDF PIT ``mean(member <= obs)``.
    levels : array_like, optional
        Cumulative probabilities of the quantile columns (required, and only
        used, for the quantile-grid form).

    Returns
    -------
    numpy.ndarray, shape (k,)
        PIT values in ``[0, 1]`` for the finite observations (non-finite rows
        dropped).  Uniformity of the result is the calibration target.
    """
    obs = np.asarray(obs, float)
    if isinstance(pred, tuple) and len(pred) == 2:
        mu, sigma = np.asarray(pred[0], float), np.asarray(pred[1], float)
        m = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(sigma)
        return _gaussian_pit(obs[m], mu[m], sigma[m])

    arr = np.asarray(pred, float)
    if arr.ndim != 2:
        raise ValueError("pred must be (mu, sigma) or a 2-D array of "
                         "quantiles/ensemble members")
    row_ok = np.isfinite(arr).any(axis=1)
    m = np.isfinite(obs) & row_ok
    if levels is not None:
        return _quantile_pit(obs[m], arr[m], levels)
    return _ensemble_pit(obs[m], arr[m])


def pit_ks_statistic(pit) -> float:
    """Kolmogorov-Smirnov distance of the PIT sample from Uniform(0, 1).

    ``0`` is perfect calibration; larger values flag mis-calibration.  Computed
    directly from the empirical CDF (no SciPy dependency).
    """
    p = np.sort(np.asarray(pit, float))
    p = p[np.isfinite(p)]
    n = p.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    d_plus = float(np.max(i / n - p))
    d_minus = float(np.max(p - (i - 1) / n))
    return max(d_plus, d_minus)


# --------------------------------------------------------------------------- #
#  Reliability / coverage / sharpness                                        #
# --------------------------------------------------------------------------- #
def reliability_curve(obs, quantile_preds, levels) -> pd.DataFrame:
    """Quantile-reliability curve: nominal vs observed non-exceedance frequency.

    For each predicted quantile level ``p_j`` the empirical frequency
    ``P(y <= q_{p_j})`` is computed; a calibrated forecast lies on the diagonal
    ``empirical == nominal``.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles, column ``j`` at cumulative probability
        ``levels[j]``.
    levels : array_like, shape (m,)
        Nominal cumulative probabilities of the quantile columns.

    Returns
    -------
    pandas.DataFrame
        Columns ``nominal``, ``empirical`` and ``count`` (one row per level,
        sorted by ``nominal``).
    """
    obs = np.asarray(obs, float)
    q = np.atleast_2d(np.asarray(quantile_preds, float))
    p = np.asarray(levels, float)
    if q.shape[1] != p.shape[0]:
        raise ValueError("quantile_preds has a different number of columns than levels")

    rows = []
    for j in range(p.shape[0]):
        col = q[:, j]
        m = np.isfinite(obs) & np.isfinite(col)
        emp = float((obs[m] <= col[m]).mean()) if m.any() else float("nan")
        rows.append({"nominal": float(p[j]), "empirical": emp, "count": int(m.sum())})
    return pd.DataFrame(rows).sort_values("nominal", ignore_index=True)


def coverage(obs, lower, upper, levels=None):
    """Empirical coverage of central prediction interval(s).

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    lower, upper : array_like
        Interval bounds.  Either 1-D ``(n,)`` for a single interval, or 2-D
        ``(n, L)`` for ``L`` nested intervals (one column per nominal level).
    levels : float or array_like, optional
        Nominal coverage level(s).  Required when ``lower`` / ``upper`` are 2-D;
        attaches the nominal column for the 1-D case.

    Returns
    -------
    float or pandas.DataFrame
        A scalar empirical coverage for the 1-D form when ``levels`` is ``None``;
        otherwise a tidy table with columns ``nominal``, ``empirical`` and
        ``mean_width`` (one row per interval).
    """
    obs = np.asarray(obs, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)

    if lo.ndim == 1:
        emp = _interval_coverage(obs, lo, hi)
        if levels is None:
            return emp
        return pd.DataFrame([{"nominal": float(np.ravel(levels)[0]),
                              "empirical": emp,
                              "mean_width": sharpness(lo, hi)}])

    if levels is None:
        raise ValueError("levels is required when lower/upper are 2-D")
    lv = np.atleast_1d(np.asarray(levels, float))
    if lv.shape[0] != lo.shape[1]:
        raise ValueError("number of levels must match the number of interval columns")
    rows = []
    for j in range(lo.shape[1]):
        rows.append({"nominal": float(lv[j]),
                     "empirical": _interval_coverage(obs, lo[:, j], hi[:, j]),
                     "mean_width": sharpness(lo[:, j], hi[:, j])})
    return pd.DataFrame(rows).sort_values("nominal", ignore_index=True)


def sharpness(lower, upper) -> float:
    """Mean predictive-interval width ``mean(upper - lower)`` (NaN-aware).

    Works for a single interval (1-D bounds) or a stack of intervals (2-D), in
    which case the mean is taken over every finite width.  Lower is sharper;
    sharpness is only meaningful *conditional on* adequate coverage.
    """
    width = np.asarray(upper, float) - np.asarray(lower, float)
    width = width[np.isfinite(width)]
    return float(np.mean(width)) if width.size else float("nan")


def crps_skill_score(crps_model, crps_ref):
    """Continuous-ranked-probability skill score ``1 - CRPS_model / CRPS_ref``.

    Parameters
    ----------
    crps_model, crps_ref : float or array_like
        Mean CRPS of the model and of a reference forecast (raw GloFAS treated
        deterministically, or a climatological distribution).

    Returns
    -------
    float or numpy.ndarray
        Skill in ``(-inf, 1]``; ``0`` ties the reference, positive beats it.
        ``NaN`` where the reference CRPS is non-positive or non-finite.
    """
    model = np.asarray(crps_model, float)
    ref = np.asarray(crps_ref, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = 1.0 - model / ref
    ss = np.where(np.isfinite(ref) & (ref > 0), ss, np.nan)
    return float(ss) if ss.ndim == 0 else ss


def calibration_error(curve_or_obs, quantile_preds=None, levels=None) -> float:
    """Mean absolute calibration error of a reliability curve.

    Accepts either a precomputed :func:`reliability_curve` DataFrame, or raw
    ``(obs, quantile_preds, levels)`` (in which case the curve is built first).
    Returns ``mean(|empirical - nominal|)`` -- the average distance from the
    perfect-calibration diagonal (lower is better).
    """
    if isinstance(curve_or_obs, pd.DataFrame):
        curve = curve_or_obs
    else:
        curve = reliability_curve(curve_or_obs, quantile_preds, levels)
    d = (curve["empirical"] - curve["nominal"]).abs()
    return float(d.mean()) if len(d) else float("nan")


def calibration_summary(obs, quantile_preds, levels,
                        crps_model: float | None = None,
                        crps_ref: float | None = None,
                        central_levels=(0.5, 0.8, 0.9)) -> dict[str, float]:
    """Compact scalar calibration bundle for the paper's tables.

    Computes the PIT mean / KS statistic, the mean absolute calibration error of
    the quantile-reliability curve, the empirical coverage and sharpness of the
    requested central intervals, and (when both CRPS inputs are given) the CRPS
    skill score.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles aligned to ``levels``.
    levels : array_like, shape (m,)
        Cumulative probabilities of the quantile columns; must span the tails of
        every requested ``central_levels`` interval.
    crps_model, crps_ref : float, optional
        Mean CRPS of the model and reference for the skill score.
    central_levels : tuple of float
        Nominal central-interval coverages to report.

    Returns
    -------
    dict
        ``pit_mean``, ``pit_ks``, ``calibration_error`` plus
        ``cov_{p}`` / ``width_{p}`` per central level, and ``crpss`` when CRPS
        inputs are supplied.
    """
    obs = np.asarray(obs, float)
    q = np.atleast_2d(np.asarray(quantile_preds, float))
    p = np.asarray(levels, float)

    pit = pit_values(obs, q, levels=p)
    curve = reliability_curve(obs, q, p)
    out: dict[str, float] = {
        "pit_mean": float(np.nanmean(pit)) if pit.size else float("nan"),
        "pit_ks": pit_ks_statistic(pit),
        "calibration_error": calibration_error(curve),
        "n": int(np.isfinite(obs).sum()),
    }
    for c in central_levels:
        lo_p, hi_p = (1.0 - c) / 2.0, (1.0 + c) / 2.0
        lo = _interp_quantile(q, p, lo_p)
        hi = _interp_quantile(q, p, hi_p)
        out[f"cov_{c:g}"] = _interval_coverage(obs, lo, hi)
        out[f"width_{c:g}"] = sharpness(lo, hi)
    if crps_model is not None and crps_ref is not None:
        out["crpss"] = crps_skill_score(float(crps_model), float(crps_ref))
    return out


def _interp_quantile(quant: np.ndarray, levels: np.ndarray, target_p: float) -> np.ndarray:
    """Per-row predictive quantile at ``target_p`` by interpolating the grid."""
    order = np.argsort(levels)
    p = levels[order]
    quant = quant[:, order]
    n = quant.shape[0]
    out = np.empty(n, float)
    for i in range(n):
        qi = np.maximum.accumulate(quant[i])
        out[i] = float(np.interp(target_p, p, qi))
    return out


# --------------------------------------------------------------------------- #
#  Plotting helpers (Agg-safe, write PNGs to results/figures)                  #
# --------------------------------------------------------------------------- #
def _new_axes(figsize: tuple[float, float]):
    """Return a fresh (fig, ax) on the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def save_reliability_diagram(obs, quantile_preds, levels, *,
                             path: str | Path | None = None,
                             title: str | None = None) -> Path:
    """Write the quantile-reliability diagram (empirical vs nominal) to PNG.

    The forecast is calibrated when the curve hugs the dashed 1:1 diagonal; a
    curve below the diagonal indicates predicted quantiles that are too low
    (the forecast is biased high), and vice versa.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles aligned to ``levels``.
    levels : array_like, shape (m,)
        Nominal cumulative probabilities.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to ``results/figures/calibration_reliability.png``).
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    curve = reliability_curve(obs, quantile_preds, levels)
    mace = calibration_error(curve)
    out = _resolve_path(path, "calibration_reliability.png")

    fig, ax = _new_axes((5.0, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0, label="perfect")
    ax.plot(curve["nominal"], curve["empirical"], marker="o", ms=4, lw=1.5,
            color="C0", label=f"forecast (MACE={mace:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("nominal cumulative probability")
    ax.set_ylabel("observed frequency  P(y <= q)")
    ax.set_title(title or "Quantile reliability diagram")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote reliability diagram -> %s (MACE=%.4f)", out, mace)
    return out


def save_pit_histogram(pit, *, path: str | Path | None = None, bins: int = 10,
                       title: str | None = None) -> Path:
    """Write a PIT histogram (vs the Uniform(0, 1) reference) to PNG.

    A flat histogram around the dashed uniform line indicates calibration; a
    U-shape signals under-dispersion (intervals too narrow) and a central hump
    over-dispersion (intervals too wide).

    Parameters
    ----------
    pit : array_like
        PIT values from :func:`pit_values`.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to ``results/figures/calibration_pit_hist.png``).
    bins : int
        Number of histogram bins on ``[0, 1]``.
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    p = np.asarray(pit, float)
    p = p[np.isfinite(p)]
    out = _resolve_path(path, "calibration_pit_hist.png")
    ks = pit_ks_statistic(p)

    fig, ax = _new_axes((6.0, 4.5))
    ax.hist(p, bins=int(bins), range=(0.0, 1.0), density=True,
            color="C0", edgecolor="white", alpha=0.85,
            label=f"PIT (KS={ks:.3f})")
    ax.axhline(1.0, ls="--", color="0.4", lw=1.0, label="uniform")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value  F(y)")
    ax.set_ylabel("density")
    ax.set_title(title or "PIT histogram")
    ax.legend(loc="upper center", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote PIT histogram -> %s (KS=%.4f)", out, ks)
    return out


# --------------------------------------------------------------------------- #
#  Self-test: small synthetic RegimeProbNet fit (3 epochs) via predict_quantiles
# --------------------------------------------------------------------------- #
def _selftest() -> None:  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.regime_prob_net import RegimeProbNet, mixture_crps
    from ..schemas import OBS_COL, SIM_COL, TARGET_COL, make_target, validate
    from ..synthetic import generate
    from ..validation.metrics import crps_gaussian
    from ..validation.splits import temporal_split

    df = generate(scale="decadal", years=8, n_basins=3,
                  gauges_per_basin=(2, 3), seed=7)
    df = build_features(df, scale="decadal")
    df = classify_regimes(df)
    df = validate(df)
    df[TARGET_COL] = make_target(df[OBS_COL].values, df[SIM_COL].values)

    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].copy(), df[te_mask].copy()
    print(f"[calibration] gauges={df['code'].nunique()} "
          f"train={len(train)} test={len(test)}")

    model = RegimeProbNet(K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                          epochs=3, batch_size=256, patience=3,
                          lambda_gate=0.5, lambda_phys=0.0, seed=0, verbose=False)
    model.fit(train, valid=test)

    levels = np.round(np.linspace(0.05, 0.95, 19), 4)
    qpred = model.predict_quantiles(test, tuple(levels))   # (n, 19) residual quantiles
    y = test[TARGET_COL].to_numpy(float)

    # PIT, reliability, coverage, sharpness ---------------------------------
    pit = pit_values(y, qpred, levels=levels)
    curve = reliability_curve(y, qpred, levels)
    cov90_lo = _interp_quantile(qpred, levels, 0.05)
    cov90_hi = _interp_quantile(qpred, levels, 0.95)
    cov90 = coverage(y, cov90_lo, cov90_hi)
    width90 = sharpness(cov90_lo, cov90_hi)

    # Gaussian-PIT path (mixture mean/std) ----------------------------------
    w, mu, sigma = model._forward(test)
    pmean = (w * mu).sum(1)
    pvar = (w * (sigma ** 2 + mu ** 2)).sum(1) - pmean ** 2
    pit_g = pit_values(y, (pmean, np.sqrt(np.clip(pvar, 1e-12, None))))

    # CRPS skill score vs a climatological Gaussian reference ---------------
    crps_model = float(mixture_crps(y, w, mu, sigma).mean())
    crps_ref = crps_gaussian(y, np.full_like(y, y.mean()), np.full_like(y, y.std() + 1e-9))
    crpss = crps_skill_score(crps_model, crps_ref)

    summary = calibration_summary(y, qpred, levels,
                                  crps_model=crps_model, crps_ref=crps_ref)

    fig1 = save_reliability_diagram(y, qpred, levels,
                                    path=PATHS.figures / "calibration_reliability_selftest.png",
                                    title="RegimeProbNet reliability (synthetic)")
    fig2 = save_pit_histogram(pit, path=PATHS.figures / "calibration_pit_hist_selftest.png",
                              title="RegimeProbNet PIT (synthetic)")

    assert pit.min() >= 0.0 and pit.max() <= 1.0, "PIT out of [0,1]"
    assert pit_g.min() >= 0.0 and pit_g.max() <= 1.0, "Gaussian PIT out of [0,1]"
    assert set(curve.columns) == {"nominal", "empirical", "count"}, "reliability cols"
    assert 0.0 <= cov90 <= 1.0, "coverage out of range"
    assert np.isfinite(width90) and width90 > 0, "bad sharpness"
    assert np.isfinite(summary["pit_ks"]) and np.isfinite(crpss), "summary scalars"
    assert fig1.exists() and fig2.exists(), "figures not written"

    print(f"[calibration] PIT mean={pit.mean():.3f} KS={pit_ks_statistic(pit):.3f} "
          f"| MACE={calibration_error(curve):.3f} | cov90={cov90:.3f} width90={width90:.3f}")
    print(f"[calibration] CRPS model={crps_model:.4f} ref={crps_ref:.4f} CRPSS={crpss:+.3f}")
    print(f"[calibration] summary={ {k: round(v, 3) for k, v in summary.items()} }")
    print(f"[calibration] figures: {fig1.name}, {fig2.name}")
    print("[calibration] SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
