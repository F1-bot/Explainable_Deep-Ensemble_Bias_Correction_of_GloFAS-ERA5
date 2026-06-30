"""Decision-relevant and extreme-flow probabilistic skill for the corrector.

KGE' and CRPS summarise *average* hydrograph and *average* distributional skill,
but an operational user asks sharper, decision-shaped questions: *will the flow
exceed the freshet flood level this decade?*, *will it drop below the irrigation
low-flow threshold?*, and *does the correction get the annual peak magnitude and
the top-of-the-flow-duration-curve right?*  Those are scored not by a continuous
efficiency but by the verification metrics of binary-event and extreme-value
forecasting (Tran et al., 2025, who evaluate streamflow corrections by
threshold-exceedance Brier/ROC skill on freshet and low-flow events).

This module supplies exactly that layer of *decision-relevant* evidence, reusing
the project's existing machinery rather than re-deriving it:

Threshold-exceedance skill (high-/low-flow events)
--------------------------------------------------
* :func:`threshold_exceedance_skill` -- given a predictive **quantile grid** and a
  flow threshold (e.g. the observed ``Q90`` freshet level or the ``Q10``
  low-flow level), reconstruct the per-row forecast probability of the event
  ``Y > T`` (upper tail) or ``Y < T`` (lower tail) from the predictive CDF and
  score it with the **Brier score**, the **Brier skill score** against the
  climatological base rate, the **ROC-AUC** (discrimination), and the **Murphy
  reliability / resolution / uncertainty** decomposition.
* :func:`reliability_table` -- the binned forecast-probability-vs-observed-frequency
  reliability diagram data (distinct from the *quantile*-reliability curve in
  :mod:`sbc.validation.calibration`, which calibrates the whole predictive CDF).
* :func:`exceedance_skill_table` -- the compact paper table that loops the above
  over several decision thresholds (Q90 freshet high-flow + Q10 low-flow by
  default).

Extreme / peak-flow deterministic skill
---------------------------------------
* :func:`peak_flow_metrics` -- annual-maximum percent bias and a KGE' computed on
  the **top-2 %** of observed flows (the high-flow tail that dominates flood
  design), plus the annual peak-timing error (reused from
  :mod:`sbc.validation.metrics`).

All functions are pure, NaN-aware NumPy/pandas; heavy / optional imports
(``scipy``, ``matplotlib``) are deferred into the functions that use them and
figures are written with the headless ``Agg`` backend.  Probabilistic skill is
evaluated in **discharge space** (pass discharge observations and
:meth:`~sbc.models.base.BaseCorrector.predict_discharge_quantiles` output) so the
thresholds are physical flow levels; the metrics themselves are space-agnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger
from .metrics import kge_prime, lognse, nse, pbias, peak_timing_error

log = get_logger(__name__)

__all__ = [
    "DEFAULT_DECISION_THRESHOLDS",
    "predictive_cdf",
    "exceedance_probability",
    "brier_score",
    "brier_skill_score",
    "roc_auc",
    "reliability_table",
    "brier_decomposition",
    "threshold_exceedance_skill",
    "exceedance_skill_table",
    "peak_flow_metrics",
    "save_exceedance_reliability",
    "save_roc_curve",
]

#: canonical decision thresholds: ``(observed-flow quantile, tail)`` pairs.  The
#: ``Q90`` upper-tail event is the snowmelt-freshet / flood exceedance; the
#: ``Q10`` lower-tail event is the irrigation-critical low-flow shortfall.
DEFAULT_DECISION_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.90, "upper"),
    (0.10, "lower"),
)


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


def _finite_pair(prob: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either the forecast probability or the outcome is non-finite."""
    prob = np.asarray(prob, float).ravel()
    outcome = np.asarray(outcome, float).ravel()
    m = np.isfinite(prob) & np.isfinite(outcome)
    return prob[m], outcome[m]


# --------------------------------------------------------------------------- #
#  Predictive CDF and event probability                                       #
# --------------------------------------------------------------------------- #
def predictive_cdf(quantile_preds, levels, threshold) -> np.ndarray:
    """Per-row predictive cumulative probability ``F_i(T)`` at a fixed threshold.

    For each row the predictive CDF is reconstructed from its quantile pairs
    ``(q_{i,j}, levels_j)`` (sorted and made monotone non-decreasing) and
    evaluated at the scalar ``threshold`` by linear interpolation.  Thresholds
    below the smallest / above the largest predicted quantile map to the closed
    endpoints ``0`` / ``1`` (flat extrapolation), exactly as the PIT in
    :mod:`sbc.validation.calibration`.

    Parameters
    ----------
    quantile_preds : array_like, shape (n, m) or (1, m)
        Predicted quantiles, column ``j`` at cumulative probability ``levels[j]``.
        A single shared row ``(1, m)`` is broadcast to every observation.
    levels : array_like, shape (m,)
        Cumulative probabilities of the quantile columns.
    threshold : float
        The value ``T`` at which to evaluate every row's predictive CDF.

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``F_i(T) = P(Y_i <= T)`` per row, in ``[0, 1]``.
    """
    q = np.atleast_2d(np.asarray(quantile_preds, float))
    p = np.asarray(levels, float)
    if q.shape[1] != p.shape[0]:
        raise ValueError("quantile_preds has a different number of columns than levels")
    order = np.argsort(p)
    p = p[order]
    q = q[:, order]
    t = float(threshold)
    n = q.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        qi = q[i]
        if not np.isfinite(qi).any():
            continue
        qi = np.maximum.accumulate(qi)
        out[i] = float(np.interp(t, qi, p, left=0.0, right=1.0))
    return out


def exceedance_probability(quantile_preds, levels, threshold, tail: str = "upper"
                           ) -> np.ndarray:
    """Per-row forecast probability of the threshold event.

    Parameters
    ----------
    quantile_preds, levels, threshold :
        See :func:`predictive_cdf`.
    tail : {"upper", "lower"}
        ``"upper"`` returns ``P(Y > T) = 1 - F(T)`` (high-flow exceedance);
        ``"lower"`` returns ``P(Y < T) = F(T)`` (low-flow shortfall).

    Returns
    -------
    numpy.ndarray, shape (n,)
        The per-row event probability in ``[0, 1]``.
    """
    cdf = predictive_cdf(quantile_preds, levels, threshold)
    tail = str(tail).lower()
    if tail == "upper":
        return 1.0 - cdf
    if tail == "lower":
        return cdf
    raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}")


def _binary_outcome(obs: np.ndarray, threshold: float, tail: str) -> np.ndarray:
    """Observed binary event indicator for the threshold/tail (NaN where obs NaN)."""
    obs = np.asarray(obs, float)
    out = np.where(obs > threshold, 1.0, 0.0) if tail == "upper" \
        else np.where(obs < threshold, 1.0, 0.0)
    return np.where(np.isfinite(obs), out, np.nan)


# --------------------------------------------------------------------------- #
#  Scalar verification scores                                                 #
# --------------------------------------------------------------------------- #
def brier_score(forecast_prob, outcome) -> float:
    """Mean squared error of a probability forecast, ``mean((p - y)^2)``.

    Parameters
    ----------
    forecast_prob : array_like
        Forecast probabilities of the event in ``[0, 1]``.
    outcome : array_like
        Binary observed outcomes ``{0, 1}``.

    Returns
    -------
    float
        The Brier score (0 is perfect, 0.25 is the no-skill climatology of a
        50 % base rate); ``NaN`` if no finite pairs.
    """
    p, y = _finite_pair(forecast_prob, outcome)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def brier_skill_score(forecast_prob, outcome, reference_prob=None) -> float:
    """Brier skill score ``1 - BS_model / BS_ref`` against a reference forecast.

    Parameters
    ----------
    forecast_prob : array_like
        Forecast probabilities of the event.
    outcome : array_like
        Binary observed outcomes.
    reference_prob : float or array_like, optional
        Reference forecast probability.  Defaults to the **climatological base
        rate** ``mean(outcome)`` (a constant forecast equal to the event
        frequency), the standard skill benchmark.

    Returns
    -------
    float
        Positive when the model beats the reference, ``0`` when it ties, ``NaN``
        when the reference Brier score is non-positive (e.g. no events).
    """
    p, y = _finite_pair(forecast_prob, outcome)
    if p.size == 0:
        return float("nan")
    base = float(y.mean()) if reference_prob is None else None
    ref = np.full_like(y, base) if base is not None else \
        np.broadcast_to(np.asarray(reference_prob, float), y.shape)
    bs_model = float(np.mean((p - y) ** 2))
    bs_ref = float(np.mean((ref - y) ** 2))
    if not np.isfinite(bs_ref) or bs_ref <= 0.0:
        return float("nan")
    return 1.0 - bs_model / bs_ref


def roc_auc(scores, outcome) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) identity.

    Measures *discrimination*: the probability that a randomly drawn event row
    receives a higher forecast probability than a randomly drawn non-event row.
    Ties contribute 0.5.  ``0.5`` is no skill, ``1.0`` perfect.

    Parameters
    ----------
    scores : array_like
        Forecast probabilities (or any monotone score) of the event.
    outcome : array_like
        Binary observed outcomes ``{0, 1}``.

    Returns
    -------
    float
        ROC-AUC, or ``NaN`` when one class is absent (AUC undefined).
    """
    s, y = _finite_pair(scores, outcome)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int(s.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(s)                       # average ranks break ties correctly
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# --------------------------------------------------------------------------- #
#  Reliability diagram and Murphy decomposition                               #
# --------------------------------------------------------------------------- #
def reliability_table(forecast_prob, outcome, n_bins: int = 10) -> pd.DataFrame:
    """Binned reliability-diagram data: forecast probability vs observed frequency.

    Forecast probabilities are grouped into ``n_bins`` equal-width bins on
    ``[0, 1]``; a *reliable* (calibrated) forecast has, in every bin, an observed
    event frequency equal to the mean forecast probability (the diagonal).

    Parameters
    ----------
    forecast_prob : array_like
        Forecast probabilities of the event.
    outcome : array_like
        Binary observed outcomes.
    n_bins : int, default 10
        Number of equal-width probability bins.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin with ``bin_lo``, ``bin_hi``, ``forecast``
        (mean forecast probability), ``observed`` (observed event frequency) and
        ``count``.
    """
    p, y = _finite_pair(forecast_prob, outcome)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    if p.size:
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
        for b in range(n_bins):
            sel = idx == b
            c = int(sel.sum())
            if c == 0:
                continue
            rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                         "forecast": float(p[sel].mean()),
                         "observed": float(y[sel].mean()), "count": c})
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "forecast",
                                       "observed", "count"])


def brier_decomposition(forecast_prob, outcome, n_bins: int = 10) -> dict[str, float]:
    """Murphy (1973) reliability / resolution / uncertainty Brier decomposition.

    With the forecast binned by :func:`reliability_table`,

    ``BS ~= REL - RES + UNC``

    where ``UNC = o_bar (1 - o_bar)`` is the climatological uncertainty (base
    rate ``o_bar``), ``REL = sum_k n_k/N (f_k - o_k)^2`` the reliability (0 is
    perfectly calibrated, lower is better) and ``RES = sum_k n_k/N (o_k - o_bar)^2``
    the resolution (higher is better -- the forecast separates events from
    non-events).

    Parameters
    ----------
    forecast_prob, outcome : array_like
        Forecast probabilities and binary outcomes.
    n_bins : int, default 10
        Number of bins (passed to :func:`reliability_table`).

    Returns
    -------
    dict
        ``reliability``, ``resolution``, ``uncertainty`` and ``base_rate``.
    """
    p, y = _finite_pair(forecast_prob, outcome)
    if p.size == 0:
        return {"reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "base_rate": float("nan")}
    o_bar = float(y.mean())
    tbl = reliability_table(p, y, n_bins=n_bins)
    n = float(p.size)
    w = tbl["count"].to_numpy(float) / n
    rel = float(np.sum(w * (tbl["forecast"].to_numpy(float)
                            - tbl["observed"].to_numpy(float)) ** 2))
    res = float(np.sum(w * (tbl["observed"].to_numpy(float) - o_bar) ** 2))
    return {"reliability": rel, "resolution": res,
            "uncertainty": o_bar * (1.0 - o_bar), "base_rate": o_bar}


# --------------------------------------------------------------------------- #
#  Threshold-exceedance skill bundle                                          #
# --------------------------------------------------------------------------- #
def threshold_exceedance_skill(obs, quantile_preds, levels, q_threshold: float = 0.9, *,
                               tail: str = "upper", threshold: float | None = None,
                               reference_prob: float | None = None,
                               n_bins: int = 10) -> dict[str, object]:
    """Decision-relevant exceedance skill for one high-/low-flow threshold.

    The flow threshold ``T`` is, by default, the ``q_threshold`` empirical
    quantile of the observations (e.g. ``q_threshold=0.9`` -> the ``Q90`` freshet
    level; pair with ``q_threshold=0.1, tail="lower"`` for the ``Q10`` low-flow
    level), or supplied directly via ``threshold``.  The per-row forecast event
    probability is read from the predictive CDF
    (:func:`exceedance_probability`), the observed binary event from ``obs``, and
    the pair is scored with the Brier score, the Brier skill score against the
    climatological base rate, the ROC-AUC and the Murphy decomposition.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations (discharge), used both for the event indicator and -- when
        ``threshold`` is ``None`` -- to define the threshold quantile.
    quantile_preds : array_like, shape (n, m)
        Predicted quantiles aligned to ``levels`` (discharge space, e.g.
        :meth:`~sbc.models.base.BaseCorrector.predict_discharge_quantiles`).
    levels : array_like, shape (m,)
        Cumulative probabilities of the quantile columns.
    q_threshold : float, default 0.9
        Observation quantile defining the threshold ``T`` (ignored if
        ``threshold`` is given).
    tail : {"upper", "lower"}
        Event direction: ``"upper"`` for ``Y > T`` (high-flow exceedance),
        ``"lower"`` for ``Y < T`` (low-flow shortfall).
    threshold : float, optional
        Explicit flow threshold; overrides ``q_threshold``.
    reference_prob : float, optional
        Reference probability for the skill score (default: climatological base
        rate).
    n_bins : int, default 10
        Reliability-diagram / decomposition bins.

    Returns
    -------
    dict
        Scalar keys ``threshold``, ``q_threshold``, ``tail``, ``n``,
        ``n_events``, ``base_rate``, ``brier``, ``brier_ref``, ``bss``,
        ``roc_auc``, ``reliability``, ``resolution``, ``uncertainty``, plus the
        binned ``reliability_table`` (a :class:`pandas.DataFrame`).
    """
    obs = np.asarray(obs, float)
    if threshold is None:
        finite = obs[np.isfinite(obs)]
        if finite.size == 0:
            raise ValueError("obs has no finite values to define a threshold")
        threshold = float(np.quantile(finite, float(q_threshold)))
    else:
        threshold = float(threshold)

    prob = exceedance_probability(quantile_preds, levels, threshold, tail=tail)
    y = _binary_outcome(obs, threshold, str(tail).lower())
    p, yy = _finite_pair(prob, y)

    bs = brier_score(p, yy)
    base = float(yy.mean()) if reference_prob is None else float(reference_prob)
    bs_ref = float(np.mean((base - yy) ** 2)) if yy.size else float("nan")
    dec = brier_decomposition(p, yy, n_bins=n_bins)
    return {
        "threshold": threshold,
        "q_threshold": float(q_threshold),
        "tail": str(tail).lower(),
        "n": int(yy.size),
        "n_events": int(yy.sum()),
        "base_rate": dec["base_rate"],
        "brier": bs,
        "brier_ref": bs_ref,
        "bss": brier_skill_score(p, yy, reference_prob=reference_prob),
        "roc_auc": roc_auc(p, yy),
        "reliability": dec["reliability"],
        "resolution": dec["resolution"],
        "uncertainty": dec["uncertainty"],
        "reliability_table": reliability_table(p, yy, n_bins=n_bins),
    }


def exceedance_skill_table(obs, quantile_preds, levels,
                           thresholds=DEFAULT_DECISION_THRESHOLDS, *,
                           reference_prob: float | None = None,
                           n_bins: int = 10) -> pd.DataFrame:
    """Compact decision-skill table over several high-/low-flow thresholds.

    Loops :func:`threshold_exceedance_skill` over each ``(quantile, tail)`` pair
    (default: ``Q90`` freshet high-flow + ``Q10`` low-flow) and collects the
    scalar scores into one tidy table for the paper.

    Parameters
    ----------
    obs : array_like, shape (n,)
        Observations (discharge).
    quantile_preds : array_like, shape (n, m)
        Predicted discharge quantiles aligned to ``levels``.
    levels : array_like, shape (m,)
        Cumulative probabilities of the quantile columns.
    thresholds : sequence of (float, str)
        ``(observation quantile, tail)`` pairs; defaults to
        :data:`DEFAULT_DECISION_THRESHOLDS`.
    reference_prob : float, optional
        Reference probability for the skill score (default: per-event base rate).
    n_bins : int, default 10
        Reliability bins.

    Returns
    -------
    pandas.DataFrame
        One row per threshold with the scalar exceedance-skill columns (the
        per-bin reliability table is omitted here).
    """
    rows = []
    for q, tail in thresholds:
        r = threshold_exceedance_skill(obs, quantile_preds, levels, q_threshold=float(q),
                                       tail=str(tail), reference_prob=reference_prob,
                                       n_bins=n_bins)
        r.pop("reliability_table", None)
        rows.append({"event": f"Q{int(round(float(q) * 100)):d}_{tail}", **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Extreme / peak-flow deterministic skill                                    #
# --------------------------------------------------------------------------- #
def _annual_extrema(df: pd.DataFrame) -> pd.DataFrame:
    """Per ``(code, year)`` annual-maximum of obs and sim (finite pairs only)."""
    rows = []
    for keys, g in df.groupby(["code", "year"], sort=True):
        o = g["obs"].to_numpy(float)
        s = g["sim"].to_numpy(float)
        if np.isfinite(o).sum() == 0 or np.isfinite(s).sum() == 0:
            continue
        rows.append({"obs_max": float(np.nanmax(o)), "sim_max": float(np.nanmax(s))})
    return pd.DataFrame(rows, columns=["obs_max", "sim_max"])


def peak_flow_metrics(obs, sim, dates, *, codes=None, high_flow_frac: float = 0.02
                      ) -> dict[str, float]:
    """Extreme / peak-flow skill: annual-max bias and top-2 % high-flow KGE'.

    Three complementary extreme-value diagnostics, none captured by a basin-wide
    KGE':

    * **Annual-maximum bias** -- the percent bias of the mean annual-maximum
      discharge (the flood-design statistic), with annual maxima taken per
      ``(gauge, year)`` so gauges of different size do not contaminate one
      another.
    * **High-flow KGE'** -- the modified Kling-Gupta efficiency restricted to the
      top ``high_flow_frac`` (default 2 %) of *observed* flows, i.e. the peak of
      the flow-duration curve where the snowmelt freshet and floods live.
    * **Peak-timing error** -- the mean absolute annual peak-date error, reused
      from :func:`sbc.validation.metrics.peak_timing_error`.

    Parameters
    ----------
    obs, sim : array_like, shape (n,)
        Observed and simulated/corrected discharge ``[m3 s-1]``.
    dates : array_like, shape (n,)
        Period dates (used for the annual grouping and peak timing).
    codes : array_like, optional
        Gauge identifier per row.  When given, annual maxima and peak timing are
        computed per ``(gauge, year)``; otherwise per year only (matching the
        existing :func:`peak_timing_error` convention).
    high_flow_frac : float, default 0.02
        Top fraction of observed flows defining the high-flow tail.

    Returns
    -------
    dict
        ``amax_pbias`` (% bias of mean annual max), ``amax_mean_obs`` /
        ``amax_mean_sim``, ``amax_kge`` (KGE' of the paired annual maxima),
        ``n_years``, ``high_flow_threshold``, ``n_high``, ``high_flow_kge`` /
        ``_r`` / ``_beta`` / ``_gamma``, ``high_flow_nse``, ``high_flow_lognse``,
        ``high_flow_pbias`` and ``peak_timing_err``.
    """
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    n = obs.shape[0]
    code_arr = (np.asarray(codes).astype(str) if codes is not None
                else np.zeros(n, dtype=int).astype(str))
    df = pd.DataFrame({"code": code_arr, "date": d.to_numpy(),
                       "obs": obs, "sim": sim})
    df["year"] = pd.to_datetime(df["date"]).dt.year

    out: dict[str, float] = {"n": int(np.isfinite(obs + sim).sum())}

    # -- annual-maximum bias ------------------------------------------------- #
    amax = _annual_extrema(df)
    if len(amax):
        mo = float(amax["obs_max"].mean())
        ms = float(amax["sim_max"].mean())
        out.update({
            "n_years": int(len(amax)),
            "amax_mean_obs": mo, "amax_mean_sim": ms,
            "amax_pbias": float(100.0 * (ms - mo) / mo) if mo != 0 else float("nan"),
            "amax_kge": kge_prime(amax["obs_max"].to_numpy(),
                                  amax["sim_max"].to_numpy())["kge"],
        })
    else:
        out.update({"n_years": 0, "amax_mean_obs": float("nan"),
                    "amax_mean_sim": float("nan"), "amax_pbias": float("nan"),
                    "amax_kge": float("nan")})

    # -- top-fraction high-flow KGE' ----------------------------------------- #
    finite = np.isfinite(obs) & np.isfinite(sim)
    if finite.sum() >= 5:
        thr = float(np.quantile(obs[finite], 1.0 - float(high_flow_frac)))
        hmask = finite & (obs >= thr)
        oh, sh = obs[hmask], sim[hmask]
        k = kge_prime(oh, sh)
        out.update({
            "high_flow_threshold": thr, "n_high": int(hmask.sum()),
            "high_flow_kge": k["kge"], "high_flow_r": k["r"],
            "high_flow_beta": k["beta"], "high_flow_gamma": k["gamma"],
            "high_flow_nse": nse(oh, sh), "high_flow_lognse": lognse(oh, sh),
            "high_flow_pbias": pbias(oh, sh),
        })
    else:
        out.update({"high_flow_threshold": float("nan"), "n_high": 0,
                    "high_flow_kge": float("nan"), "high_flow_r": float("nan"),
                    "high_flow_beta": float("nan"), "high_flow_gamma": float("nan"),
                    "high_flow_nse": float("nan"), "high_flow_lognse": float("nan"),
                    "high_flow_pbias": float("nan")})

    # -- peak-timing error (reuse metrics; per-gauge when codes given) -------- #
    if codes is None:
        out["peak_timing_err"] = peak_timing_error(df["date"], obs, sim)
    else:
        errs, weights = [], []
        for _, g in df.groupby("code", sort=False):
            e = peak_timing_error(g["date"], g["obs"].to_numpy(), g["sim"].to_numpy())
            if np.isfinite(e):
                errs.append(e)
                weights.append(int(g["year"].nunique()))
        out["peak_timing_err"] = (float(np.average(errs, weights=weights))
                                  if errs else float("nan"))
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


def save_exceedance_reliability(forecast_prob, outcome, *, n_bins: int = 10,
                                path: str | Path | None = None,
                                title: str | None = None) -> Path:
    """Write the forecast-probability reliability diagram for a threshold event.

    The curve hugs the dashed 1:1 diagonal when the forecast probabilities are
    calibrated; points above the diagonal are under-confident, below are
    over-confident.  The marker area scales with the bin count.

    Parameters
    ----------
    forecast_prob, outcome : array_like
        Event forecast probabilities and binary outcomes.
    n_bins : int, default 10
        Reliability bins.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to
        ``results/figures/decision_reliability.png``).
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    tbl = reliability_table(forecast_prob, outcome, n_bins=n_bins)
    dec = brier_decomposition(forecast_prob, outcome, n_bins=n_bins)
    out = _resolve_path(path, "decision_reliability.png")

    fig, ax = _new_axes((5.0, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0, label="perfect")
    if not tbl.empty:
        sizes = 20.0 + 180.0 * tbl["count"].to_numpy(float) / tbl["count"].max()
        ax.plot(tbl["forecast"], tbl["observed"], "-", color="C0", lw=1.2, zorder=1)
        ax.scatter(tbl["forecast"], tbl["observed"], s=sizes, color="C0",
                   edgecolor="white", zorder=2,
                   label=f"REL={dec['reliability']:.3f} RES={dec['resolution']:.3f}")
    if np.isfinite(dec["base_rate"]):
        ax.axhline(dec["base_rate"], color="C3", lw=0.8, ls=":",
                   label=f"base rate={dec['base_rate']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed event frequency")
    ax.set_title(title or "Exceedance reliability diagram")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote exceedance reliability diagram -> %s (REL=%.4f)",
             out, dec["reliability"])
    return out


def save_roc_curve(forecast_prob, outcome, *, path: str | Path | None = None,
                   title: str | None = None) -> Path:
    """Write the ROC curve (hit rate vs false-alarm rate) for a threshold event.

    Parameters
    ----------
    forecast_prob, outcome : array_like
        Event forecast probabilities and binary outcomes.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to ``results/figures/decision_roc.png``).
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    p, y = _finite_pair(forecast_prob, outcome)
    auc = roc_auc(p, y)
    out = _resolve_path(path, "decision_roc.png")

    # sweep thresholds from high to low probability -> monotone ROC
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    n_pos = float((y > 0.5).sum())
    n_neg = float(y.size - n_pos)
    tpr = np.concatenate([[0.0], np.cumsum(ys) / n_pos]) if n_pos > 0 else np.array([0.0, 0.0])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - ys) / n_neg]) if n_neg > 0 else np.array([0.0, 0.0])

    fig, ax = _new_axes((5.0, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0, label="no skill")
    ax.plot(fpr, tpr, color="C0", lw=1.6,
            label=f"ROC (AUC={auc:.3f})" if np.isfinite(auc) else "ROC (AUC=n/a)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("hit rate")
    ax.set_title(title or "Exceedance ROC curve")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote ROC curve -> %s (AUC=%.4f)", out, auc)
    return out


# --------------------------------------------------------------------------- #
#  Self-test: small synthetic QRF fit (probabilistic) on a temporal split      #
# --------------------------------------------------------------------------- #
def _selftest() -> None:  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.probabilistic_baselines import QRFCorrector
    from ..schemas import OBS_COL, SIM_COL, validate
    from ..synthetic import generate
    from ..validation.splits import temporal_split

    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(df, scale="decadal"))
    df = validate(df).reset_index(drop=True)

    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    print(f"[decision] gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    # probabilistic model -> discharge quantile grid (decision space) ----------
    qrf = QRFCorrector(method="gbr", n_estimators=80, learning_rate=0.08,
                       max_depth=2, min_samples_leaf=15, seed=0).fit(train)
    levels = np.asarray(qrf.quantile_levels, float)
    qdis = qrf.predict_discharge_quantiles(test, tuple(levels))   # (n, m) discharge
    obs = test[OBS_COL].to_numpy(float)
    sim_raw = test[SIM_COL].to_numpy(float)
    sim_cor = qrf.predict(test)

    # -- threshold-exceedance skill (Q90 freshet high-flow) ------------------- #
    hi = threshold_exceedance_skill(obs, qdis, levels, q_threshold=0.9, tail="upper")
    lo = threshold_exceedance_skill(obs, qdis, levels, q_threshold=0.1, tail="lower")
    tbl = exceedance_skill_table(obs, qdis, levels)

    # internal consistency: Brier ~= REL - RES + UNC (binned identity) --------
    approx = hi["reliability"] - hi["resolution"] + hi["uncertainty"]

    # -- peak / extreme-flow skill (raw vs corrected) ------------------------- #
    codes = test["code"].to_numpy()
    pf_raw = peak_flow_metrics(obs, sim_raw, test["date"], codes=codes)
    pf_cor = peak_flow_metrics(obs, sim_cor, test["date"], codes=codes)

    # -- figures -------------------------------------------------------------- #
    prob_hi = exceedance_probability(qdis, levels, hi["threshold"], tail="upper")
    y_hi = _binary_outcome(obs, hi["threshold"], "upper")
    fig1 = save_exceedance_reliability(prob_hi, y_hi,
                                       path=PATHS.figures / "decision_reliability_selftest.png",
                                       title="QRF Q90 exceedance reliability (synthetic)")
    fig2 = save_roc_curve(prob_hi, y_hi, path=PATHS.figures / "decision_roc_selftest.png",
                          title="QRF Q90 exceedance ROC (synthetic)")

    # -- assertions ----------------------------------------------------------- #
    for r in (hi, lo):
        assert 0.0 <= r["brier"] <= 1.0, "Brier out of [0,1]"
        assert r["roc_auc"] != r["roc_auc"] or 0.0 <= r["roc_auc"] <= 1.0, "AUC range"
        assert r["bss"] <= 1.0 + 1e-9, "BSS exceeds 1"
        assert 0.0 <= r["base_rate"] <= 1.0, "base rate range"
    assert abs(hi["brier"] - approx) < 0.05, "Murphy decomposition inconsistent"
    assert set(tbl["event"]) == {"Q90_upper", "Q10_lower"}, "skill-table events"
    assert isinstance(hi["reliability_table"], pd.DataFrame)
    assert np.isfinite(pf_cor["high_flow_kge"]) and np.isfinite(pf_cor["amax_pbias"])
    assert pf_cor["n_high"] >= 1 and pf_cor["n_years"] >= 1
    assert fig1.exists() and fig2.exists(), "figures not written"

    print(f"[decision] Q90 high-flow: Brier={hi['brier']:.3f} BSS={hi['bss']:+.3f} "
          f"AUC={hi['roc_auc']:.3f} base={hi['base_rate']:.2f} "
          f"(REL={hi['reliability']:.3f} RES={hi['resolution']:.3f})")
    print(f"[decision] Q10 low-flow : Brier={lo['brier']:.3f} BSS={lo['bss']:+.3f} "
          f"AUC={lo['roc_auc']:.3f} base={lo['base_rate']:.2f}")
    print(f"[decision] peak/extreme  amax_pbias raw={pf_raw['amax_pbias']:+.1f}% "
          f"-> cor={pf_cor['amax_pbias']:+.1f}% | "
          f"top2% KGE' raw={pf_raw['high_flow_kge']:+.3f} -> cor={pf_cor['high_flow_kge']:+.3f}"
          f" | peak-timing {pf_raw['peak_timing_err']:.1f} -> {pf_cor['peak_timing_err']:.1f} d")
    print(f"[decision] skill table:\n{tbl[['event', 'brier', 'bss', 'roc_auc', 'n_events']]}")
    print(f"[decision] figures: {fig1.name}, {fig2.name}")
    print("[decision] SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
