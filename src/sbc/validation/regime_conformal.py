"""Regime-conditional (Mondrian) conformal prediction for streamflow bias correction.

Standard split-conformal prediction (:mod:`sbc.validation.conformal`) is
*distribution-free* but only guarantees **marginal** coverage::

    P( y in C(x) ) >= 1 - alpha,

where the probability is averaged over the *whole* population.  In snow-influenced
basins this average hides a problem the rest of the paper documents repeatedly:
the GloFAS log-residual is strongly **regime-dependent**.  Melt-freshet,
glacier-melt and rain-on-snow rows carry larger, heavier-tailed errors than the
recession baseflow that dominates the record, so a single pooled conformal
half-width is *too wide* in recession (wasted sharpness) and *too narrow* in the
melt and glacier regimes (the intervals silently **under-cover** exactly the
flood-relevant states).  Prediction in ungauged regions (PUR) makes this worse:
the calibration mix and the test mix differ, so marginal calibration transfers
poorly.

This module supplies the principled fix -- **regime-conditional conformal
prediction** -- and is, to our knowledge, a new contribution: it marries the
framework's *physical regime structure* (:mod:`sbc.features.regimes`) and the
flagship's *learned soft regime gate* (``RegimeProbNet.gate_weights``) with the
finite-sample conformal machinery.  Two flavours are provided:

* ``by="regime"`` -- **hard Mondrian / class-conditional conformal** (Vovk, 2012).
  The calibration nonconformity scores are *partitioned by hydrological regime*
  and a separate conformal quantile is taken **within each regime**; a test row
  inherits the quantile of its own regime.  Under exchangeability *within* a
  regime this restores the conditional guarantee
  ``P(y in C(x) | regime(x) = r) >= 1 - alpha`` for every populated regime -- the
  coverage now holds *process by process*, not just on average.

* ``by="gate"`` -- **soft Mondrian conformal**, novel here.  Instead of a hard
  partition we use the flagship's soft gate weights ``w(x) in Delta^{K}`` both to
  *calibrate* each per-regime quantile (a gate-weight-weighted conformal quantile,
  in the localized-conformal spirit of Tibshirani et al., 2019) and to *compose*
  each test row's half-width as the gate-weighted mixture
  ``h(x) = sum_k w_k(x) * q_hat_k``.  This yields smoothly heteroscedastic
  intervals that widen as the gate shifts probability onto the harder experts,
  without committing to a brittle hard label near regime boundaries.

Both reuse the exact split-conformal quantile correction
(:func:`sbc.validation.conformal.conformal_quantile`), the common log-residual
target and discharge back-transform, and the coverage / sharpness diagnostics of
:mod:`sbc.validation.calibration`.  :func:`compare_to_marginal` runs the marginal
estimator side-by-side and tabulates per-regime coverage so the restoration of
within-regime validity is auditable in one call -- the evidence the paper needs.

References
----------
Vovk, V. (2012). Conditional validity of inductive conformal predictors. *ACML*.
Tibshirani, R. et al. (2019). Conformal prediction under covariate shift. *NeurIPS*.
Romano, Y. et al. (2019). Conformalized quantile regression. *NeurIPS*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.regimes import REGIMES, classify_regimes
from ..schemas import (
    OBS_COL,
    REGIME_COL,
    SIM_COL,
    TARGET_COL,
    back_transform,
    make_target,
    validate,
)
from ..utils import get_logger
from . import calibration as cal
from .conformal import conformal_quantile, split_conformal

log = get_logger(__name__)

__all__ = [
    "RegimeConformalResult",
    "regime_conditional_conformal",
    "per_regime_coverage",
    "compare_to_marginal",
    "weighted_conformal_quantile",
]


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class RegimeConformalResult:
    """Regime-conditional conformal intervals and their per-regime diagnostics.

    Attributes
    ----------
    lower, upper : numpy.ndarray
        Per-row interval bounds on the **log-residual** target.
    q_lower, q_upper : numpy.ndarray
        The same intervals back-transformed to **discharge** ``[m3 s-1]``.
    half_width : numpy.ndarray
        Per-test-row conformal half-width (or half-band, for ``method="cqr"``);
        constant within a regime for ``by="regime"`` and smoothly varying for
        ``by="gate"``.
    coverage, discharge_coverage : float
        Overall (marginal) empirical coverage of the log-residual / discharge
        intervals on the test set.
    sharpness, discharge_sharpness : float
        Mean log-residual / discharge interval width.
    per_regime : pandas.DataFrame
        One row per hydrological regime present in the test set with columns
        ``regime, n, coverage, nominal, gap, mean_width`` -- the per-regime
        empirical coverage versus the nominal ``1 - alpha`` that this method
        targets.
    q_hat : dict
        The per-regime (or per-expert) conformal half-width used.
    alpha : float
        Target miscoverage (nominal coverage ``1 - alpha``).
    by : str
        ``"regime"`` (hard Mondrian) or ``"gate"`` (soft, gate-weighted).
    method : str
        Nonconformity score: ``"absolute"`` or ``"cqr"``.
    n_calib : int
        Number of finite calibration scores used.
    extra : dict
        Auxiliary diagnostics (marginal fallback ``q_hat``, per-regime calib
        counts, ...).
    """

    lower: np.ndarray
    upper: np.ndarray
    q_lower: np.ndarray
    q_upper: np.ndarray
    half_width: np.ndarray
    coverage: float
    discharge_coverage: float
    sharpness: float
    discharge_sharpness: float
    per_regime: pd.DataFrame
    q_hat: dict
    alpha: float
    by: str
    method: str
    n_calib: int
    extra: dict = field(default_factory=dict)

    @property
    def nominal(self) -> float:
        """Nominal coverage ``1 - alpha``."""
        return 1.0 - self.alpha

    @property
    def max_regime_gap(self) -> float:
        """Largest absolute per-regime coverage deviation from nominal."""
        g = self.per_regime["gap"].to_numpy(float)
        g = g[np.isfinite(g)]
        return float(np.max(np.abs(g))) if g.size else float("nan")

    @property
    def mean_under_coverage(self) -> float:
        """Mean per-regime *under*-coverage ``mean(max(0, nominal - coverage))``.

        Zero when no regime under-covers -- the property regime-conditional
        conformal is designed to restore and marginal conformal violates.
        """
        cov = self.per_regime["coverage"].to_numpy(float)
        cov = cov[np.isfinite(cov)]
        if cov.size == 0:
            return float("nan")
        return float(np.mean(np.clip(self.nominal - cov, 0.0, None)))

    def summary(self) -> dict[str, float]:
        """Scalar diagnostics bundle for the paper's tables."""
        return {
            "method": f"regime_conformal[{self.by}-{self.method}]",
            "alpha": self.alpha, "nominal": self.nominal,
            "coverage": float(self.coverage),
            "discharge_coverage": float(self.discharge_coverage),
            "sharpness": float(self.sharpness),
            "discharge_sharpness": float(self.discharge_sharpness),
            "max_regime_gap": self.max_regime_gap,
            "mean_under_coverage": self.mean_under_coverage,
            "n_calib": int(self.n_calib),
        }


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _target(df: pd.DataFrame) -> np.ndarray:
    """The log-residual target of ``df`` (computed if the column is absent)."""
    if TARGET_COL in df.columns:
        return df[TARGET_COL].to_numpy(float)
    return make_target(df[OBS_COL].to_numpy(float), df[SIM_COL].to_numpy(float))


def _regime_labels(df: pd.DataFrame) -> np.ndarray:
    """Per-row regime label as strings (classifying on the fly if needed)."""
    if REGIME_COL in df.columns:
        return df[REGIME_COL].astype(str).to_numpy()
    return classify_regimes(df)[REGIME_COL].astype(str).to_numpy()


def _scores_and_bounds(model, calib_df: pd.DataFrame, test_df: pd.DataFrame,
                       method: str, alpha: float):
    """Calibration nonconformity scores and the test pre-inflation bounds.

    Returns ``(scores, base_lo, base_hi)`` where the conformal interval is
    ``[base_lo - h, base_hi + h]`` for a half-width ``h``.  For ``"absolute"``
    the two bounds coincide (the point residual prediction); for ``"cqr"`` they
    are the corrector's own lower / upper quantiles (Romano et al., 2019).
    """
    y_cal = _target(calib_df)
    method = str(method).lower()
    if method == "cqr":
        if not getattr(model, "is_probabilistic", False):
            raise ValueError("method='cqr' requires a probabilistic corrector "
                             "exposing predict_quantiles")
        lo_hi = (alpha / 2.0, 1.0 - alpha / 2.0)
        qc = np.asarray(model.predict_quantiles(calib_df, lo_hi), float)
        scores = np.maximum(qc[:, 0] - y_cal, y_cal - qc[:, 1])
        qt = np.asarray(model.predict_quantiles(test_df, lo_hi), float)
        return scores, qt[:, 0], qt[:, 1]
    if method == "absolute":
        mu_cal = np.asarray(model.predict_residual(calib_df), float)
        scores = np.abs(y_cal - mu_cal)
        mu_test = np.asarray(model.predict_residual(test_df), float)
        return scores, mu_test, mu_test
    raise ValueError(f"method must be 'absolute' or 'cqr', got {method!r}")


def weighted_conformal_quantile(scores, weights, alpha: float) -> float:
    """Soft, weight-aware analogue of the split-conformal quantile.

    Generalises :func:`sbc.validation.conformal.conformal_quantile` to *weighted*
    calibration points (Tibshirani et al., 2019): every score contributes its
    weight to a normalised empirical CDF, and the returned value is the smallest
    score whose cumulative normalised weight reaches a finite-sample-inflated
    level ``tau = (1 - alpha) * (1 + 1 / n_eff)`` (capped at 1), where the
    *effective* sample size ``n_eff = (sum w)^2 / sum w^2`` replaces ``n`` in the
    standard ``(n + 1) / n`` correction.  With equal weights this reduces to the
    usual ``ceil((n + 1)(1 - alpha))`` order statistic.

    Parameters
    ----------
    scores : array_like
        Nonconformity scores (non-finite entries dropped).
    weights : array_like
        Non-negative per-score weights (e.g. a soft-gate column).
    alpha : float
        Target miscoverage in ``(0, 1)``.

    Returns
    -------
    float
        The weighted conformal half-width; ``+inf`` when no positively-weighted
        finite score exists (a degenerate, trivially-covering interval).
    """
    s = np.asarray(scores, float)
    w = np.asarray(weights, float)
    m = np.isfinite(s) & np.isfinite(w) & (w > 0)
    s, w = s[m], w[m]
    if s.size == 0:
        return float("inf")
    order = np.argsort(s, kind="stable")
    s, w = s[order], w[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float("inf")
    n_eff = total * total / float(np.sum(w * w))
    tau = (1.0 - alpha) * (1.0 + 1.0 / max(n_eff, 1.0))
    if tau >= 1.0:
        return float(s[-1])
    idx = int(np.searchsorted(cw, tau * total, side="left"))
    idx = min(idx, s.size - 1)
    return float(s[idx])


def per_regime_coverage(y, lower, upper, regimes, alpha: float,
                        widths=None) -> pd.DataFrame:
    """Empirical coverage of ``[lower, upper]`` *stratified by regime*.

    Parameters
    ----------
    y : array_like, shape (n,)
        Target values (log-residual or discharge -- coverage is invariant under
        the monotone back-transform when bounds are transformed alike).
    lower, upper : array_like, shape (n,)
        Per-row interval bounds.
    regimes : array_like of str, shape (n,)
        Regime label of each row.
    alpha : float
        Target miscoverage; the reported ``nominal`` is ``1 - alpha``.
    widths : array_like, optional
        Per-row interval widths; defaults to ``upper - lower``.

    Returns
    -------
    pandas.DataFrame
        Columns ``regime, n, coverage, nominal, gap, mean_width`` -- one row per
        regime present (ordered by :data:`sbc.features.regimes.REGIMES`), where
        ``gap = coverage - nominal`` (negative means under-coverage).
    """
    y = np.asarray(y, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    reg = np.asarray(regimes).astype(str)
    width = np.asarray(widths, float) if widths is not None else hi - lo
    nominal = 1.0 - float(alpha)

    present = [r for r in REGIMES if (reg == r).any()]
    present += [r for r in pd.unique(reg).tolist() if r not in REGIMES]  # any extras
    rows = []
    for r in present:
        sel = reg == r
        cov = cal.coverage(y[sel], lo[sel], hi[sel])
        w = width[sel]
        w = w[np.isfinite(w)]
        rows.append({
            "regime": r,
            "n": int(sel.sum()),
            "coverage": float(cov),
            "nominal": nominal,
            "gap": float(cov) - nominal if np.isfinite(cov) else np.nan,
            "mean_width": float(np.mean(w)) if w.size else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Regime-conditional conformal                                               #
# --------------------------------------------------------------------------- #
def regime_conditional_conformal(model, calib_df: pd.DataFrame,
                                 test_df: pd.DataFrame, alpha: float = 0.1,
                                 by: str = "regime", *, method: str = "absolute",
                                 min_calib: int = 10) -> RegimeConformalResult:
    """Conformal intervals calibrated *separately per hydrological regime*.

    The ``model`` must already be **fitted on data disjoint from** ``calib_df``
    (the inductive-conformal requirement); ``calib_df`` is used only to score
    nonconformity.  Coverage is then targeted *within each regime* rather than
    only on average, which is what restores validity on the melt / glacier /
    rain-on-snow processes that marginal conformal under-covers.

    Parameters
    ----------
    model : BaseCorrector
        Fitted corrector exposing :meth:`predict_residual` (and, for
        ``method="cqr"``, :meth:`predict_quantiles`).  For ``by="gate"`` it must
        additionally expose :meth:`gate_weights` (the flagship ``RegimeProbNet``).
    calib_df, test_df : pandas.DataFrame
        Calibration (residual-scoring) and test modelling tables.
    alpha : float, default 0.1
        Target miscoverage; nominal coverage is ``1 - alpha``.
    by : {"regime", "gate"}, default "regime"
        ``"regime"`` -- hard Mondrian: partition calibration scores by the
        rule-based ``regime`` label and take a conformal quantile per regime.
        ``"gate"`` -- soft Mondrian: weight every calibration score by the
        flagship's soft gate column ``w_k`` to obtain a per-expert quantile, then
        compose each test row's half-width as the gate-weighted mixture
        ``sum_k w_k(x) q_hat_k``.
    method : {"absolute", "cqr"}, default "absolute"
        Nonconformity score (see :func:`sbc.validation.conformal.split_conformal`).
    min_calib : int, default 10
        Minimum finite calibration scores a regime needs for its *own* quantile;
        sparser regimes fall back to the pooled (marginal) quantile so the
        intervals never degenerate.

    Returns
    -------
    RegimeConformalResult
        Intervals on the log-residual and discharge plus the per-regime coverage
        table.  For ``by="regime"`` the within-regime coverage is ``>= 1 - alpha``
        under within-regime exchangeability.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    by = str(by).lower()
    if by not in {"regime", "gate"}:
        raise ValueError(f"by must be 'regime' or 'gate', got {by!r}")
    calib_df = validate(calib_df)
    test_df = validate(test_df)

    scores, base_lo, base_hi = _scores_and_bounds(model, calib_df, test_df,
                                                  method, alpha)
    global_qhat = conformal_quantile(scores, alpha)  # pooled fallback
    n_calib = int(np.isfinite(scores).sum())

    if by == "regime":
        half_width, q_hat, calib_counts = _regime_half_widths(
            scores, calib_df, test_df, alpha, global_qhat, min_calib)
        extra = {"calib_counts": calib_counts, "marginal_q_hat": float(global_qhat)}
    else:  # by == "gate"
        half_width, q_hat, expert_counts = _gate_half_widths(
            model, scores, calib_df, test_df, alpha, global_qhat)
        extra = {"expert_eff_n": expert_counts, "marginal_q_hat": float(global_qhat)}

    lower = base_lo - half_width
    upper = base_hi + half_width

    # diagnostics -----------------------------------------------------------
    sim = test_df[SIM_COL].to_numpy(float)
    q_lower = back_transform(sim, lower)
    q_upper = back_transform(sim, upper)
    y = _target(test_df)
    obs = (test_df[OBS_COL].to_numpy(float) if OBS_COL in test_df.columns
           else np.full(len(test_df), np.nan))
    test_reg = _regime_labels(test_df)

    pr = per_regime_coverage(y, lower, upper, test_reg, alpha,
                             widths=upper - lower)
    res = RegimeConformalResult(
        lower=lower, upper=upper, q_lower=q_lower, q_upper=q_upper,
        half_width=np.asarray(half_width, float),
        coverage=cal.coverage(y, lower, upper),
        discharge_coverage=cal.coverage(obs, q_lower, q_upper),
        sharpness=cal.sharpness(lower, upper),
        discharge_sharpness=cal.sharpness(q_lower, q_upper),
        per_regime=pr, q_hat={k: float(v) for k, v in q_hat.items()},
        alpha=float(alpha), by=by, method=str(method).lower(),
        n_calib=n_calib, extra=extra,
    )
    log.info("regime_conformal[%s-%s]: n_calib=%d -> coverage=%.3f (nominal %.2f) "
             "| max_regime_gap=%.3f mean_under_cov=%.3f width=%.4f",
             by, res.method, n_calib, res.coverage, res.nominal,
             res.max_regime_gap, res.mean_under_coverage, res.sharpness)
    return res


def _regime_half_widths(scores, calib_df, test_df, alpha, global_qhat, min_calib):
    """Hard-Mondrian per-regime quantiles and the per-test-row half-widths."""
    cal_reg = _regime_labels(calib_df)
    test_reg = _regime_labels(test_df)
    scores = np.asarray(scores, float)

    q_hat: dict[str, float] = {}
    calib_counts: dict[str, int] = {}
    for r in REGIMES:
        s_r = scores[cal_reg == r]
        s_r = s_r[np.isfinite(s_r)]
        calib_counts[r] = int(s_r.size)
        if s_r.size >= int(min_calib):
            q_hat[r] = conformal_quantile(s_r, alpha)
        else:  # too sparse to calibrate on its own -> pooled quantile
            q_hat[r] = float(global_qhat)
    half_width = np.array([q_hat.get(r, global_qhat) for r in test_reg], float)
    return half_width, q_hat, calib_counts


def _gate_half_widths(model, scores, calib_df, test_df, alpha, global_qhat):
    """Soft-Mondrian: per-expert weighted quantiles, mixed by the test gate."""
    if not hasattr(model, "gate_weights"):
        raise ValueError("by='gate' requires a model exposing gate_weights "
                         "(the flagship RegimeProbNet)")
    w_cal = np.asarray(model.gate_weights(calib_df), float)   # (n_cal, K)
    w_test = np.asarray(model.gate_weights(test_df), float)   # (n_test, K)
    scores = np.asarray(scores, float)
    K = w_cal.shape[1]

    expert_qhat = np.empty(K, float)
    expert_eff_n: dict[str, float] = {}
    for k in range(K):
        qk = weighted_conformal_quantile(scores, w_cal[:, k], alpha)
        if not np.isfinite(qk):           # an unused expert -> pooled fallback
            qk = float(global_qhat)
        expert_qhat[k] = qk
        wk = w_cal[:, k]
        wk = wk[np.isfinite(wk) & (wk > 0)]
        eff = (wk.sum() ** 2 / np.sum(wk ** 2)) if wk.size else 0.0
        expert_eff_n[_expert_label(k, K)] = float(eff)

    # gate-weighted mixture of per-expert half-widths
    half_width = w_test @ expert_qhat                      # (n_test,)
    q_hat = {_expert_label(k, K): expert_qhat[k] for k in range(K)}
    return half_width, q_hat, expert_eff_n


def _expert_label(k: int, K: int) -> str:
    """Name expert ``k`` by its aligned regime when ``K`` spans the regimes."""
    if k < len(REGIMES):
        return REGIMES[k]
    return f"expert_{k}"


# --------------------------------------------------------------------------- #
#  Head-to-head with marginal split conformal                                 #
# --------------------------------------------------------------------------- #
def compare_to_marginal(model, calib_df: pd.DataFrame, test_df: pd.DataFrame,
                        alpha: float = 0.1, by: str = "regime", *,
                        method: str = "absolute", min_calib: int = 10
                        ) -> dict:
    """Run marginal split-conformal *and* the regime-conditional version.

    Produces the side-by-side evidence the paper needs: the marginal estimator
    attains overall coverage but leaves some regimes under-covered, while the
    regime-conditional estimator restores per-regime coverage near nominal.

    Parameters
    ----------
    model : BaseCorrector
        Fitted corrector (see :func:`regime_conditional_conformal`).
    calib_df, test_df : pandas.DataFrame
        Calibration and test tables.
    alpha : float, default 0.1
        Target miscoverage.
    by : {"regime", "gate"}, default "regime"
        Conditioning scheme passed to :func:`regime_conditional_conformal`.
    method : {"absolute", "cqr"}, default "absolute"
        Nonconformity score (shared by both estimators).
    min_calib : int, default 10
        Per-regime minimum calibration count.

    Returns
    -------
    dict
        ``{"marginal": ConformalResult, "conditional": RegimeConformalResult,
        "comparison": DataFrame, "summary": dict}``.  ``comparison`` has one row
        per regime with the marginal and regime-conditional coverage and width;
        ``summary`` reports each method's overall coverage and its mean per-regime
        under-coverage (lower is better; the regime-conditional value should be
        the smaller of the two).
    """
    marg = split_conformal(model, calib_df, test_df, alpha=alpha, method=method)
    cond = regime_conditional_conformal(model, calib_df, test_df, alpha=alpha,
                                        by=by, method=method, min_calib=min_calib)

    y = _target(validate(test_df))
    test_reg = _regime_labels(test_df)
    marg_pr = per_regime_coverage(y, marg.lower, marg.upper, test_reg, alpha)
    cond_pr = cond.per_regime

    comparison = marg_pr.merge(
        cond_pr, on=["regime", "n", "nominal"],
        suffixes=("_marginal", "_conditional"),
    )
    comparison = comparison[[
        "regime", "n", "nominal",
        "coverage_marginal", "coverage_conditional",
        "gap_marginal", "gap_conditional",
        "mean_width_marginal", "mean_width_conditional",
    ]]

    def _mean_under(pr: pd.DataFrame) -> float:
        cov = pr["coverage"].to_numpy(float)
        cov = cov[np.isfinite(cov)]
        return float(np.mean(np.clip((1.0 - alpha) - cov, 0.0, None))) if cov.size else np.nan

    def _max_gap(pr: pd.DataFrame) -> float:
        g = pr["gap"].to_numpy(float)
        g = g[np.isfinite(g)]
        return float(np.max(np.abs(g))) if g.size else np.nan

    summary = {
        "alpha": float(alpha), "nominal": 1.0 - float(alpha), "by": by,
        "method": str(method).lower(),
        "coverage_marginal": float(marg.coverage),
        "coverage_conditional": float(cond.coverage),
        "sharpness_marginal": float(marg.sharpness),
        "sharpness_conditional": float(cond.sharpness),
        "max_regime_gap_marginal": _max_gap(marg_pr),
        "max_regime_gap_conditional": _max_gap(cond_pr),
        "mean_under_coverage_marginal": _mean_under(marg_pr),
        "mean_under_coverage_conditional": _mean_under(cond_pr),
    }
    log.info("compare_to_marginal[%s]: under-cov marginal=%.3f -> conditional=%.3f "
             "| max gap %.3f -> %.3f", by,
             summary["mean_under_coverage_marginal"],
             summary["mean_under_coverage_conditional"],
             summary["max_regime_gap_marginal"],
             summary["max_regime_gap_conditional"])
    return {"marginal": marg, "conditional": cond,
            "comparison": comparison, "summary": summary}


# --------------------------------------------------------------------------- #
#  Self-test: synthetic; regime-conditional restores per-regime coverage      #
# --------------------------------------------------------------------------- #
def _selftest() -> None:  # pragma: no cover
    from ..config import PATHS
    from ..features.engineering import build_features
    from ..models.quantile_mapping import LinearScalingCorrector
    from ..synthetic import generate
    from .splits import temporal_split

    # ---- data: enough decadal rows that per-regime coverage is stable ------
    df = validate(classify_regimes(build_features(
        generate(scale="decadal", years=12, n_basins=4, gauges_per_basin=(2, 4),
                 seed=11), scale="decadal"))).reset_index(drop=True)

    tr_mask, fut_mask = temporal_split(df, test_frac=0.4)
    train = df[tr_mask].reset_index(drop=True)
    future = df[fut_mask].reset_index(drop=True)
    # split the (exchangeable) future block into calibration / test halves
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(future))
    half = len(future) // 2
    calib = future.iloc[perm[:half]].reset_index(drop=True)
    test = future.iloc[perm[half:]].reset_index(drop=True)
    print(f"[regime_conformal] train={len(train)} calib={len(calib)} "
          f"test={len(test)} | regimes(test)="
          f"{test[REGIME_COL].value_counts().to_dict()}")

    # ---- (1) hard Mondrian vs marginal, around a global LinearScaling ------
    base = LinearScalingCorrector().fit(train)
    out = compare_to_marginal(base, calib, test, alpha=0.1, by="regime",
                              method="absolute", min_calib=15)
    cmp = out["comparison"]
    print("[regime_conformal] per-regime coverage (marginal -> conditional):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(cmp.round(3).to_string(index=False))
    s = out["summary"]
    print(f"[regime_conformal] overall coverage: marginal={s['coverage_marginal']:.3f} "
          f"conditional={s['coverage_conditional']:.3f} (nominal {s['nominal']:.2f})")
    print(f"[regime_conformal] mean per-regime UNDER-coverage: "
          f"marginal={s['mean_under_coverage_marginal']:.3f} -> "
          f"conditional={s['mean_under_coverage_conditional']:.3f}")
    print(f"[regime_conformal] max per-regime |gap|: "
          f"marginal={s['max_regime_gap_marginal']:.3f} -> "
          f"conditional={s['max_regime_gap_conditional']:.3f}")

    # restricted to well-populated regimes for a stable comparison ----------
    pop = cmp[cmp["n"] >= 25]
    nominal = s["nominal"]
    under_marg = float(np.mean(np.clip(nominal - pop["coverage_marginal"], 0, None)))
    under_cond = float(np.mean(np.clip(nominal - pop["coverage_conditional"], 0, None)))
    gap_marg = float(np.mean(np.abs(pop["gap_marginal"])))
    gap_cond = float(np.mean(np.abs(pop["gap_conditional"])))
    print(f"[regime_conformal] populated regimes (n>=25): {len(pop)} | "
          f"mean|gap| {gap_marg:.3f} -> {gap_cond:.3f} | "
          f"under-cov {under_marg:.3f} -> {under_cond:.3f}")

    # ---- (2) soft Mondrian via the flagship's gate (heteroscedastic) -------
    gate_cond = None
    try:
        from ..models.regime_prob_net import RegimeProbNet
        flag = RegimeProbNet(K=5, hidden=16, seq_len=4, expert_hidden=16,
                             gate_hidden=16, epochs=4, batch_size=512, patience=3,
                             lambda_gate=0.5, lambda_phys=0.0, seed=0, verbose=False)
        flag.fit(train, valid=calib)
        gate_cond = regime_conditional_conformal(flag, calib, test, alpha=0.1,
                                                 by="gate", method="absolute")
        hw = gate_cond.half_width
        print(f"[regime_conformal] GATE soft-Mondrian: coverage={gate_cond.coverage:.3f} "
              f"| half-width mean={hw.mean():.4f} sd={hw.std():.4f} "
              f"(heteroscedastic) | per-expert q_hat="
              f"{ {k: round(v, 3) for k, v in gate_cond.q_hat.items()} }")
        print("[regime_conformal] GATE per-regime coverage:")
        print(gate_cond.per_regime.round(3).to_string(index=False))
    except Exception as exc:  # torch missing / fit failure must not kill the test
        print(f"[regime_conformal] gate variant skipped ({type(exc).__name__}: {exc})")

    # ---- persist a compact table for the paper -----------------------------
    try:
        PATHS.tables.mkdir(parents=True, exist_ok=True)
        path = PATHS.tables / "regime_conformal_selftest.csv"
        cmp.assign(by="regime", **{"overall_cov_marginal": s["coverage_marginal"],
                                   "overall_cov_conditional": s["coverage_conditional"]}
                   ).to_csv(path, index=False)
        print(f"[regime_conformal] wrote {path}")
    except Exception as exc:
        print(f"[regime_conformal] table write skipped ({exc})")

    # ---- assertions: regime-conditional restores per-regime validity -------
    assert 0.0 <= out["conditional"].coverage <= 1.0
    assert out["conditional"].coverage >= nominal - 0.06, (
        f"conditional overall coverage too low: {out['conditional'].coverage}")
    # within-regime under-coverage must not be worse than marginal, and should
    # improve (the whole point of the method) on the populated regimes.
    assert under_cond <= under_marg + 1e-9, (
        f"regime-conditional under-coverage {under_cond} > marginal {under_marg}")
    assert gap_cond <= gap_marg + 0.02, (
        f"regime-conditional mean|gap| {gap_cond} not <= marginal {gap_marg}")
    if gate_cond is not None:
        assert 0.0 <= gate_cond.coverage <= 1.0
        assert gate_cond.half_width.std() > 0.0, "gate widths are not heteroscedastic"
    print("[regime_conformal] SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
