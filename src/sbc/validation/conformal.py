"""Conformal prediction intervals that wrap *any* corrector.

The flagship's uncertainty is calibrated *in-sample* but tends to under-cover on
the hardest split -- prediction in ungauged regions (PUR) -- because its
parametric predictive distribution has no finite-sample guarantee under
distribution shift.  *Conformal prediction* (Vovk et al., 2005; Lei et al., 2018)
is the principled, distribution-free fix: given a held-out **calibration** set it
turns any point (or quantile) corrector into prediction intervals with a
*finite-sample marginal coverage guarantee* ``>= 1 - alpha`` -- no assumption on
the model or the residual distribution beyond exchangeability of the calibration
and test points.

This module supplies two estimators, both operating on the framework's common
log-residual target and back-transforming to corrected-discharge bands:

* :func:`split_conformal` -- *split (inductive) conformal* regression.  Scores the
  calibration residuals of an already-fitted corrector, takes the conformal
  quantile (with the exact ``ceil((n+1)(1-alpha))``-th order-statistic
  correction), and inflates the test predictions into intervals of constant
  half-width (``"absolute"`` score) or, for probabilistic correctors, adaptively
  widens/narrows the model's own quantiles (``"cqr"``: conformalized quantile
  regression, Romano et al., 2019).  Marginal coverage ``>= 1 - alpha`` whenever
  calibration and test are exchangeable.

* :func:`enbpi_intervals` -- an *EnbPI*-style variant (Xu & Xie, 2021) for
  **time-series** residuals, which are *not* exchangeable.  It bootstraps an
  ensemble of correctors, scores each training point on the members that did not
  see it (out-of-bag, leakage-free), and forms test intervals from a *sliding
  window* of the most recent residuals that is updated online as observations
  arrive -- so the half-width tracks non-stationary error and stays valid without
  a dedicated held-out calibration block.

Both return a :class:`ConformalResult` carrying the interval arrays plus empirical
coverage and sharpness (mean width), computed with
:mod:`sbc.validation.calibration`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..schemas import OBS_COL, SIM_COL, TARGET_COL, back_transform, make_target, validate
from ..utils import get_logger
from . import calibration as cal

log = get_logger(__name__)

__all__ = ["ConformalResult", "split_conformal", "enbpi_intervals",
           "conformal_quantile"]

#: type of a zero-argument corrector factory (e.g. the corrector class itself)
ModelFactory = Callable[[], "object"]


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class ConformalResult:
    """Conformal prediction intervals and their calibration diagnostics.

    Attributes
    ----------
    lower, upper : numpy.ndarray
        Per-row interval bounds on the **log-residual** target.
    q_lower, q_upper : numpy.ndarray
        The same intervals back-transformed to **discharge** ``[m3 s-1]``.
    coverage : float
        Empirical coverage of the log-residual intervals on the test set.
    discharge_coverage : float
        Empirical coverage of the discharge bands against ``q_obs``.
    sharpness : float
        Mean log-residual interval width (narrower is better given coverage).
    discharge_sharpness : float
        Mean discharge band width ``[m3 s-1]``.
    q_hat : float or numpy.ndarray
        The conformal half-width(s); a scalar for split conformal, a per-test-row
        vector for the online EnbPI variant.
    alpha : float
        Target miscoverage (nominal coverage ``1 - alpha``).
    method : str
        Estimator label (``"split-absolute"``, ``"split-cqr"`` or ``"enbpi"``).
    n_calib : int
        Number of calibration / out-of-bag scores used.
    """

    lower: np.ndarray
    upper: np.ndarray
    q_lower: np.ndarray
    q_upper: np.ndarray
    coverage: float
    discharge_coverage: float
    sharpness: float
    discharge_sharpness: float
    q_hat: float | np.ndarray
    alpha: float
    method: str
    n_calib: int
    extra: dict = field(default_factory=dict)

    @property
    def nominal(self) -> float:
        """Nominal coverage ``1 - alpha``."""
        return 1.0 - self.alpha

    def summary(self) -> dict[str, float]:
        """Scalar diagnostics bundle for the paper's tables."""
        qh = self.q_hat
        return {
            "method": self.method, "alpha": self.alpha, "nominal": self.nominal,
            "coverage": float(self.coverage),
            "discharge_coverage": float(self.discharge_coverage),
            "sharpness": float(self.sharpness),
            "discharge_sharpness": float(self.discharge_sharpness),
            "q_hat": float(np.mean(qh)) if np.ndim(qh) else float(qh),
            "n_calib": int(self.n_calib),
        }


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity ``scores``.

    Returns the ``k``-th smallest score with ``k = ceil((n + 1)(1 - alpha))`` --
    the standard split-conformal correction that guarantees marginal coverage
    ``>= 1 - alpha``.  ``+inf`` when no finite score is available (a degenerate,
    trivially-covering interval) and the largest score once ``k`` exceeds ``n``.

    Parameters
    ----------
    scores : array_like
        Nonconformity scores (non-finite entries are dropped).
    alpha : float
        Target miscoverage in ``(0, 1)``.

    Returns
    -------
    float
        The conformal quantile (interval half-width / adjustment).
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k >= n:
        return float(np.max(s))
    return float(np.sort(s)[k - 1])


def _target(df: pd.DataFrame) -> np.ndarray:
    """The log-residual target of ``df`` (computed if the column is absent)."""
    if TARGET_COL in df.columns:
        return df[TARGET_COL].to_numpy(float)
    return make_target(df[OBS_COL].to_numpy(float), df[SIM_COL].to_numpy(float))


def _bundle(lower: np.ndarray, upper: np.ndarray, test: pd.DataFrame,
            q_hat, alpha: float, method: str, n_calib: int,
            extra: dict | None = None) -> ConformalResult:
    """Back-transform to discharge and attach coverage / sharpness diagnostics."""
    sim = test[SIM_COL].to_numpy(float)
    q_lower = back_transform(sim, lower)
    q_upper = back_transform(sim, upper)
    y = _target(test)
    obs = test[OBS_COL].to_numpy(float) if OBS_COL in test.columns else np.full(len(test), np.nan)
    return ConformalResult(
        lower=lower, upper=upper, q_lower=q_lower, q_upper=q_upper,
        coverage=cal.coverage(y, lower, upper),
        discharge_coverage=cal.coverage(obs, q_lower, q_upper),
        sharpness=cal.sharpness(lower, upper),
        discharge_sharpness=cal.sharpness(q_lower, q_upper),
        q_hat=q_hat, alpha=float(alpha), method=method, n_calib=int(n_calib),
        extra=dict(extra or {}),
    )


# --------------------------------------------------------------------------- #
#  Split (inductive) conformal                                                #
# --------------------------------------------------------------------------- #
def split_conformal(model, calib_df: pd.DataFrame, test_df: pd.DataFrame,
                    alpha: float = 0.1, method: str = "absolute"
                    ) -> ConformalResult:
    """Split-conformal prediction intervals around an already-fitted corrector.

    The ``model`` must already be **fitted on data disjoint from**
    ``calib_df`` (the inductive-conformal requirement); ``calib_df`` is used only
    to score nonconformity and is never fitted on, so coverage is leakage-free.

    Parameters
    ----------
    model : BaseCorrector
        Any fitted corrector exposing :meth:`predict_residual` (and, for
        ``method="cqr"``, :meth:`predict_quantiles`).
    calib_df : pandas.DataFrame
        Calibration modelling table (exchangeable with ``test_df``).
    test_df : pandas.DataFrame
        Table to predict intervals for.
    alpha : float, default 0.1
        Target miscoverage; nominal coverage is ``1 - alpha``.
    method : {"absolute", "cqr"}, default "absolute"
        ``"absolute"`` uses the symmetric residual score
        ``|y - mu(x)|`` (constant half-width; works for *any* corrector).
        ``"cqr"`` is conformalized quantile regression -- it conformalizes the
        corrector's own ``alpha/2`` and ``1 - alpha/2`` quantiles, giving
        adaptive (heteroscedastic) widths (probabilistic correctors only).

    Returns
    -------
    ConformalResult
        Intervals on the log-residual and discharge with coverage / sharpness.
        Under exchangeability the log-residual coverage is ``>= 1 - alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    calib_df = validate(calib_df)
    test_df = validate(test_df)
    y_cal = _target(calib_df)

    method = str(method).lower()
    if method == "cqr":
        if not getattr(model, "is_probabilistic", False):
            raise ValueError("method='cqr' requires a probabilistic corrector "
                             "exposing predict_quantiles")
        lo_hi = (alpha / 2.0, 1.0 - alpha / 2.0)
        qc = np.asarray(model.predict_quantiles(calib_df, lo_hi), float)
        # CQR nonconformity: signed distance outside the predicted band
        scores = np.maximum(qc[:, 0] - y_cal, y_cal - qc[:, 1])
        q_hat = conformal_quantile(scores, alpha)
        qt = np.asarray(model.predict_quantiles(test_df, lo_hi), float)
        lower = qt[:, 0] - q_hat
        upper = qt[:, 1] + q_hat
        label = "split-cqr"
    elif method == "absolute":
        mu_cal = np.asarray(model.predict_residual(calib_df), float)
        scores = np.abs(y_cal - mu_cal)
        q_hat = conformal_quantile(scores, alpha)
        mu_test = np.asarray(model.predict_residual(test_df), float)
        lower = mu_test - q_hat
        upper = mu_test + q_hat
        label = "split-absolute"
    else:
        raise ValueError(f"method must be 'absolute' or 'cqr', got {method!r}")

    n_cal = int(np.isfinite(scores).sum())
    res = _bundle(lower, upper, test_df, q_hat, alpha, label, n_cal)
    log.info("split_conformal[%s]: n_calib=%d q_hat=%.4f -> coverage=%.3f "
             "(nominal %.2f), width=%.4f", label, n_cal,
             float(np.mean(q_hat)) if np.ndim(q_hat) else float(q_hat),
             res.coverage, 1.0 - alpha, res.sharpness)
    return res


# --------------------------------------------------------------------------- #
#  EnbPI (ensemble batch prediction intervals) for time series                #
# --------------------------------------------------------------------------- #
def enbpi_intervals(make_model: ModelFactory, train_df: pd.DataFrame,
                    test_df: pd.DataFrame, alpha: float = 0.1, *,
                    n_bootstrap: int = 25, window: int | None = None,
                    online: bool = True, seed: int = 0) -> ConformalResult:
    """EnbPI-style conformal intervals for non-exchangeable (time-series) residuals.

    Implements the ensemble-batch-prediction-interval scheme (Xu & Xie, 2021)
    without a held-out calibration block:

    1. Fit ``n_bootstrap`` correctors on bootstrap resamples of ``train_df``.
    2. Score every training row on the members that did *not* sample it
       (out-of-bag) -- a leakage-free leave-one-out residual.
    3. Aggregate the members' predictions on ``test_df`` (the ensemble point
       forecast) and form each test interval from the conformal quantile of a
       *sliding window* of the most recent absolute residuals, optionally updated
       **online** with the realised test residual as each observation arrives.

    The sliding window lets the half-width adapt to non-stationary error, which a
    single static calibration quantile cannot, making the method appropriate for
    serially-correlated streamflow residuals.

    Parameters
    ----------
    make_model : callable
        Zero-argument factory returning a *fresh, unfitted* corrector (the
        corrector class itself works, e.g. ``LinearScalingCorrector``).
    train_df, test_df : pandas.DataFrame
        Training (residual-scoring) and test modelling tables.  ``test_df`` is
        treated chronologically (sorted by ``date``) for the online update.
    alpha : float, default 0.1
        Target miscoverage; nominal coverage ``1 - alpha``.
    n_bootstrap : int, default 25
        Number of bootstrap ensemble members ``B``.
    window : int, optional
        Sliding-window length of recent residuals (defaults to the number of
        out-of-bag scores, i.e. a full-history window).
    online : bool, default True
        When ``True`` the realised test residual is appended (and the oldest
        dropped) after each test point, so later widths reflect recent error.
    seed : int, default 0
        Seed for the bootstrap resampling (determinism).

    Returns
    -------
    ConformalResult
        Intervals on the log-residual and discharge with per-row ``q_hat`` and
        coverage / sharpness diagnostics.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if int(n_bootstrap) < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")

    train_df = validate(train_df).reset_index(drop=True)
    # chronological order so the online residual update respects time
    test_df = validate(test_df).sort_values("date", kind="stable").reset_index(drop=True)

    n = len(train_df)
    y_train = _target(train_df)
    rng = np.random.default_rng(seed)

    member_train = np.full((n_bootstrap, n), np.nan)   # member preds on train rows
    member_test = []                                   # member preds on test rows
    in_bag = np.zeros((n_bootstrap, n), dtype=bool)
    n_ok = 0
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n, n)
        model = make_model()
        try:
            model.fit(train_df.iloc[idx])
        except Exception as exc:  # pragma: no cover - one bad resample must not kill the run
            log.warning("enbpi: bootstrap member %d failed to fit (%s); skipping", b, exc)
            continue
        in_bag[n_ok, np.unique(idx)] = True
        member_train[n_ok] = np.asarray(model.predict_residual(train_df), float)
        member_test.append(np.asarray(model.predict_residual(test_df), float))
        n_ok += 1
    if n_ok == 0:
        raise RuntimeError("enbpi: every bootstrap member failed to fit")
    member_train = member_train[:n_ok]
    in_bag = in_bag[:n_ok]
    member_test = np.asarray(member_test)              # (n_ok, n_test)

    # out-of-bag training prediction: average over members that did not sample i
    oob = np.where(in_bag, np.nan, member_train)
    with np.errstate(invalid="ignore"):
        oob_pred = np.nanmean(oob, axis=0)
    # rows in every bag fall back to the full ensemble mean
    full_mean = np.nanmean(member_train, axis=0)
    oob_pred = np.where(np.isfinite(oob_pred), oob_pred, full_mean)
    oob_resid = np.abs(y_train - oob_pred)
    oob_resid = oob_resid[np.isfinite(oob_resid)]
    if oob_resid.size == 0:
        raise RuntimeError("enbpi: no finite out-of-bag residuals")

    # test ensemble point forecast (mean over members)
    f_test = np.nanmean(member_test, axis=0)
    y_test = _target(test_df)

    w = int(window) if window else oob_resid.size
    w = max(1, w)
    win = list(oob_resid[-w:])                          # recent-history window
    n_test = len(test_df)
    q_hat = np.empty(n_test, float)
    lower = np.empty(n_test, float)
    upper = np.empty(n_test, float)
    for t in range(n_test):
        h = conformal_quantile(np.asarray(win), alpha)
        q_hat[t] = h
        lower[t] = f_test[t] - h
        upper[t] = f_test[t] + h
        if online and np.isfinite(y_test[t]) and np.isfinite(f_test[t]):
            win.append(abs(y_test[t] - f_test[t]))
            if len(win) > w:
                win.pop(0)

    res = _bundle(lower, upper, test_df, q_hat, alpha, "enbpi", oob_resid.size,
                  extra={"n_members": n_ok, "window": w, "online": bool(online)})
    log.info("enbpi: B=%d members, window=%d, mean q_hat=%.4f -> coverage=%.3f "
             "(nominal %.2f), width=%.4f", n_ok, w, float(q_hat.mean()),
             res.coverage, 1.0 - alpha, res.sharpness)
    return res


# --------------------------------------------------------------------------- #
#  Self-test: synthetic temporal split; wrap LinearScaling; coverage ~ 0.9    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.probabilistic_baselines import QRFCorrector
    from ..models.quantile_mapping import LinearScalingCorrector
    from ..synthetic import generate
    from .splits import temporal_split

    df = validate(classify_regimes(build_features(
        generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3),
                 seed=5), scale="decadal"))).reset_index(drop=True)

    # temporal split: train strictly precedes the held-out future block; the
    # future block is then randomly halved into exchangeable calib / test sets.
    tr_mask, fut_mask = temporal_split(df, test_frac=0.4)
    train = df[tr_mask].reset_index(drop=True)
    future = df[fut_mask].reset_index(drop=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(future))
    half = len(future) // 2
    calib = future.iloc[perm[:half]].reset_index(drop=True)
    test = future.iloc[perm[half:]].reset_index(drop=True)
    print(f"[conformal] train={len(train)} calib={len(calib)} test={len(test)}")

    # --- (1) split conformal around a fitted LinearScaling corrector -------- #
    base = LinearScalingCorrector().fit(train)
    sc = split_conformal(base, calib, test, alpha=0.1, method="absolute")
    print(f"[conformal] SPLIT(scaling) q_hat={sc.q_hat:.4f} | "
          f"resid_cov={sc.coverage:.3f} disc_cov={sc.discharge_coverage:.3f} | "
          f"width={sc.sharpness:.4f} disc_width={sc.discharge_sharpness:.3f}")
    assert 0.80 <= sc.coverage <= 1.0, f"split-conformal coverage off: {sc.coverage}"

    # --- (2) CQR around a fitted QRF (adaptive widths) ---------------------- #
    qrf = QRFCorrector(method="gbr", n_estimators=120, max_depth=2,
                       min_samples_leaf=15, seed=0).fit(train)
    cqr = split_conformal(qrf, calib, test, alpha=0.1, method="cqr")
    print(f"[conformal] CQR(qrf)     resid_cov={cqr.coverage:.3f} | "
          f"width={cqr.sharpness:.4f}")
    assert 0.78 <= cqr.coverage <= 1.0, f"CQR coverage off: {cqr.coverage}"

    # --- (3) EnbPI time-series intervals (bootstrap OOB, sliding window) ---- #
    enb = enbpi_intervals(LinearScalingCorrector, train, future, alpha=0.1,
                          n_bootstrap=20, seed=0)
    print(f"[conformal] ENBPI(scaling) members={enb.extra['n_members']} "
          f"window={enb.extra['window']} | resid_cov={enb.coverage:.3f} "
          f"disc_cov={enb.discharge_coverage:.3f} | "
          f"mean_q_hat={float(np.mean(enb.q_hat)):.4f} width={enb.sharpness:.4f}")
    assert 0.75 <= enb.coverage <= 1.0, f"EnbPI coverage off: {enb.coverage}"
    print(f"[conformal] summaries: split={sc.summary()}")
    print("[conformal] SELF-TEST OK")
