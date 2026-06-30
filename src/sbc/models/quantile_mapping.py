"""Classical statistical bias-correction baselines.

These are the textbook reference methods every hydrological bias-correction study
is benchmarked against.  Both operate directly on discharge (not on engineered
features) and therefore provide an honest, transferable floor that the
machine-learning correctors must beat to justify their added complexity.

* :class:`QuantileMappingCorrector` (``"qmap"``) -- *empirical quantile mapping*
  (a.k.a. the empirical CDF / quantile-quantile transform; Panofsky & Brier
  1968; Gudmundsson et al., 2012, HESS).  For each gauge the empirical CDF of the
  raw GloFAS series is matched onto the empirical CDF of the observations, so a
  simulated value is replaced by the observed value sharing its non-exceedance
  probability.  This corrects the *whole distribution* (volume, variability and
  flow-duration shape), not just the mean.

* :class:`LinearScalingCorrector` (``"scaling"``) -- *linear (multiplicative)
  scaling* (Lenderink et al., 2007; Teutschbein & Seibert, 2012, J. Hydrol.).
  Each gauge's flow is multiplied by the ratio of observed to simulated mean
  flow, computed per calendar month so the dominant seasonal (snowmelt-freshet)
  volume bias is removed while preserving the simulated timing.

Both methods are fitted **on the training split only** (strictly leakage-safe)
and predict the framework's common log-residual target.  For gauges absent from
the training set -- the prediction-in-ungauged-regions (PUR) / leave-one-basin-out
(LOBO) setting -- they fall back to a **pooled** transform built from every
training gauge, so the correctors degrade gracefully to ungauged mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COL_DATE, COL_GAUGE, EPS
from ..schemas import OBS_COL, SIM_COL
from ..utils import get_logger
from .base import BaseCorrector, register

log = get_logger(__name__)

# A quantile map is the pair (sim_quantiles, obs_quantiles); ``None`` means the
# gauge had too little / too degenerate data to fit a usable transform.
QMap = tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
#  Empirical quantile mapping                                                 #
# --------------------------------------------------------------------------- #
def _fit_quantile_map(sim: np.ndarray, obs: np.ndarray,
                      n_quantiles: int, min_samples: int) -> QMap | None:
    """Fit a sim->obs empirical quantile map.

    Parameters
    ----------
    sim, obs : np.ndarray
        Paired raw-simulated and observed values (any non-finite entries are
        dropped independently before the marginal CDFs are estimated).
    n_quantiles : int
        Number of equally spaced probability levels used to discretise the two
        empirical CDFs.
    min_samples : int
        Minimum finite samples required of *each* series to fit a map.

    Returns
    -------
    tuple of np.ndarray or None
        ``(sim_q, obs_q)`` strictly increasing in ``sim_q`` and ready for
        :func:`numpy.interp`, or ``None`` when the data are insufficient or the
        simulated series is degenerate (a single distinct value).
    """
    sim = np.asarray(sim, float)
    obs = np.asarray(obs, float)
    sim = sim[np.isfinite(sim)]
    obs = obs[np.isfinite(obs)]
    if sim.size < min_samples or obs.size < min_samples:
        return None

    n = int(min(n_quantiles, sim.size, obs.size))
    n = max(n, 2)
    p = np.linspace(0.0, 1.0, n)
    sim_q = np.quantile(sim, p)
    obs_q = np.quantile(obs, p)

    # Collapse ties so ``sim_q`` is strictly increasing (required by np.interp);
    # the observed quantile at the first occurrence of each simulated level is
    # kept.  A perfectly flat simulated CDF (one unique value) cannot be mapped.
    sim_q, idx = np.unique(sim_q, return_index=True)
    if sim_q.size < 2:
        return None
    obs_q = obs_q[idx]
    return sim_q, obs_q


def _apply_quantile_map(x: np.ndarray, qmap: QMap) -> np.ndarray:
    """Map raw values through a fitted quantile map.

    Linear interpolation between stored quantiles; values outside the training
    support are clipped to the observed end-points (np.interp's clamped
    extrapolation), which prevents unphysical extrapolation of the correction.
    Non-finite inputs propagate as NaN.
    """
    sim_q, obs_q = qmap
    return np.interp(np.asarray(x, float), sim_q, obs_q)


@register
class QuantileMappingCorrector(BaseCorrector):
    """Per-gauge empirical quantile mapping with a pooled ungauged fallback.

    Parameters
    ----------
    n_quantiles : int, default 100
        Number of probability levels discretising each empirical CDF.
    min_samples : int, default 12
        Minimum training samples a gauge needs for its own map; otherwise the
        pooled map is used for that gauge.

    Notes
    -----
    The correction is distribution-wide and fitted on the training split only.
    :meth:`predict_residual` returns ``log(q_mapped + EPS) - log(q_glofas + EPS)``
    so it plugs into the framework's common log-residual target.
    """

    name = "qmap"

    def __init__(self, n_quantiles: int = 100, min_samples: int = 12) -> None:
        self.n_quantiles = int(n_quantiles)
        self.min_samples = int(min_samples)
        self.maps_: dict[str, QMap] = {}
        self.pooled_: QMap | None = None
        self._fitted = False

    # -- fit ---------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "QuantileMappingCorrector":
        """Fit one empirical quantile map per gauge plus a pooled fallback."""
        self.maps_ = {}
        sim_pool: list[np.ndarray] = []
        obs_pool: list[np.ndarray] = []
        for code, g in train.groupby(COL_GAUGE):
            sim = g[SIM_COL].to_numpy(float)
            obs = g[OBS_COL].to_numpy(float)
            sim_pool.append(sim)
            obs_pool.append(obs)
            qmap = _fit_quantile_map(sim, obs, self.n_quantiles, self.min_samples)
            if qmap is not None:
                self.maps_[str(code)] = qmap

        if sim_pool:
            self.pooled_ = _fit_quantile_map(
                np.concatenate(sim_pool), np.concatenate(obs_pool),
                self.n_quantiles, self.min_samples,
            )
        else:
            self.pooled_ = None

        self._fitted = True
        log.info("qmap fitted: %d per-gauge maps, pooled=%s",
                 len(self.maps_), self.pooled_ is not None)
        return self

    # -- predict ------------------------------------------------------------ #
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-residual; unseen gauges use the pooled map."""
        if not self._fitted:
            raise RuntimeError("QuantileMappingCorrector.fit must be called first")

        sim = df[SIM_COL].to_numpy(float)
        codes = df[COL_GAUGE].astype(str).to_numpy()
        q_mapped = np.array(sim, dtype=float, copy=True)  # identity default
        for code in pd.unique(codes):
            idx = np.flatnonzero(codes == code)
            qmap = self.maps_.get(code, self.pooled_)
            if qmap is not None:
                q_mapped[idx] = _apply_quantile_map(sim[idx], qmap)

        return (np.log(np.clip(q_mapped, 0.0, None) + EPS)
                - np.log(np.clip(sim, 0.0, None) + EPS))


# --------------------------------------------------------------------------- #
#  Linear (multiplicative) scaling                                            #
# --------------------------------------------------------------------------- #
def _scaling_factor(obs_sum: float, sim_sum: float) -> float:
    """Volume-preserving multiplicative factor obs/sim, guarded for tiny sim."""
    if not np.isfinite(obs_sum) or not np.isfinite(sim_sum) or sim_sum <= EPS:
        return 1.0
    return float(obs_sum / sim_sum)


@register
class LinearScalingCorrector(BaseCorrector):
    """Per-gauge multiplicative mean-bias correction (linear scaling).

    Parameters
    ----------
    seasonal : bool, default True
        If ``True`` a separate factor is estimated for each calendar month so the
        seasonal volume bias (notably the snowmelt freshet) is corrected; if
        ``False`` a single annual factor per gauge is used.
    min_samples : int, default 4
        Minimum training samples a (gauge, season) cell needs before its own
        factor is trusted; otherwise the pooled / global factor is used.

    Notes
    -----
    Each flow is multiplied by ``mean(q_obs) / mean(q_glofas)`` estimated on the
    training split.  Lookups cascade gauge-season -> pooled-season -> global so
    the corrector still applies to ungauged gauges and unseen seasons.
    """

    name = "scaling"

    def __init__(self, seasonal: bool = True, min_samples: int = 4) -> None:
        self.seasonal = bool(seasonal)
        self.min_samples = int(min_samples)
        self.factors_: dict[str, dict[int, float]] = {}
        self.pooled_factors_: dict[int, float] = {}
        self.global_factor_: float = 1.0
        self._fitted = False

    def _season(self, df: pd.DataFrame) -> np.ndarray:
        """Season key per row: calendar month (seasonal) or 0 (annual)."""
        if not self.seasonal:
            return np.zeros(len(df), dtype=int)
        return pd.to_datetime(df[COL_DATE]).dt.month.to_numpy()

    # -- fit ---------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "LinearScalingCorrector":
        """Estimate per-(gauge, season) factors with pooled/global fallbacks."""
        g = train[[COL_GAUGE, COL_DATE, OBS_COL, SIM_COL]].copy()
        g = g[np.isfinite(g[OBS_COL]) & np.isfinite(g[SIM_COL])]
        g["_season"] = self._season(g)

        self.factors_ = {}
        for (code, season), cell in g.groupby([COL_GAUGE, "_season"]):
            if len(cell) < self.min_samples:
                continue
            factor = _scaling_factor(cell[OBS_COL].sum(), cell[SIM_COL].sum())
            self.factors_.setdefault(str(code), {})[int(season)] = factor

        self.pooled_factors_ = {
            int(season): _scaling_factor(cell[OBS_COL].sum(), cell[SIM_COL].sum())
            for season, cell in g.groupby("_season")
        }
        self.global_factor_ = _scaling_factor(g[OBS_COL].sum(), g[SIM_COL].sum())

        self._fitted = True
        log.info("scaling fitted: %d gauges, %d seasonal cells, global=%.3f",
                 len(self.factors_), len(self.pooled_factors_), self.global_factor_)
        return self

    def _lookup(self, code: str, season: int) -> float:
        gauge = self.factors_.get(code)
        if gauge is not None and season in gauge:
            return gauge[season]
        if season in self.pooled_factors_:
            return self.pooled_factors_[season]
        return self.global_factor_

    # -- predict ------------------------------------------------------------ #
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-residual from the multiplicative scaling factors."""
        if not self._fitted:
            raise RuntimeError("LinearScalingCorrector.fit must be called first")

        sim = df[SIM_COL].to_numpy(float)
        codes = df[COL_GAUGE].astype(str).to_numpy()
        seasons = self._season(df)
        factors = np.array(
            [self._lookup(c, int(s)) for c, s in zip(codes, seasons)], dtype=float
        )
        q_corr = np.clip(sim, 0.0, None) * factors
        return np.log(q_corr + EPS) - np.log(np.clip(sim, 0.0, None) + EPS)


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
def _temporal_split(df: pd.DataFrame, train_frac: float = 0.75):
    """Split by a global date cutoff: earlier periods train, later periods test."""
    cutoff = df[COL_DATE].quantile(train_frac)
    train = df[df[COL_DATE] <= cutoff].copy()
    test = df[df[COL_DATE] > cutoff].copy()
    return train, test


def _mean_gauge_kge(df: pd.DataFrame, sim_col: str) -> float:
    """Mean per-gauge KGE' of ``sim_col`` against the observations."""
    from ..validation.metrics import kge_prime

    vals = [kge_prime(g[OBS_COL].values, g[sim_col].values)["kge"]
            for _, g in df.groupby(COL_GAUGE)]
    return float(np.nanmean(vals))


if __name__ == "__main__":
    from ..synthetic import generate

    table = generate(n_basins=4, years=14, seed=7)
    train, test = _temporal_split(table, train_frac=0.75)

    raw_kge = _mean_gauge_kge(test, SIM_COL)
    line = f"n_train={len(train)} n_test={len(test)} raw KGE'={raw_kge:.3f}"

    for corrector in (QuantileMappingCorrector(), LinearScalingCorrector()):
        corrector.fit(train)
        scored = test.copy()
        scored["q_pred"] = corrector.predict(test)
        kge_after = _mean_gauge_kge(scored, "q_pred")
        line += f" | {corrector.name} KGE'={kge_after:.3f}"
        assert np.isfinite(kge_after), f"{corrector.name} produced non-finite KGE'"

    # Sanity: empirical quantile mapping should improve over raw GloFAS.
    qmap = QuantileMappingCorrector().fit(train)
    scored = test.copy()
    scored["q_pred"] = qmap.predict(test)
    improved = _mean_gauge_kge(scored, "q_pred") > raw_kge
    print(line + f" | qmap improves={improved}")
