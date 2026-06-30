"""Probabilistic quantile baselines for the GloFAS log-residual.

The flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet` makes a strong
*uncertainty-quantification* (UQ) claim -- calibrated predictive distributions of
the bias-correction log-residual.  A claim is only credible against equally
principled competitors, so this module supplies the classical, well-evidenced
quantile-regression baselines every probabilistic-forecasting study is measured
against:

``QRFCorrector`` (registry name ``"qrf"``)
    *Quantile regression* on the log-residual.  By default it fits one
    :class:`sklearn.ensemble.GradientBoostingRegressor` with the pinball
    (``loss="quantile"``) objective per quantile level -- the textbook
    gradient-boosted quantile regressor -- and assembles their predictions into a
    monotone predictive quantile grid (quantile crossing is removed by per-row
    rearrangement; Chernozhukov et al., 2010).  An alternative
    ``method="rf"`` implements the leaf-sample *quantile regression forest*
    (Meinshausen, 2006): a single random forest whose conditional quantiles come
    from the empirical distribution of the training targets falling in the same
    leaves.  Either way the model exposes the full probabilistic API
    (:meth:`predict_quantiles`, :meth:`sample`, :meth:`predict_variance`) so its
    CRPS and interval coverage are evaluated by the same machinery as the
    flagship -- it is the QRF/quantile competitor to the Gaussian-mixture model.

``AsymLaplaceCorrector`` (registry name ``"alaplace"``)
    A light *asymmetric-Laplace* (two-piece exponential) quantile model: three
    pinball-loss boosters estimate a lower quantile, the median and an upper
    quantile, which parameterise a closed-form skewed predictive density.  It is
    a fast, smooth, analytically-sampleable distribution that captures the
    right-skew of snowmelt-freshet residuals without a deep network.

Both predict the common log-residual target; :func:`sbc.schemas.back_transform`
reconstructs corrected-discharge bands.  All heavy imports (scikit-learn) are
deferred into the fitting routines, the models accept a ``seed`` and are
deterministic, and feature columns are discovered with
:func:`sbc.schemas.feature_columns` (never hard-coded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import TARGET_COL, feature_columns, validate
from ..utils import get_logger
from .base import BaseCorrector, register

log = get_logger(__name__)

__all__ = ["QRFCorrector", "AsymLaplaceCorrector", "DEFAULT_QUANTILE_LEVELS"]

#: probability grid the boosters are trained on; arbitrary requested quantiles
#: are obtained by monotone interpolation of this grid.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
)


# --------------------------------------------------------------------------- #
#  Shared helpers                                                             #
# --------------------------------------------------------------------------- #
def _interp_cols(grid: np.ndarray, levels: np.ndarray, p: float) -> np.ndarray:
    """Linear interpolation of a per-row monotone quantile grid at level ``p``.

    Parameters
    ----------
    grid : numpy.ndarray, shape (n, G)
        Per-row predictive quantiles, non-decreasing along the columns and
        aligned to ``levels``.
    levels : numpy.ndarray, shape (G,)
        Ascending cumulative probabilities of the grid columns.
    p : float
        Target cumulative probability.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The interpolated quantile at ``p`` for every row (flat extrapolation at
        the grid endpoints).
    """
    if p <= levels[0]:
        return grid[:, 0]
    if p >= levels[-1]:
        return grid[:, -1]
    hi = int(np.searchsorted(levels, p))
    lo = hi - 1
    w = (p - levels[lo]) / (levels[hi] - levels[lo])
    return grid[:, lo] * (1.0 - w) + grid[:, hi] * w


# --------------------------------------------------------------------------- #
#  Quantile-regression corrector                                              #
# --------------------------------------------------------------------------- #
@register
class QRFCorrector(BaseCorrector):
    """Quantile-regression corrector for the log-residual.

    Parameters
    ----------
    method : {"gbr", "rf"}, default "gbr"
        ``"gbr"`` fits one pinball-loss
        :class:`sklearn.ensemble.GradientBoostingRegressor` per quantile level;
        ``"rf"`` fits a single :class:`sklearn.ensemble.RandomForestRegressor`
        and derives conditional quantiles from the training targets sharing each
        prediction's leaves (quantile regression forest, Meinshausen 2006).
    quantile_levels : sequence of float, optional
        Probability grid trained / stored (default :data:`DEFAULT_QUANTILE_LEVELS`).
        Requested quantiles are interpolated within this grid, so it should span
        the tails of every interval the analysis reports.
    n_estimators, learning_rate, max_depth, min_samples_leaf :
        Forwarded to the underlying scikit-learn estimator(s).  ``learning_rate``
        is used only by the ``"gbr"`` backend.
    n_jobs : int, default -1
        Parallelism for the ``"rf"`` backend.
    seed : int, default 0
        Random seed wired into the estimators (determinism).

    Attributes
    ----------
    features : list of str
        Feature columns discovered at fit time.
    quantile_levels : numpy.ndarray
        The sorted unique probability grid.
    """

    name: str = "qrf"
    is_probabilistic: bool = True

    def __init__(self, method: str = "gbr",
                 quantile_levels=DEFAULT_QUANTILE_LEVELS,
                 n_estimators: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 3, min_samples_leaf: int = 20,
                 n_jobs: int = -1, seed: int = 0) -> None:
        method = str(method).lower()
        if method not in ("gbr", "rf"):
            raise ValueError(f"method must be 'gbr' or 'rf', got {method!r}")
        self.method = method
        self.quantile_levels = np.unique(np.asarray(quantile_levels, float))
        if self.quantile_levels.size < 2:
            raise ValueError("need at least two distinct quantile levels")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)

        # learned state
        self.features: list[str] = []
        self._medians: np.ndarray | None = None
        self._models: dict[float, object] = {}      # gbr backend
        self.model = None                            # rf backend
        self._train_y: np.ndarray | None = None
        self._leaf_index: list[dict[int, np.ndarray]] = []
        self._fitted = False

    # -- sklearn-style introspection ---------------------------------------- #
    def get_params(self) -> dict:
        """Constructor kwargs (so the model can be cloned by an ensemble)."""
        return {"method": self.method,
                "quantile_levels": tuple(self.quantile_levels),
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "n_jobs": self.n_jobs, "seed": self.seed}

    # -- design matrix ------------------------------------------------------ #
    def _design(self, df: pd.DataFrame) -> np.ndarray:
        """Numeric feature matrix with median-imputed missing values."""
        X = df[self.features].to_numpy(float)
        if self._medians is not None:
            bad = ~np.isfinite(X)
            if bad.any():
                X = np.where(bad, np.broadcast_to(self._medians, X.shape), X)
        return X

    # -- fitting ------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "QRFCorrector":
        """Fit the quantile estimator(s) on the log-residual target."""
        train = validate(train)
        self.features = feature_columns(train)
        if not self.features:
            raise ValueError("no numeric feature columns found in training table")

        Xraw = train[self.features].to_numpy(float)
        med = np.nanmedian(np.where(np.isfinite(Xraw), Xraw, np.nan), axis=0)
        self._medians = np.nan_to_num(med, nan=0.0)
        y = train[TARGET_COL].to_numpy(float)
        ok = np.isfinite(y)
        X = self._design(train)[ok]
        y = y[ok]
        if y.size < 2:
            raise ValueError("not enough finite target rows to fit QRFCorrector")

        if self.method == "gbr":
            self._fit_gbr(X, y)
        else:
            self._fit_rf(X, y)
        self._fitted = True
        log.info("qrf fitted: method=%s, %d rows x %d features, %d quantile levels",
                 self.method, X.shape[0], len(self.features), self.quantile_levels.size)
        return self

    def _fit_gbr(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self._models = {}
        for q in self.quantile_levels:
            gbr = GradientBoostingRegressor(
                loss="quantile", alpha=float(q),
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
            )
            gbr.fit(X, y)
            self._models[float(q)] = gbr

    def _fit_rf(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=(self.max_depth or None),
            min_samples_leaf=self.min_samples_leaf, n_jobs=self.n_jobs,
            bootstrap=True, random_state=self.seed,
        )
        rf.fit(X, y)
        self.model = rf
        self._train_y = y
        leaves = rf.apply(X)                       # (n_train, T)
        self._leaf_index = []
        for t in range(leaves.shape[1]):
            col = leaves[:, t]
            order = np.argsort(col, kind="stable")
            col_sorted = col[order]
            bounds = np.flatnonzero(np.diff(col_sorted)) + 1
            groups = np.split(order, bounds)
            self._leaf_index.append(
                {int(col_sorted[g[0]]): g for g in groups})

    # -- predictive quantile grid ------------------------------------------- #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("QRFCorrector is not fitted; call fit() first")

    def _grid_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row predictive quantiles on the stored level grid, shape (n, G)."""
        self._check_fitted()
        X = self._design(df)
        if self.method == "gbr":
            grid = np.column_stack(
                [np.asarray(self._models[float(q)].predict(X), float)
                 for q in self.quantile_levels])
        else:
            grid = self._rf_grid(X)
        # remove quantile crossing by per-row rearrangement
        return np.sort(grid, axis=1)

    def _rf_grid(self, X: np.ndarray) -> np.ndarray:
        """Leaf-weighted empirical quantiles of the training targets, (n, G)."""
        leaves = self.model.apply(X)               # (n, T)
        n, T = leaves.shape
        levels = self.quantile_levels
        y = self._train_y
        order = np.argsort(y)
        y_sorted = y[order]
        out = np.empty((n, levels.size), float)
        for i in range(n):
            w = np.zeros(y.size, float)
            for t in range(T):
                members = self._leaf_index[t].get(int(leaves[i, t]))
                if members is not None and members.size:
                    w[members] += 1.0 / members.size
            sw = w.sum()
            if sw <= 0:
                out[i] = np.quantile(y, levels)
                continue
            cw = np.cumsum(w[order] / sw)
            out[i] = np.interp(levels, cw, y_sorted)
        return out

    # -- probabilistic API -------------------------------------------------- #
    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)
                          ) -> np.ndarray:
        """Predicted log-residual quantiles, shape ``(n, len(quantiles))``.

        Arbitrary requested probabilities are obtained by monotone linear
        interpolation of the trained quantile grid, so the columns are
        non-decreasing whenever the requested probabilities are sorted.
        """
        grid = self._grid_predict(df)
        req = np.atleast_1d(np.asarray(quantiles, float))
        out = np.empty((grid.shape[0], req.size), float)
        for j, p in enumerate(req):
            out[:, j] = _interp_cols(grid, self.quantile_levels, float(p))
        return out

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Median predicted log-residual, shape ``(n,)``."""
        grid = self._grid_predict(df)
        return _interp_cols(grid, self.quantile_levels, 0.5)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Gaussian-equivalent predictive variance from the central spread.

        ``sigma ~= (q_{0.8413} - q_{0.1587}) / 2`` (the +/-1 sigma inter-quantile
        range of a Gaussian); returned as ``sigma**2``.  Lets the quantile
        baseline plug into variance-consuming machinery (e.g. the deep ensemble).
        """
        grid = self._grid_predict(df)
        lo = _interp_cols(grid, self.quantile_levels, 0.158655)
        hi = _interp_cols(grid, self.quantile_levels, 0.841345)
        sigma = np.clip(0.5 * (hi - lo), 0.0, None)
        return sigma ** 2

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Inverse-CDF samples from the predictive quantile grid, ``(n_rows, n)``."""
        grid = self._grid_predict(df)
        rng = np.random.default_rng(seed)
        nr = grid.shape[0]
        u = rng.random((nr, n))
        out = np.empty((nr, n), float)
        lv = self.quantile_levels
        for i in range(nr):
            out[i] = np.interp(u[i], lv, grid[i])
        return out


# --------------------------------------------------------------------------- #
#  Asymmetric-Laplace (two-piece exponential) corrector                       #
# --------------------------------------------------------------------------- #
@register
class AsymLaplaceCorrector(BaseCorrector):
    """Light skewed predictive distribution from three pinball-loss boosters.

    The predictive law of the log-residual is a *two-piece exponential*
    (asymmetric Laplace) with location ``m`` (the conditional median) and
    separate left/right scales ``b_lo``/``b_hi``.  The scales are calibrated so
    that the model's lower quantile at ``p`` and upper quantile at ``1 - p`` --
    each estimated by a gradient-boosted pinball-loss regressor -- are matched
    exactly.  The resulting CDF is

    ``F(x) = 0.5 * exp((x - m) / b_lo)``           for ``x <= m``
    ``F(x) = 1 - 0.5 * exp(-(x - m) / b_hi)``       for ``x  > m``

    which inverts in closed form (fast, exact sampling) and captures the
    right-skew of snowmelt-freshet residuals without a deep network.

    Parameters
    ----------
    p : float, default 0.1
        Lower tail probability whose quantile (and its mirror ``1 - p``) anchors
        the two exponential scales; must lie in ``(0, 0.5)``.
    n_estimators, learning_rate, max_depth, min_samples_leaf, seed :
        Forwarded to the three :class:`GradientBoostingRegressor` quantile fits.
    """

    name: str = "alaplace"
    is_probabilistic: bool = True

    def __init__(self, p: float = 0.1, n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 3,
                 min_samples_leaf: int = 20, seed: int = 0) -> None:
        if not 0.0 < p < 0.5:
            raise ValueError(f"p must be in (0, 0.5), got {p}")
        self.p = float(p)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)

        self.features: list[str] = []
        self._medians: np.ndarray | None = None
        self._models: dict[str, object] = {}
        self._fitted = False

    def get_params(self) -> dict:
        return {"p": self.p, "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf, "seed": self.seed}

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(float)
        if self._medians is not None:
            bad = ~np.isfinite(X)
            if bad.any():
                X = np.where(bad, np.broadcast_to(self._medians, X.shape), X)
        return X

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "AsymLaplaceCorrector":
        """Fit the lower/median/upper pinball-loss boosters."""
        from sklearn.ensemble import GradientBoostingRegressor

        train = validate(train)
        self.features = feature_columns(train)
        if not self.features:
            raise ValueError("no numeric feature columns found in training table")
        Xraw = train[self.features].to_numpy(float)
        self._medians = np.nan_to_num(
            np.nanmedian(np.where(np.isfinite(Xraw), Xraw, np.nan), axis=0), nan=0.0)
        y = train[TARGET_COL].to_numpy(float)
        ok = np.isfinite(y)
        X, y = self._design(train)[ok], y[ok]
        if y.size < 2:
            raise ValueError("not enough finite target rows to fit AsymLaplaceCorrector")

        self._models = {}
        for tag, alpha in (("lo", self.p), ("med", 0.5), ("hi", 1.0 - self.p)):
            gbr = GradientBoostingRegressor(
                loss="quantile", alpha=float(alpha),
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
            )
            gbr.fit(X, y)
            self._models[tag] = gbr
        self._fitted = True
        log.info("alaplace fitted: %d rows x %d features (p=%.2f)",
                 X.shape[0], len(self.features), self.p)
        return self

    def _params(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row ``(m, b_lo, b_hi)`` of the two-piece exponential."""
        if not self._fitted:
            raise RuntimeError("AsymLaplaceCorrector is not fitted; call fit() first")
        X = self._design(df)
        m = np.asarray(self._models["med"].predict(X), float)
        q_lo = np.minimum(np.asarray(self._models["lo"].predict(X), float), m)
        q_hi = np.maximum(np.asarray(self._models["hi"].predict(X), float), m)
        log2p = np.log(2.0 * self.p)               # < 0 for p < 0.5
        b_lo = np.clip((q_lo - m) / log2p, 1e-6, None)     # (neg)/(neg) > 0
        b_hi = np.clip((q_hi - m) / (-log2p), 1e-6, None)
        return m, b_lo, b_hi

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Conditional median log-residual, shape ``(n,)``."""
        return self._params(df)[0]

    def _quantile(self, m, b_lo, b_hi, u: np.ndarray) -> np.ndarray:
        """Inverse CDF of the two-piece exponential at probabilities ``u``."""
        u = np.clip(u, 1e-9, 1.0 - 1e-9)
        left = m + b_lo * np.log(2.0 * u)
        right = m - b_hi * np.log(2.0 * (1.0 - u))
        return np.where(u < 0.5, left, right)

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)
                          ) -> np.ndarray:
        """Predicted log-residual quantiles, shape ``(n, len(quantiles))``."""
        m, b_lo, b_hi = self._params(df)
        req = np.atleast_1d(np.asarray(quantiles, float))
        return np.column_stack(
            [self._quantile(m, b_lo, b_hi, np.full_like(m, float(p))) for p in req])

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Closed-form variance of the two-piece exponential, shape ``(n,)``.

        With each half carrying probability 0.5, the mixture mean is
        ``m + 0.5 (b_hi - b_lo)`` and the second moment about ``m`` is
        ``b_lo**2 + b_hi**2`` (each side an exponential with that scale), giving
        ``var = b_lo**2 + b_hi**2 - 0.25 (b_hi - b_lo)**2``.
        """
        _, b_lo, b_hi = self._params(df)
        return b_lo ** 2 + b_hi ** 2 - 0.25 * (b_hi - b_lo) ** 2

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Exact inverse-CDF samples, shape ``(n_rows, n)``."""
        m, b_lo, b_hi = self._params(df)
        rng = np.random.default_rng(seed)
        u = rng.random((m.shape[0], n))
        return self._quantile(m[:, None], b_lo[:, None], b_hi[:, None], u)


# --------------------------------------------------------------------------- #
#  Self-test: small synthetic temporal split (fast GBR config)                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..schemas import OBS_COL, SIM_COL
    from ..synthetic import generate
    from ..validation.calibration import coverage as cov_fn
    from ..validation.calibration import sharpness
    from ..validation.metrics import crps_ensemble
    from ..validation.splits import temporal_split
    from .base import available

    df = validate(classify_regimes(build_features(
        generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3),
                 seed=11), scale="decadal"))).reset_index(drop=True)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    print(f"[probbase] registered: {[m for m in available() if m in ('qrf', 'alaplace')]}")
    print(f"[probbase] gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    fast = dict(n_estimators=120, learning_rate=0.08, max_depth=2, min_samples_leaf=15)
    y = test[TARGET_COL].to_numpy(float)

    # --- QRF (gradient-boosted quantile regression) ------------------------- #
    qrf = QRFCorrector(method="gbr", seed=0, **fast).fit(train)
    levels = np.round(np.linspace(0.05, 0.95, 19), 4)
    qgrid = qrf.predict_quantiles(test, tuple(levels))      # (n, 19) residual ens
    monotone = bool(np.all(np.diff(qgrid, axis=1) >= -1e-9))
    crps = crps_ensemble(y, qgrid)
    band = qrf.predict_quantiles(test, (0.05, 0.95))
    cov90 = cov_fn(y, band[:, 0], band[:, 1])
    width90 = sharpness(band[:, 0], band[:, 1])
    disc_q = qrf.predict_discharge_quantiles(test, (0.05, 0.5, 0.95))
    assert qgrid.shape == (len(test), 19) and monotone, "QRF quantiles not monotone"
    assert 0.0 <= cov90 <= 1.0 and np.isfinite(width90) and width90 > 0
    assert disc_q.shape == (len(test), 3)
    print(f"[probbase] QRF(gbr)  CRPS(resid)={crps:.4f} | cov90={cov90:.3f} "
          f"width90={width90:.3f} | monotone={monotone}")

    # quick RF backend smoke (tiny forest) ----------------------------------- #
    qrf_rf = QRFCorrector(method="rf", n_estimators=60, max_depth=6,
                          min_samples_leaf=10, seed=0).fit(train)
    rf_band = qrf_rf.predict_quantiles(test, (0.05, 0.95))
    rf_cov = cov_fn(y, rf_band[:, 0], rf_band[:, 1])
    assert 0.0 <= rf_cov <= 1.0
    print(f"[probbase] QRF(rf)   cov90={rf_cov:.3f} "
          f"CRPS={crps_ensemble(y, qrf_rf.predict_quantiles(test, tuple(levels))):.4f}")

    # --- Asymmetric-Laplace ------------------------------------------------- #
    ala = AsymLaplaceCorrector(seed=0, **fast).fit(train)
    a_grid = ala.predict_quantiles(test, tuple(levels))
    a_mono = bool(np.all(np.diff(a_grid, axis=1) >= -1e-9))
    a_band = ala.predict_quantiles(test, (0.05, 0.95))
    a_cov = cov_fn(y, a_band[:, 0], a_band[:, 1])
    a_var = ala.predict_variance(test)
    a_samp = ala.sample(test, n=32, seed=1)
    assert a_mono and a_samp.shape == (len(test), 32) and np.all(a_var >= 0)
    print(f"[probbase] ALaplace  CRPS(resid)={crps_ensemble(y, a_grid):.4f} | "
          f"cov90={a_cov:.3f} | var>=0={bool(np.all(a_var >= 0))} | monotone={a_mono}")

    print("[probbase] SELF-TEST OK")
