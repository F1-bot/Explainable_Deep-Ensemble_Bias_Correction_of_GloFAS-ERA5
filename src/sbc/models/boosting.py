"""Gradient-boosted decision-tree correctors for the GloFAS log-residual.

Gradient-boosted decision trees (GBDTs) are the work-horse of this study's
deterministic bias correction.  They are a natural fit for the snow-influenced
streamflow problem for three reasons:

* **Heterogeneous, partly-missing predictors.**  The modelling table mixes
  dynamic ERA5-Land forcings (temperature, snowmelt, SWE, ...) with static
  catchment attributes (elevation, glacier fraction, aridity, ...).  XGBoost,
  LightGBM and CatBoost all route missing values down a learned default
  branch, so *no imputation is performed* — gaps in reanalysis forcings carry
  information and are handled natively.
* **Non-linear, threshold-like hydrology.**  Degree-day melt onset, freezing
  thresholds and storage saturation are inherently piecewise; axis-aligned
  tree splits capture them without hand-crafted interactions.
* **Interpretability.**  The fitted booster is retained on :attr:`.model` so
  the analysis layer can run exact TreeSHAP attributions, and
  :meth:`feature_importance` exposes the split-based importances directly.

All three backends predict the **log-space residual**
``log(q_obs + EPS) - log(q_glofas + EPS)``; :func:`sbc.schemas.back_transform`
reconstructs the corrected discharge.  Hyper-parameters can optionally be tuned
with Optuna's TPE sampler under median pruning, where the objective is the
hydrologically meaningful **median per-gauge KGE'** of the back-transformed
discharge on a held-out validation split (rather than a generic regression
loss), so tuning rewards what the paper actually reports.

The same :class:`BoostingCorrector` drives every backend; thin registered
subclasses (``xgb``, ``lgbm``, ``catboost``) expose them through the model
registry in :mod:`sbc.models.base`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COL_GAUGE
from ..schemas import OBS_COL, SIM_COL, TARGET_COL, back_transform, feature_columns
from ..utils import get_logger
from ..validation.metrics import kge_prime
from .base import BaseCorrector, register

log = get_logger(__name__)

#: supported gradient-boosting backends
BACKENDS: tuple[str, ...] = ("xgb", "lgbm", "catboost")

#: early-stopping patience (boosting rounds) when a validation split is given
EARLY_STOPPING_ROUNDS: int = 50

#: number of staged checkpoints used for Optuna intermediate reporting/pruning
N_PRUNING_RUNGS: int = 4

#: default tail fraction winsorised off the predicted log-residual.  The
#: back-transform is multiplicative (``q = q_glofas * exp(residual)``), so an
#: unbounded residual can amplify discharge by ``exp`` of a large number.  By
#: default predictions are clipped to the central ``1 - 2*RESIDUAL_CLIP_QUANTILE``
#: of the *training* residual distribution — a standard robust safeguard that is
#: inert on well-behaved data but prevents unphysical blow-ups in extrapolation.
RESIDUAL_CLIP_QUANTILE: float = 0.05


# --------------------------------------------------------------------------- #
#  Backend defaults                                                           #
# --------------------------------------------------------------------------- #
def _default_params(backend: str, seed: int) -> dict:
    """Return sensible default hyper-parameters for ``backend``.

    Parameters
    ----------
    backend:
        One of :data:`BACKENDS`.
    seed:
        Random seed wired into the backend's own RNG for determinism.

    Returns
    -------
    dict
        Keyword arguments for the backend's scikit-learn-style regressor.
    """
    if backend == "lgbm":
        return dict(
            n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=-1,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    if backend == "xgb":
        return dict(
            n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=5.0,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            gamma=0.0, tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0,
        )
    if backend == "catboost":
        return dict(
            iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            random_strength=1.0, bagging_temperature=1.0,
            random_seed=seed, allow_writing_files=False, verbose=False,
        )
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


# --------------------------------------------------------------------------- #
#  Core corrector                                                             #
# --------------------------------------------------------------------------- #
class BoostingCorrector(BaseCorrector):
    """Gradient-boosting residual corrector with optional Optuna HPO.

    Parameters
    ----------
    backend:
        Which gradient-boosting library to use: ``'xgb'``, ``'lgbm'`` or
        ``'catboost'``.
    params:
        Optional overrides merged on top of the backend defaults
        (see :func:`_default_params`).
    n_optuna_trials:
        If ``> 0``, run Optuna TPE hyper-parameter optimisation for this many
        trials before the final fit.  ``0`` disables tuning.
    seed:
        Seed for the backend RNG and the Optuna sampler (reproducibility).
    clip_quantile:
        Tail fraction winsorised off the predicted log-residual to keep the
        multiplicative back-transform physical (see :data:`RESIDUAL_CLIP_QUANTILE`).
        Set to ``0`` to disable clipping.

    Attributes
    ----------
    model:
        The fitted raw booster (scikit-learn-style estimator).  Exposed for
        exact TreeSHAP attribution downstream.
    features:
        Ordered list of feature column names used at fit time.
    residual_bounds_:
        ``(low, high)`` clip bounds derived from the training residuals.
    best_params_:
        Hyper-parameters of the final model (defaults + overrides + any tuned
        values).
    """

    name: str = "boosting"
    is_probabilistic: bool = False

    def __init__(self, backend: str = "lgbm", params: dict | None = None,
                 n_optuna_trials: int = 0, seed: int = 0,
                 clip_quantile: float = RESIDUAL_CLIP_QUANTILE) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend: str = backend
        self.name = backend                       # registry / reporting label
        self.params: dict = dict(params or {})
        self.n_optuna_trials: int = int(n_optuna_trials)
        self.seed: int = int(seed)
        self.clip_quantile: float = float(clip_quantile)
        self.model = None
        self.features: list[str] = []
        self.residual_bounds_: tuple[float, float] = (-np.inf, np.inf)
        self.best_params_: dict = {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        fitted = "fitted" if self.model is not None else "unfitted"
        return (f"{type(self).__name__}(backend={self.backend!r}, "
                f"n_optuna_trials={self.n_optuna_trials}, {fitted})")

    # -- public API --------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "BoostingCorrector":
        """Fit the booster, optionally tuning hyper-parameters first.

        Parameters
        ----------
        train:
            Modelling table used for training.  Features are discovered with
            :func:`sbc.schemas.feature_columns`; the target is the
            ``log_residual`` column.
        valid:
            Optional held-out modelling table.  When given it is used for
            early stopping and, during HPO, as the validation split whose
            median per-gauge KGE' is maximised.

        Returns
        -------
        BoostingCorrector
            ``self`` (fitted).
        """
        self.features = feature_columns(train)
        if not self.features:
            raise ValueError("no numeric feature columns found in training table")
        X = train[self.features]
        y = train[TARGET_COL].to_numpy(float)
        self.residual_bounds_ = self._residual_bounds(y)

        if self.n_optuna_trials > 0:
            tuned = self._optuna_search(X, y, valid)
            self.params = {**self.params, **tuned}

        self.best_params_ = {**_default_params(self.backend, self.seed), **self.params}
        self.model = self._build_and_fit(X, y, self.params, valid)
        log.info("Fitted %s corrector on %d rows x %d features (trees=%d)",
                 self.backend, len(X), len(self.features), self._n_trees(self.model))
        return self

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the log-space residual for every row of ``df``.

        The raw booster prediction is winsorised to :attr:`residual_bounds_`
        (the training-residual clip range) so the multiplicative back-transform
        stays physical.  NaNs in the feature columns are passed through
        unchanged — the boosters handle missing values natively.
        """
        if self.model is None:
            raise RuntimeError("call fit() before predict_residual()")
        pred = np.asarray(self.model.predict(df[self.features]), dtype=float)
        return self._clip(pred)

    # -- residual safeguard ------------------------------------------------- #
    def _residual_bounds(self, y: np.ndarray) -> tuple[float, float]:
        """Clip bounds = ``[q, 1-q]`` quantiles of the training residuals."""
        q = self.clip_quantile
        if not (0.0 < q < 0.5):
            return (-np.inf, np.inf)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return (-np.inf, np.inf)
        lo, hi = np.quantile(finite, [q, 1.0 - q])
        return (float(lo), float(hi))

    def _clip(self, residual: np.ndarray) -> np.ndarray:
        lo, hi = self.residual_bounds_
        return np.clip(residual, lo, hi)

    def feature_importance(self) -> pd.Series:
        """Return split-based feature importances, descending.

        Returns
        -------
        pandas.Series
            Importance indexed by feature name.  For LightGBM/XGBoost these are
            the scikit-learn ``feature_importances_``; for CatBoost the
            ``PredictionValuesChange`` importances.
        """
        if self.model is None:
            raise RuntimeError("call fit() before feature_importance()")
        if self.backend == "catboost":
            imp = np.asarray(self.model.get_feature_importance(), dtype=float)
        else:
            imp = np.asarray(self.model.feature_importances_, dtype=float)
        return (pd.Series(imp, index=self.features, name=f"{self.name}_importance")
                .sort_values(ascending=False))

    # -- backend dispatch --------------------------------------------------- #
    def _build_and_fit(self, X: pd.DataFrame, y: np.ndarray, params: dict,
                       valid: pd.DataFrame | None):
        """Construct the chosen regressor and fit it (with early stopping)."""
        cfg = {**_default_params(self.backend, self.seed), **params}
        has_valid = valid is not None and len(valid) > 0
        eval_X = valid[self.features] if has_valid else None
        eval_y = valid[TARGET_COL].to_numpy(float) if has_valid else None

        if self.backend == "lgbm":
            import lightgbm as lgb

            model = lgb.LGBMRegressor(**cfg)
            if has_valid:
                model.fit(X, y, eval_set=[(eval_X, eval_y)],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                     lgb.log_evaluation(0)])
            else:
                model.fit(X, y)
            return model

        if self.backend == "xgb":
            import xgboost as xgb

            if has_valid:
                cfg = {**cfg, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}
            model = xgb.XGBRegressor(**cfg)
            if has_valid:
                model.fit(X, y, eval_set=[(eval_X, eval_y)], verbose=False)
            else:
                model.fit(X, y, verbose=False)
            return model

        # catboost
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**cfg)
        if has_valid:
            model.fit(X, y, eval_set=(eval_X, eval_y),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        else:
            model.fit(X, y, verbose=False)
        return model

    def _n_trees(self, model) -> int:
        """Number of trees actually used by ``model`` (respects early stopping)."""
        if self.backend == "lgbm":
            bi = getattr(model, "best_iteration_", None)
            return int(bi) if bi else int(model.n_estimators)
        if self.backend == "xgb":
            bi = getattr(model, "best_iteration", None)
            return int(bi) + 1 if bi is not None else int(model.n_estimators)
        return int(model.tree_count_)

    def _staged_predict(self, model, X: pd.DataFrame, n_trees: int) -> np.ndarray:
        """Predict using only the first ``n_trees`` boosting rounds."""
        if self.backend == "lgbm":
            return np.asarray(model.predict(X, num_iteration=n_trees), float)
        if self.backend == "xgb":
            return np.asarray(model.predict(X, iteration_range=(0, n_trees)), float)
        return np.asarray(model.predict(X, ntree_end=n_trees), float)

    # -- hyper-parameter optimisation -------------------------------------- #
    def _suggest(self, trial) -> dict:
        """Sample a backend-specific hyper-parameter set from ``trial``."""
        if self.backend == "lgbm":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
                learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            )
        if self.backend == "xgb":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
                learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            )
        # catboost
        return dict(
            iterations=trial.suggest_int("iterations", 200, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        )

    @staticmethod
    def _median_gauge_kge(codes: np.ndarray, obs: np.ndarray, sim: np.ndarray) -> float:
        """Median over gauges of the per-gauge KGE' (NaN-aware)."""
        frame = pd.DataFrame({"code": codes, "obs": obs, "sim": sim})
        scores = [kge_prime(g["obs"].to_numpy(), g["sim"].to_numpy())["kge"]
                  for _, g in frame.groupby("code")]
        scores = [s for s in scores if np.isfinite(s)]
        return float(np.median(scores)) if scores else -np.inf

    def _optuna_search(self, X: pd.DataFrame, y: np.ndarray,
                       valid: pd.DataFrame | None) -> dict:
        """Run TPE search with median pruning; return the best parameters.

        The validation objective is the *negated* median per-gauge KGE' of the
        back-transformed corrected discharge on ``valid``.  When ``valid`` is
        ``None`` it falls back to in-sample residual MSE (no pruning rungs).
        """
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        has_valid = valid is not None and len(valid) > 0
        if has_valid:
            v_obs = valid[OBS_COL].to_numpy(float)
            v_sim = valid[SIM_COL].to_numpy(float)
            v_codes = valid[COL_GAUGE].to_numpy()
            v_X = valid[self.features]

        def objective(trial) -> float:
            params = {**self.params, **self._suggest(trial)}
            model = self._build_and_fit(X, y, params, valid)
            if not has_valid:
                resid = np.asarray(model.predict(X), float) - y
                return float(np.mean(resid ** 2))
            n_trees = self._n_trees(model)
            rungs = max(1, min(N_PRUNING_RUNGS, n_trees))
            value = np.inf
            for step in range(rungs):
                k = max(1, int(round(n_trees * (step + 1) / rungs)))
                pred = self._clip(self._staged_predict(model, v_X, k))
                q_pred = back_transform(v_sim, pred)
                value = -self._median_gauge_kge(v_codes, v_obs, q_pred)
                trial.report(value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return value

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        log.info("Optuna(%s): best objective=%.4f over %d trials",
                 self.backend, study.best_value, len(study.trials))
        return dict(study.best_params)


# --------------------------------------------------------------------------- #
#  Registered backend subclasses                                              #
# --------------------------------------------------------------------------- #
@register
class XGBoostCorrector(BoostingCorrector):
    """XGBoost residual corrector (registry name ``'xgb'``)."""

    name = "xgb"

    def __init__(self, params: dict | None = None, n_optuna_trials: int = 0,
                 seed: int = 0, clip_quantile: float = RESIDUAL_CLIP_QUANTILE) -> None:
        super().__init__("xgb", params, n_optuna_trials, seed, clip_quantile)


@register
class LightGBMCorrector(BoostingCorrector):
    """LightGBM residual corrector (registry name ``'lgbm'``)."""

    name = "lgbm"

    def __init__(self, params: dict | None = None, n_optuna_trials: int = 0,
                 seed: int = 0, clip_quantile: float = RESIDUAL_CLIP_QUANTILE) -> None:
        super().__init__("lgbm", params, n_optuna_trials, seed, clip_quantile)


@register
class CatBoostCorrector(BoostingCorrector):
    """CatBoost residual corrector (registry name ``'catboost'``)."""

    name = "catboost"

    def __init__(self, params: dict | None = None, n_optuna_trials: int = 0,
                 seed: int = 0, clip_quantile: float = RESIDUAL_CLIP_QUANTILE) -> None:
        super().__init__("catboost", params, n_optuna_trials, seed, clip_quantile)


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
def _add_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append minimal seasonal features (self-test only fallback).

    Adds smooth day-of-year harmonics so the table always carries at least a
    couple of dynamic predictors even if upstream feature engineering modules
    are not yet implemented.
    """
    out = df.copy()
    doy = out["date"].dt.dayofyear.to_numpy(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def _median_per_gauge_kge(df: pd.DataFrame, sim: np.ndarray) -> float:
    """Median over gauges of per-gauge KGE' (the framework's headline metric)."""
    return BoostingCorrector._median_gauge_kge(
        df[COL_GAUGE].to_numpy(), df[OBS_COL].to_numpy(), np.asarray(sim, float))


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from ..schemas import validate
    from ..synthetic import generate
    from .base import available

    # tiny synthetic table -> log-residual target -> raw features -> split by date
    df = _add_raw_features(validate(generate(n_basins=3, years=8, seed=0)))
    cut_tr, cut_va = df["date"].quantile([0.6, 0.75]).to_numpy()
    train = df[df["date"] <= cut_tr]
    valid = df[(df["date"] > cut_tr) & (df["date"] <= cut_va)]
    test = df[df["date"] > cut_va]

    model = BoostingCorrector("lgbm", n_optuna_trials=0, seed=0).fit(train, valid)
    q_corr = model.predict(test)

    kge_raw = _median_per_gauge_kge(test, test[SIM_COL].to_numpy())
    kge_cor = _median_per_gauge_kge(test, q_corr)
    top = model.feature_importance().head(3).index.tolist()
    print(f"[boosting] registered backends = {available()}")
    print(f"[boosting] n_test={len(test)}  top_features={top}")
    print(f"[boosting] median per-gauge KGE'  raw={kge_raw:+.3f}  "
          f"corrected={kge_cor:+.3f}  improvement={kge_cor - kge_raw:+.3f}")
