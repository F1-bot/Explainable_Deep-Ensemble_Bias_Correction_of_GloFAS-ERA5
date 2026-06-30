"""Regime-consistent stacked-ensemble meta-learner for residual bias correction.

Stacking (Wolpert, 1992) combines several base correctors by learning a small
*meta-learner* on their predictions.  The naive failure mode is information
leakage (meta trained on *in-sample* base predictions), but the failure that
actually crippled the earlier implementation of this class was the *opposite*
problem -- a **regime mismatch** between the data the meta was trained on and the
data it is applied to:

* the meta was trained on **leave-one-basin-out** out-of-fold (OOF) predictions,
  i.e. base predictions made in a *spatial-extrapolation* regime in which the
  per-gauge correctors fall back to pooled/global transforms and the boosters see
  none of the test basin -- so the OOF residual predictions are systematically
  *attenuated*;
* at inference the base models are refit on all of the training data and applied
  *in-region* (temporal holdout / same gauges), where their residual predictions
  have a much larger amplitude.

A least-squares meta fitted on the attenuated OOF matrix therefore learns
**inflated** weights (a single base could receive ``w ~ 1.45``) that *overshoot*
when multiplied into the larger in-region predictions -- which is exactly why the
old stacked model scored *below* its best member (temporal KGE' ~0.62 vs ~0.80).
A second, subtler issue is an **objective mismatch**: NNLS minimises residual MSE
and so prefers a shrunk (averaged) blend, but the reported metric is KGE', whose
variability term (``gamma``) is *hurt* by shrinkage -- a blend can have lower MSE
yet lower KGE' than its sharpest member.

This rewrite fixes both:

1.  **Regime-consistent meta features.**  Base correctors are fitted on an early
    temporal slice of the training record and used to predict a strictly later,
    *held-out* slice of the **same** gauges.  Those predictions are therefore
    produced in the same in-region regime as deployment, so the meta sees
    correctly-scaled inputs and learns sane, non-inflated weights.
2.  **A KGE'-validated safety net.**  On a third, even later held-out slice
    (untouched by the meta fit) we score -- *in discharge / KGE' space, the metric
    that matters* -- the learned meta blend, a constrained convex (equal-weight)
    average, and every single base.  Whichever wins is deployed.  The ensemble can
    therefore never do worse than its best single member by more than validation
    noise, and falls back to that member (or to the convex average) automatically
    when the learned blend does not help.

The learned weights (:pyattr:`StackedEnsemble.weights_`) are an interpretability
output: they describe the *deployed* blend (the learned meta weights when the meta
is kept, a one-hot vector when a single base is selected, or a uniform vector for
the convex fallback), so reading them always tells you what ``predict`` does.

Cloning assumption
------------------
Base correctors are cloned by :func:`copy.deepcopy` of the (unfitted) template,
which preserves *every* constructor hyper-parameter -- crucial because several
correctors in this package (e.g. :class:`RegimeProbNet`, the boosting family)
expose no ``get_params``; reconstructing them via ``type(m)()`` would silently
reset, say, ``epochs=3`` back to the default ``epochs=100``.  Re-fitting a deep
copy overwrites any learned state, which is what stacking requires.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..schemas import OBS_COL, SIM_COL, TARGET_COL, back_transform, validate
from ..utils import get_logger
from ..validation.metrics import kge_prime
from ..validation.splits import temporal_split
from .base import BaseCorrector, register

log = get_logger(__name__)

_VALID_META = ("nnls", "ridge", "catboost")

# strategy labels recorded on the fitted ensemble
_META, _CONVEX, _BASE = "meta", "convex", "base"


# --------------------------------------------------------------------------- #
#  Cloning & naming helpers                                                    #
# --------------------------------------------------------------------------- #
def _clone_corrector(model: BaseCorrector) -> BaseCorrector:
    """Return a fresh, unfitted copy of ``model`` (see module docstring).

    A deep copy of an *unfitted* template reproduces the template exactly --
    including hyper-parameters that are not recoverable through ``get_params`` --
    and re-fitting overwrites any learned state.

    Parameters
    ----------
    model : BaseCorrector
        Template estimator (typically unfitted; learned state, if any, is reset
        by the subsequent ``fit``).

    Returns
    -------
    BaseCorrector
        A new instance ready to be fitted from scratch.
    """
    try:
        return copy.deepcopy(model)
    except Exception:  # pragma: no cover - exotic un-deepcopyable estimator
        cls = type(model)
        if hasattr(model, "get_params"):
            try:
                return cls(**model.get_params())  # type: ignore[misc]
            except Exception:
                pass
        return cls()


def _unique_names(models: list[BaseCorrector]) -> list[str]:
    """Stable, de-duplicated display names for the base correctors."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, m in enumerate(models):
        base = getattr(m, "name", None) or f"base{i}"
        if base in seen:
            seen[base] += 1
            base = f"{base}#{seen[base]}"
        else:
            seen[base] = 0
        names.append(base)
    return names


# --------------------------------------------------------------------------- #
#  Stacked ensemble                                                           #
# --------------------------------------------------------------------------- #
@register
class StackedEnsemble(BaseCorrector):
    """Regime-consistent, KGE'-validated stacked ensemble of residual correctors.

    Parameters
    ----------
    base_models : list of BaseCorrector
        Unfitted base correctors to blend.  They are cloned, never mutated.
    meta : {"nnls", "ridge", "catboost"}, default "nnls"
        Meta-learner fitted on the (regime-consistent) base-prediction matrix.
        ``nnls`` gives interpretable non-negative weights; ``ridge`` is a
        regularised linear blend; ``catboost`` is a shallow non-linear stacker.
    seed : int, default 0
        Seed for any stochastic meta-learner.
    val_frac : float, default 0.3
        Per-gauge fraction of the latest training record held out as the
        **safety-net selection** slice (never used to fit the meta).
    blend_frac : float, default 0.3
        Per-gauge fraction of the *remaining* (development) record held out to
        build the **meta-training** matrix.  The earliest part trains the bases.
    meta_margin : float, default 0.02
        KGE' margin by which a blend (learned meta or convex average) must beat
        the best single base on the selection slice before it is deployed.  A
        positive margin makes the *best single base* the conservative default and
        only adopts a blend when it robustly helps, absorbing the finite-sample
        noise of the held-out selection estimate.
    n_folds, group_col : optional
        Accepted for backward compatibility (the old OOF-stacking knobs); they no
        longer affect fitting and are ignored.

    Attributes
    ----------
    weights_ : pandas.Series
        Per-base weights of the *deployed* blend, indexed by base name.
    base_names_ : list of str
        Names of the deployed (successfully fitted) base correctors.
    strategy_ : str
        Which predictor was deployed: ``"meta"``, ``"convex"`` or ``"base"``.
    selection_ : dict
        Safety-net KGE' scores on the selection slice for every candidate.
    """

    name = "stacked"
    is_probabilistic = False

    def __init__(self, base_models: list[BaseCorrector], meta: str = "nnls",
                 seed: int = 0, n_folds: int = 5, group_col: str = "basin",
                 val_frac: float = 0.3, blend_frac: float = 0.3,
                 meta_margin: float = 0.02) -> None:
        if not base_models:
            raise ValueError("StackedEnsemble needs at least one base model")
        if meta not in _VALID_META:
            raise ValueError(f"meta must be one of {_VALID_META}, got {meta!r}")
        self.base_models = list(base_models)
        self.meta = meta
        self.seed = int(seed)
        self.n_folds = int(n_folds)            # accepted for back-compat (unused)
        self.group_col = group_col             # accepted for back-compat (unused)
        self.val_frac = float(val_frac)
        self.blend_frac = float(blend_frac)
        self.meta_margin = float(meta_margin)

        # learned state
        self.weights_: pd.Series | None = None
        self.base_names_: list[str] = []
        self.strategy_: str | None = None
        self.selection_: dict[str, float] = {}
        self.n_folds_: int = 0
        self._deployed: list[tuple[str, BaseCorrector]] = []
        self._base_fill_: np.ndarray | None = None
        self._meta_kind: str | None = None
        self._meta_coef: np.ndarray | None = None
        self._meta_obj = None
        self._meta_weights_: pd.Series | None = None
        self._deploy_w_: np.ndarray | None = None

    # -- sklearn-style introspection (lets the ensemble itself be cloned) ----
    def get_params(self) -> dict:
        return {"base_models": self.base_models, "meta": self.meta,
                "seed": self.seed, "n_folds": self.n_folds,
                "group_col": self.group_col, "val_frac": self.val_frac,
                "blend_frac": self.blend_frac, "meta_margin": self.meta_margin}

    # -- internal temporal segmentation -------------------------------------
    def _segment(self, train: pd.DataFrame, valid: pd.DataFrame | None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into (fit, blend, sel) leakage-safe per-gauge temporal slices.

        ``fit`` trains the bases, ``blend`` trains the meta, ``sel`` selects the
        deployment strategy.  Chronology within each gauge is ``fit < blend <
        sel``.  An externally supplied ``valid`` is used directly as ``sel``.
        """
        if valid is not None and len(valid):
            sel_df = validate(valid).reset_index(drop=True)
            fm, bm = temporal_split(train, test_frac=self.blend_frac)
            fit_df = train[fm].reset_index(drop=True)
            blend_df = train[bm].reset_index(drop=True)
        else:
            dm, sm = temporal_split(train, test_frac=self.val_frac)
            dev = train[dm].reset_index(drop=True)
            sel_df = train[sm].reset_index(drop=True)
            fm, bm = temporal_split(dev, test_frac=self.blend_frac)
            fit_df = dev[fm].reset_index(drop=True)
            blend_df = dev[bm].reset_index(drop=True)

        # graceful degradation when a slice is empty (tiny / single-record data)
        if len(fit_df) == 0:
            fit_df = train
        if len(blend_df) == 0:
            blend_df = sel_df if len(sel_df) else fit_df
        if len(sel_df) == 0:
            sel_df = blend_df
        return fit_df, blend_df, sel_df

    # -- fitting ------------------------------------------------------------
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "StackedEnsemble":
        """Fit bases, fit the meta on regime-consistent predictions, then select."""
        train = validate(train).reset_index(drop=True)
        names = _unique_names(self.base_models)
        fit_df, blend_df, sel_df = self._segment(train, valid)

        # --- 1. fit each base twice: on the early slice (for the meta) and on
        #        all training data (for deployment); keep only those that succeed
        #        on BOTH so every downstream matrix shares one column order. -----
        kept_names: list[str] = []
        fit_models: list[BaseCorrector] = []
        full_models: list[BaseCorrector] = []
        for name, tmpl in zip(names, self.base_models):
            try:
                m_fit = _clone_corrector(tmpl)
                m_fit.fit(fit_df, None)
            except Exception as exc:
                log.warning("base '%s' failed on the meta-slice fit (%s); dropping",
                            name, exc)
                continue
            try:
                m_full = _clone_corrector(tmpl)
                m_full.fit(train, valid)
            except Exception as exc:
                log.warning("base '%s' failed on the full-train refit (%s); dropping",
                            name, exc)
                continue
            kept_names.append(name)
            fit_models.append(m_fit)
            full_models.append(m_full)

        if not kept_names:
            raise RuntimeError("every base model failed to fit")

        # --- 2. regime-consistent prediction matrices (bases trained on `fit`,
        #        predicting held-out future rows of the same gauges) ------------
        P_blend = self._raw_matrix(fit_models, blend_df)
        self._base_fill_ = self._column_fill(P_blend)
        P_blend = self._sanitise(P_blend, self._base_fill_)
        P_sel = self._sanitise(self._raw_matrix(fit_models, sel_df), self._base_fill_)
        y_blend = blend_df[TARGET_COL].to_numpy(float)

        # --- 3. fit the meta-learner on the blend matrix -----------------------
        good = np.isfinite(P_blend).all(axis=1) & np.isfinite(y_blend)
        if good.sum() >= P_blend.shape[1] + 1:
            self._fit_meta(P_blend[good], y_blend[good], kept_names)
        else:  # not enough rows to fit a meta -> degenerate to convex average
            log.warning("too few complete blend rows (%d); meta defaults to convex",
                        int(good.sum()))
            self._fit_meta_convex(kept_names)

        # --- 4. KGE'-validated safety net on the (held-out) selection slice ----
        strategy, deploy_w, scores = self._select_strategy(P_sel, sel_df, kept_names)

        # --- 5. record deployment state ----------------------------------------
        self._deployed = list(zip(kept_names, full_models))
        self.base_names_ = kept_names
        self.strategy_ = strategy
        self.selection_ = scores
        self.n_folds_ = len(kept_names)
        self._deploy_w_ = deploy_w
        self.weights_ = self._deployed_weights(strategy, deploy_w, kept_names)

        s = self.summary()
        log.info("stacked[%s]: %d/%d base(s) deployed | strategy=%s | "
                 "dominant=%s (w=%.3f) | sel KGE' meta=%.3f convex=%.3f best_base=%.3f",
                 self.meta, len(kept_names), len(self.base_models), strategy,
                 s["dominant"], s["dominant_weight"], scores.get("meta", np.nan),
                 scores.get("convex", np.nan), scores.get("best_base", np.nan))
        return self

    # -- prediction-matrix utilities ----------------------------------------
    def _raw_matrix(self, models: list[BaseCorrector], df: pd.DataFrame) -> np.ndarray:
        """Stack base log-residual predictions on ``df`` into an ``(n, m)`` array."""
        n = len(df)
        cols: list[np.ndarray] = []
        for j, model in enumerate(models):
            try:
                p = np.asarray(model.predict_residual(df), float).ravel()
                if p.shape[0] != n:
                    raise ValueError("prediction length mismatch")
            except Exception as exc:
                log.warning("base #%d failed at predict (%s); using fill", j, exc)
                p = np.full(n, np.nan)
            cols.append(p)
        return np.column_stack(cols) if cols else np.empty((n, 0))

    @staticmethod
    def _column_fill(P: np.ndarray) -> np.ndarray:
        """Per-column fill value (finite mean, 0 if a column is all non-finite)."""
        if P.size == 0:
            return np.zeros(P.shape[1])
        fill = np.nanmean(np.where(np.isfinite(P), P, np.nan), axis=0)
        return np.nan_to_num(fill, nan=0.0)

    @staticmethod
    def _sanitise(P: np.ndarray, fill: np.ndarray) -> np.ndarray:
        """Replace non-finite entries column-wise with ``fill``."""
        P = np.array(P, float, copy=True)
        for j in range(P.shape[1]):
            bad = ~np.isfinite(P[:, j])
            if bad.any():
                P[bad, j] = fill[j]
        return P

    # -- meta-learner -------------------------------------------------------
    def _fit_meta(self, P: np.ndarray, y: np.ndarray, names: list[str]) -> None:
        """Fit the chosen meta-learner and record interpretable learned weights."""
        self._meta_kind = self.meta
        if self.meta == "nnls":
            from scipy.optimize import nnls

            w, _ = nnls(P, y)
            w = np.asarray(w, float)
            if not np.isfinite(w).all() or w.sum() <= 0:  # degenerate -> convex
                w = np.full(P.shape[1], 1.0 / max(P.shape[1], 1))
            self._meta_coef = w
            weights = w
        elif self.meta == "ridge":
            from sklearn.linear_model import Ridge

            m = Ridge(alpha=1.0, random_state=self.seed)
            m.fit(P, y)
            self._meta_obj = m
            weights = np.asarray(m.coef_, float)
        else:  # catboost
            from catboost import CatBoostRegressor

            m = CatBoostRegressor(depth=3, iterations=300, learning_rate=0.05,
                                  loss_function="RMSE", random_seed=self.seed,
                                  allow_writing_files=False, verbose=False)
            m.fit(P, y)
            self._meta_obj = m
            fi = np.asarray(m.get_feature_importance(), float)
            tot = fi.sum()
            weights = fi / tot if tot > 0 else fi
        self._meta_weights_ = pd.Series(weights, index=names, name="weight")

    def _fit_meta_convex(self, names: list[str]) -> None:
        """Fallback meta == equal-weight convex average (no data to fit)."""
        m = len(names)
        self._meta_kind = "nnls"
        self._meta_coef = np.full(m, 1.0 / max(m, 1))
        self._meta_weights_ = pd.Series(self._meta_coef, index=names, name="weight")

    def _apply_meta(self, P: np.ndarray) -> np.ndarray:
        """Apply the *learned meta* to a base-prediction matrix."""
        if self._meta_kind == "nnls":
            return P @ self._meta_coef
        return np.asarray(self._meta_obj.predict(P), float)

    # -- safety-net strategy selection --------------------------------------
    def _select_strategy(self, P_sel: np.ndarray, sel_df: pd.DataFrame,
                         names: list[str]) -> tuple[str, np.ndarray, dict]:
        """Pick meta / convex / best-base by KGE' on the held-out selection slice.

        The comparison is made in *discharge* (KGE') space -- the reported metric
        -- because the meta is trained to minimise residual MSE, which does not
        guarantee a higher KGE' than the sharpest single member.
        """
        m = len(names)
        q_glofas = sel_df[SIM_COL].to_numpy(float)
        q_obs = sel_df[OBS_COL].to_numpy(float)

        def kge_of(resid: np.ndarray) -> float:
            k = kge_prime(q_obs, back_transform(q_glofas, resid))["kge"]
            return k if np.isfinite(k) else -np.inf

        uniform = np.full(m, 1.0 / max(m, 1))
        meta_kge = kge_of(self._apply_meta(P_sel))
        convex_kge = kge_of(P_sel @ uniform)
        base_kges = [kge_of(P_sel[:, j]) for j in range(m)]
        best_j = int(np.argmax(base_kges)) if base_kges else 0
        best_base_kge = base_kges[best_j] if base_kges else -np.inf

        scores = {"meta": meta_kge, "convex": convex_kge,
                  "best_base": best_base_kge, "best_base_name": names[best_j],
                  **{f"base[{names[j]}]": base_kges[j] for j in range(m)}}

        # The best single base is the conservative default: its KGE' ranking is
        # far more stable across temporal slices than that of a (variance-shrunk)
        # blend, so deploying it keeps the ensemble within noise of the best
        # member.  A blend is adopted only when it beats the best base on the
        # held-out slice by ``meta_margin`` -- enough to outrun the finite-sample
        # noise of a single selection window.  Among qualifying blends the higher
        # scorer wins (the learned meta is preferred on an exact tie).
        threshold = best_base_kge + self.meta_margin
        qualifying = [(s, k) for s, k in ((_META, meta_kge), (_CONVEX, convex_kge))
                      if np.isfinite(k) and k >= threshold]
        if qualifying:
            strategy = max(qualifying, key=lambda sk: sk[1])[0]
        else:
            strategy = _BASE

        if strategy == _META:
            deploy_w = (self._meta_coef if self._meta_kind == "nnls"
                        else None)  # ridge/catboost applied via _apply_meta
        elif strategy == _CONVEX:
            deploy_w = uniform
        else:  # single best base
            deploy_w = np.zeros(m)
            deploy_w[best_j] = 1.0
        return strategy, deploy_w, scores

    def _deployed_weights(self, strategy: str, deploy_w: np.ndarray | None,
                          names: list[str]) -> pd.Series:
        """Weights describing the *deployed* predictor (for interpretability)."""
        if strategy == _META:
            return self._meta_weights_.copy()
        return pd.Series(deploy_w, index=names, name="weight")

    # -- prediction ---------------------------------------------------------
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Blend the deployed base correctors' log-residual predictions on ``df``."""
        if not self._deployed or self.weights_ is None:
            raise RuntimeError("StackedEnsemble is not fitted; call fit() first")
        models = [m for _, m in self._deployed]
        P = self._sanitise(self._raw_matrix(models, df), self._base_fill_)
        if self.strategy_ == _META and self._meta_kind in ("ridge", "catboost"):
            return self._apply_meta(P)
        return P @ self._deploy_w_

    # -- interpretability ---------------------------------------------------
    def summary(self) -> dict:
        """Describe the learned blend: weights, normalised weights, dominant base."""
        if self.weights_ is None:
            raise RuntimeError("StackedEnsemble is not fitted; call fit() first")
        w = self.weights_
        total = float(w.sum())
        norm = (w / total) if total != 0 else w
        dominant = str(w.abs().idxmax())
        return {
            "meta": self.meta,
            "strategy": self.strategy_,
            "n_base": int(len(w)),
            "weights": {k: float(v) for k, v in w.items()},
            "weights_normalized": {k: float(v) for k, v in norm.items()},
            "weight_sum": total,
            "dominant": dominant,
            "dominant_weight": float(w[dominant]),
            "selection_kge": dict(self.selection_),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = f"fitted:{self.strategy_}" if self._deployed else "unfitted"
        return (f"StackedEnsemble(meta={self.meta!r}, n_base={len(self.base_models)}, "
                f"seed={self.seed}, {state})")


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from sbc.features.engineering import build_features
    from sbc.features.regimes import classify_regimes
    from sbc.models.boosting import LightGBMCorrector
    from sbc.models.quantile_mapping import LinearScalingCorrector
    from sbc.models.regime_prob_net import RegimeProbNet
    from sbc.synthetic import generate
    from sbc.validation.metrics import kge_prime
    from sbc.validation.splits import temporal_split

    # --- small synthetic decadal table -> features -> regimes --------------- #
    df = classify_regimes(build_features(
        generate(scale="decadal", years=8, n_basins=3, seed=11), "decadal"))
    df = df.reset_index(drop=True)
    trm, tem = temporal_split(df, test_frac=0.3)
    train, test = df[trm].reset_index(drop=True), df[tem].reset_index(drop=True)
    print(f"synthetic decadal: {df['code'].nunique()} gauges / "
          f"{df['basin'].nunique()} basins; train={len(train)} test={len(test)}")

    def _kge(q) -> float:
        return kge_prime(test["q_obs"].to_numpy(float), np.asarray(q, float))["kge"]

    bases = [LinearScalingCorrector(),
             LightGBMCorrector(n_optuna_trials=0),
             RegimeProbNet(epochs=3, seq_len=4, hidden=12, verbose=False)]
    ens = StackedEnsemble(bases, meta="nnls", seed=0).fit(train)

    print(f"deployed strategy : {ens.strategy_}")
    print("deployed weights_ :")
    for k, v in ens.weights_.items():
        print(f"   {k:14s} {v: .4f}")

    raw = _kge(test["q_glofas"].to_numpy(float))
    per_base = {nm: _kge(m.predict(test)) for nm, m in ens._deployed}
    ens_kge = _kge(ens.predict(test))
    best_base = max(per_base.values())

    base_str = " ".join(f"{nm}={v:+.3f}" for nm, v in per_base.items())
    print(f"KGE'  raw={raw:+.3f} | {base_str} | stacked={ens_kge:+.3f} "
          f"(best base={best_base:+.3f})")
    print(f"selection-slice KGE': "
          f"{ {k: round(v, 3) for k, v in ens.selection_.items() if isinstance(v, float)} }")

    assert ens_kge >= best_base - 0.01, (
        f"stacked {ens_kge:.3f} < best base {best_base:.3f} - 0.01")
    assert ens_kge > raw, f"stacked {ens_kge:.3f} did not beat raw GloFAS {raw:.3f}"
    print(f"SANITY OK: stacked ({ens_kge:+.3f}) >= best base ({best_base:+.3f}) "
          f"- 0.01 and beats raw ({raw:+.3f})")
    print("[ensemble] SELF-TEST OK")
