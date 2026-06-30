"""SHAP / attribution analysis tying model behaviour to snow processes.

Bias correction is only trustworthy for a paper if it is *explainable*: a
reviewer must be able to see that the learned correction keys on physically
sensible drivers (snow-water-equivalent, snowmelt, snow-cover fraction, air
temperature and their engineered descriptors) rather than on spurious
correlations.  This module turns a fitted :class:`~sbc.models.base.BaseCorrector`
into reproducible attribution tables and publication-grade figures.

For gradient-boosted trees (``BoostingCorrector``) we use exact tree SHAP
(Lundberg et al., 2020, *Nature Machine Intelligence*) on the raw booster, which
gives consistent, locally-accurate Shapley values at low cost.  For the deep
models (EA-LSTM, regime-probability network) tree SHAP does not apply, so we
fall back to model-agnostic **permutation importance** on the log-residual
target — every model in the framework therefore remains explainable.

All returned objects are tidy ``pandas.DataFrame``s so the paper's figures and
tables are fully reproducible.  Plotting helpers are Agg-safe (no ``plt.show``)
and write PNGs to :pyattr:`sbc.config.Paths.shap_dir` (``results/shap``).

Heavy, optional dependencies (``shap``, ``matplotlib``, ``lightgbm``) are
imported lazily inside the functions that need them so that importing the
package stays light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import PATHS
from ..schemas import TARGET_COL, feature_columns, make_target
from ..utils import get_logger

log = get_logger(__name__)

#: substrings that flag a feature as snow-/cryosphere-/melt-related.  Matching is
#: case-insensitive and also covers engineered ``f_*`` descriptors that embed one
#: of these tokens (e.g. ``f_swe_amax``, ``f_melt_onset``, ``f_t2m_djf``).
SNOW_TOKENS: tuple[str, ...] = (
    "swe", "smlt", "melt", "scf", "snow", "snw", "sf",
    "t2m", "temp", "tas", "freez", "thaw", "ros", "rain_on_snow",
    "glac", "glacier", "ice",
)


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
def _model_features(model: Any, df: pd.DataFrame) -> list[str]:
    """Resolve the feature list a model was trained on.

    Prefers ``model.features`` (the contract every corrector exposes) and falls
    back to schema-driven discovery when that attribute is absent.
    """
    feats = getattr(model, "features", None)
    if feats is None:
        feats = feature_columns(df)
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise ValueError("no model features are present in the supplied table")
    return list(feats)


def _subsample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Deterministically subsample rows (keeps the original index intact)."""
    if max_samples is None or len(df) <= max_samples:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=int(max_samples), replace=False)
    idx.sort()
    return df.iloc[idx]


def _expected_value(raw: Any) -> float:
    """Coerce a (possibly array-valued) SHAP expected value to a float."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float).ravel()
        return float(arr[0]) if arr.size else float("nan")
    return float(raw)


def _resolve_path(path: str | Path | None, default_name: str) -> Path:
    """Return a writable PNG path, defaulting under ``PATHS.shap_dir``."""
    if path is None:
        PATHS.shap_dir.mkdir(parents=True, exist_ok=True)
        return PATHS.shap_dir / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def snow_features(features: list[str]) -> list[str]:
    """Subset ``features`` to the snow-/cryosphere-/melt-related ones.

    Parameters
    ----------
    features
        Candidate feature names.

    Returns
    -------
    list of str
        Features whose (lower-cased) name contains any :data:`SNOW_TOKENS`
        token, order-preserved.
    """
    out = []
    for f in features:
        low = str(f).lower()
        if any(tok in low for tok in SNOW_TOKENS):
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
#  Tree SHAP                                                                   #
# --------------------------------------------------------------------------- #
def tree_shap(model: Any, df: pd.DataFrame,
              max_samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Exact tree-SHAP attribution for a gradient-boosted corrector.

    Runs :class:`shap.TreeExplainer` on the model's *raw booster*
    (``model.model``) over ``X = df[model.features]``.  Works for the LightGBM /
    XGBoost / CatBoost boosters wrapped by ``BoostingCorrector``.

    Parameters
    ----------
    model
        A fitted corrector exposing ``model.model`` (the raw booster) and
        ``model.features`` (the training feature names).
    df
        Modelling table to explain.  At most ``max_samples`` rows are used.
    max_samples
        Cap on the number of rows passed to the explainer (kept deterministic
        via ``seed``).  ``None`` uses every row.
    seed
        Seed for the row subsample.

    Returns
    -------
    dict
        ``{"shap_values": (n, p) ndarray, "base_value": float,
        "features": list[str], "X": DataFrame (n, p)}``.
    """
    import shap

    booster = getattr(model, "model", None)
    if booster is None:
        raise AttributeError(
            "tree_shap expects a fitted boosting corrector exposing `.model` "
            "(the raw booster); got an object without it"
        )
    features = _model_features(model, df)
    sub = _subsample(df, max_samples, seed)
    X = sub[features].astype(float).copy()

    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):              # multi-output safeguard
        sv = sv[0]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 1:                      # single-row edge case
        sv = sv[None, :]

    base_value = _expected_value(explainer.expected_value)
    log.info("tree_shap: explained %d rows x %d features (base=%.4f)",
             X.shape[0], X.shape[1], base_value)
    return {"shap_values": sv, "base_value": base_value,
            "features": list(features), "X": X}


def global_importance(shap_result: dict[str, Any]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending.

    Parameters
    ----------
    shap_result
        Output of :func:`tree_shap`.

    Returns
    -------
    DataFrame
        Columns ``feature`` and ``mean_abs_shap`` (one row per feature).
    """
    sv = np.asarray(shap_result["shap_values"], dtype=float)
    feats = list(shap_result["features"])
    mean_abs = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def snow_dependence(shap_result: dict[str, Any], feature: str) -> pd.DataFrame:
    """Dependence-plot data (feature value vs SHAP value) for one feature.

    Designed for the snow drivers (``swe`` / ``smlt`` / ``scf`` / ``t2m`` and
    engineered ``f_*`` snow variants) but works for any explained feature.

    Parameters
    ----------
    shap_result
        Output of :func:`tree_shap`.
    feature
        Name of the feature to extract.

    Returns
    -------
    DataFrame
        Columns ``feature_value`` and ``shap_value``, sorted by feature value;
        carries a ``feature`` attribute in ``df.attrs`` for plotting labels.
    """
    feats = list(shap_result["features"])
    if feature not in feats:
        raise KeyError(f"{feature!r} is not among the explained features")
    j = feats.index(feature)
    X = shap_result["X"]
    fv = np.asarray(X[feature].values if isinstance(X, pd.DataFrame) else X[:, j],
                    dtype=float)
    sv = np.asarray(shap_result["shap_values"], dtype=float)[:, j]
    out = pd.DataFrame({"feature_value": fv, "shap_value": sv})
    out = out.sort_values("feature_value", ignore_index=True)
    out.attrs["feature"] = feature
    return out


def regime_conditional_importance(shap_result: dict[str, Any],
                                  regimes: pd.Series) -> pd.DataFrame:
    """Per-regime mean ``|SHAP|`` table (tidy / long format).

    Reveals *which drivers matter in which hydrological regime* -- e.g. snowmelt
    descriptors dominating the melt-freshet regime, temperature dominating
    rain-on-snow, and glacier descriptors dominating the late-summer glacier
    regime.  Regime labels are typically produced by ``sbc.features.regimes``.

    Parameters
    ----------
    shap_result
        Output of :func:`tree_shap`.
    regimes
        Regime label per row.  Aligned to the explained rows by index when
        possible, otherwise positionally (when lengths match).

    Returns
    -------
    DataFrame
        Long table with columns ``regime``, ``feature`` and ``mean_abs_shap``,
        sorted by regime then importance descending.
    """
    sv = np.asarray(shap_result["shap_values"], dtype=float)
    feats = list(shap_result["features"])
    X = shap_result["X"]
    index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(sv.shape[0])

    reg = pd.Series(regimes)
    if not reg.index.equals(index):
        aligned = reg.reindex(index)
        if aligned.notna().any():
            reg = aligned
        elif len(reg) == sv.shape[0]:
            reg = pd.Series(np.asarray(reg.values), index=index)
        else:
            reg = aligned

    abs_df = pd.DataFrame(np.abs(sv), columns=feats, index=index)
    abs_df["regime"] = reg.values
    abs_df = abs_df[abs_df["regime"].notna()]
    if abs_df.empty:
        log.warning("regime_conditional_importance: no rows with a valid regime")
        return pd.DataFrame(columns=["regime", "feature", "mean_abs_shap"])

    grouped = abs_df.groupby("regime", observed=True)[feats].mean()
    long = (grouped.reset_index()
            .melt(id_vars="regime", var_name="feature", value_name="mean_abs_shap"))
    return long.sort_values(["regime", "mean_abs_shap"],
                            ascending=[True, False], ignore_index=True)


# --------------------------------------------------------------------------- #
#  Model-agnostic fallback for non-tree (deep) models                          #
# --------------------------------------------------------------------------- #
def gradient_importance(model: Any, df: pd.DataFrame, features: list[str] | None = None,
                        n_repeats: int = 5, max_samples: int = 2000,
                        seed: int = 0) -> pd.DataFrame:
    """Permutation importance on the log-residual target (non-tree fallback).

    Tree SHAP is undefined for the deep correctors, so we measure each feature's
    contribution model-agnostically: shuffle the feature, re-predict with
    ``model.predict_residual`` and record the rise in mean-squared error against
    the log-residual target.  Larger rises mean the model relies on the feature
    more.  This keeps the EA-LSTM and regime-probability network explainable
    alongside the boosting models.

    Parameters
    ----------
    model
        Fitted corrector exposing ``predict_residual(df) -> ndarray``.
    df
        Modelling table (subsampled to ``max_samples`` rows).
    features
        Features to score; defaults to ``model.features`` / schema discovery.
    n_repeats
        Number of shuffles averaged per feature.
    max_samples, seed
        Subsample size and RNG seed (deterministic).

    Returns
    -------
    DataFrame
        Columns ``feature``, ``importance`` (mean MSE increase) and
        ``importance_std``, sorted by importance descending.
    """
    if not hasattr(model, "predict_residual"):
        raise AttributeError("gradient_importance needs a model with predict_residual()")
    feats = features or _model_features(model, df)
    sub = _subsample(df, max_samples, seed).reset_index(drop=True)

    if TARGET_COL in sub.columns:
        y = sub[TARGET_COL].to_numpy(dtype=float)
    else:
        y = make_target(sub["q_obs"], sub["q_glofas"])

    def _score(frame: pd.DataFrame) -> float:
        pred = np.asarray(model.predict_residual(frame), dtype=float)
        m = np.isfinite(pred) & np.isfinite(y)
        if not m.any():
            return np.nan
        return float(np.mean((pred[m] - y[m]) ** 2))

    base = _score(sub)
    rng = np.random.default_rng(seed)
    rows = []
    for f in feats:
        deltas = []
        original = sub[f].to_numpy(copy=True)
        for _ in range(int(n_repeats)):
            shuffled = sub.copy()
            shuffled[f] = rng.permutation(original)
            deltas.append(_score(shuffled) - base)
        rows.append({"feature": f,
                     "importance": float(np.mean(deltas)),
                     "importance_std": float(np.std(deltas))})
    out = pd.DataFrame(rows)
    log.info("gradient_importance: permuted %d features over %d rows (base MSE=%.4g)",
             len(feats), len(sub), base)
    return out.sort_values("importance", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
#  Plotting helpers (Agg-safe, write PNGs to results/shap)                      #
# --------------------------------------------------------------------------- #
def _new_axes(figsize: tuple[float, float]):
    """Return a fresh (fig, ax) on the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def save_beeswarm(shap_result: dict[str, Any], path: str | Path | None = None,
                  max_display: int = 20, title: str | None = None) -> Path:
    """Write a SHAP beeswarm summary plot to PNG.

    Parameters
    ----------
    shap_result
        Output of :func:`tree_shap`.
    path
        Destination PNG (defaults to ``results/shap/shap_beeswarm.png``).
    max_display
        Maximum number of features to show.
    title
        Optional figure title.

    Returns
    -------
    Path
        The written PNG path.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import shap

    out = _resolve_path(path, "shap_beeswarm.png")
    plt.figure()
    shap.summary_plot(shap_result["shap_values"], shap_result["X"],
                      feature_names=shap_result["features"],
                      max_display=max_display, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote beeswarm plot -> %s", out)
    return out


def save_dependence(shap_result: dict[str, Any], feature: str,
                    path: str | Path | None = None) -> Path:
    """Write a SHAP dependence scatter (value vs SHAP) for one feature.

    Parameters
    ----------
    shap_result
        Output of :func:`tree_shap`.
    feature
        Feature to plot (typically a snow driver).
    path
        Destination PNG (defaults to ``results/shap/shap_dependence_<feature>.png``).

    Returns
    -------
    Path
        The written PNG path.
    """
    data = snow_dependence(shap_result, feature)
    out = _resolve_path(path, f"shap_dependence_{feature}.png")
    fig, ax = _new_axes((6.0, 4.5))
    sc = ax.scatter(data["feature_value"], data["shap_value"],
                    c=data["feature_value"], cmap="viridis", s=14, alpha=0.8)
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"Dependence: {feature}")
    fig.colorbar(sc, ax=ax, label=feature)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote dependence plot -> %s", out)
    return out


def save_regime_bar(regime_table: pd.DataFrame, path: str | Path | None = None,
                    top_n: int = 12) -> Path:
    """Write a grouped bar chart of per-regime mean ``|SHAP|`` importance.

    Parameters
    ----------
    regime_table
        Long table from :func:`regime_conditional_importance`.
    path
        Destination PNG (defaults to ``results/shap/shap_regime_importance.png``).
    top_n
        Number of top features (by overall mean importance) to display.

    Returns
    -------
    Path
        The written PNG path.
    """
    out = _resolve_path(path, "shap_regime_importance.png")
    if regime_table.empty:
        log.warning("save_regime_bar: empty regime table, nothing to plot")
        fig, ax = _new_axes((6.0, 4.0))
        ax.set_title("No regime-conditional SHAP available")
        fig.savefig(out, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return out

    wide = regime_table.pivot_table(index="feature", columns="regime",
                                    values="mean_abs_shap", fill_value=0.0)
    order = wide.mean(axis=1).sort_values(ascending=False).index[:top_n]
    wide = wide.loc[order]

    feats = list(wide.index)
    regimes = list(wide.columns)
    y = np.arange(len(feats))
    height = 0.8 / max(len(regimes), 1)

    fig, ax = _new_axes((7.0, 0.45 * len(feats) + 1.5))
    for k, reg in enumerate(regimes):
        ax.barh(y + k * height, wide[reg].values, height=height, label=str(reg))
    ax.set_yticks(y + 0.4 - height / 2)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Regime-conditional feature importance")
    ax.legend(title="regime", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote regime importance bar -> %s", out)
    return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                   #
# --------------------------------------------------------------------------- #
def _fallback_booster(df: pd.DataFrame, features: list[str], seed: int = 0):
    """Tiny LightGBM booster wrapper used when sbc.models.boosting is absent."""
    import lightgbm as lgb

    class _Wrapped:
        def __init__(self, booster, feats):
            self.model = booster
            self.features = list(feats)

        def predict_residual(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(self.model.predict(frame[self.features].astype(float)),
                              dtype=float)

    y = make_target(df["q_obs"], df["q_glofas"])
    dset = lgb.Dataset(df[features].astype(float), label=y)
    params = {"objective": "regression", "num_leaves": 15,
              "learning_rate": 0.1, "verbose": -1, "seed": seed}
    booster = lgb.train(params, dset, num_boost_round=60)
    return _Wrapped(booster, features)


def _fallback_regimes(df: pd.DataFrame) -> pd.Series:
    """Minimal internal melt/snow-accumulation/baseflow split for the self-test."""
    smlt = df.get("smlt", pd.Series(0.0, index=df.index))
    swe = df.get("swe", pd.Series(0.0, index=df.index))
    melt_hi = smlt > smlt.quantile(0.66)
    snow_hi = (~melt_hi) & (swe > swe.quantile(0.5))
    reg = np.where(melt_hi, "melt", np.where(snow_hi, "snow_accumulation", "baseflow"))
    return pd.Series(reg, index=df.index, name="regime")


def _selftest() -> None:
    from ..schemas import validate
    from ..synthetic import generate

    df = validate(generate(n_basins=2, gauges_per_basin=(2, 3), years=4,
                           scale="decadal", seed=7))
    features = feature_columns(df)

    # Prefer the real corrector; fall back to a tiny internal booster if the
    # sibling module is not implemented yet (parallel development).
    try:
        from ..models.boosting import BoostingCorrector  # type: ignore

        model = BoostingCorrector("lgbm").fit(df)
        if getattr(model, "model", None) is None:
            raise RuntimeError("BoostingCorrector exposed no raw booster")
        source = "BoostingCorrector('lgbm')"
    except Exception as exc:  # pragma: no cover - depends on sibling module
        log.info("BoostingCorrector unavailable (%s); using fallback booster", exc)
        model = _fallback_booster(df, features, seed=0)
        source = "fallback LightGBM booster"

    res = tree_shap(model, df, max_samples=500, seed=0)
    gi = global_importance(res)
    snow = snow_features(res["features"])
    dep = snow_dependence(res, snow[0]) if snow else None
    reg_tbl = regime_conditional_importance(res, _fallback_regimes(df))
    perm = gradient_importance(model, df, features=features[:4],
                               n_repeats=2, max_samples=300)
    png = save_regime_bar(reg_tbl)

    top5 = list(zip(gi["feature"].head(5), gi["mean_abs_shap"].head(5).round(4)))
    print(f"[shap_analysis] model={source} | rows={res['X'].shape[0]} "
          f"feats={len(res['features'])} | snow_feats={len(snow)} "
          f"| regimes={reg_tbl['regime'].nunique()} | perm_top={perm['feature'].iloc[0]} "
          f"| dep_rows={0 if dep is None else len(dep)} | png={png.name}")
    print(f"[shap_analysis] top-5 global importance: {top5}")


if __name__ == "__main__":
    _selftest()
