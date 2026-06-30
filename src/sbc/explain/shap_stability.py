"""Trustworthiness audit of SHAP / attribution rankings (Slater et al., 2025).

A single SHAP beeswarm or importance bar is an *illustration*: it shows what one
fitted model thought mattered on one explanation set.  A Q1 explainable-ML
reviewer rightly asks the harder question -- *would the same picture survive a
re-fit under a different random seed, or a resample of the explanation rows?*
If the headline drivers reshuffle every time the model is retrained, the
attribution is an artefact of the optimiser, not evidence about the hydrology.

This module turns the framework's attribution layer into a **defensible audit**.
Following the attribution-stability protocol of Slater et al. (2025), it refits
and re-explains a model under ``n_seeds`` independent seeds (optionally
bootstrapping the explanation set as well), then quantifies how stable the global
feature importances are with three complementary statistics:

* **Mean Kendall-tau** between the per-seed importance rankings -- a rank-
  correlation in ``[-1, 1]``; values near ``1`` mean the ordering of drivers is
  reproducible.
* **Jaccard overlap of the top-``k`` feature sets** -- how often the same
  ``k`` headline drivers are recovered, in ``[0, 1]``.
* **Per-feature mean +/- SD of mean\\|SHAP\\|** -- confidence bands on each driver's
  importance, so the paper can plot error bars rather than a single point.

The explainers are reused, never re-implemented:
:func:`sbc.explain.shap_analysis.tree_shap` for the gradient-boosted correctors
and :func:`sbc.explain.flagship_xai.deep_shap` /
:func:`sbc.explain.flagship_xai.integrated_gradients` for the deep flagship.
Feature lists are always discovered from the model / schema, never hard-coded.

All heavy, optional dependencies (``scipy``, ``matplotlib``, the explainers'
``shap`` / ``torch``) are imported lazily inside the functions that need them;
figures are written on the headless Agg backend (never ``plt.show``) to
``results/figures``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger
from .shap_analysis import global_importance, tree_shap

log = get_logger(__name__)

__all__ = [
    "StabilityResult",
    "stability_across_seeds",
    "save_stability_plot",
]


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class StabilityResult:
    """Bundle returned by :func:`stability_across_seeds`.

    Attributes
    ----------
    table : pandas.DataFrame
        Tidy per-feature audit with columns ``feature``, ``mean_abs_shap`` (mean
        over seeds of the global mean\\|SHAP\\|), ``sd`` (between-seed standard
        deviation -- the confidence band), ``cv`` (``sd / mean_abs_shap``, a
        unitless instability score) and ``rank`` (1 = most important), sorted by
        ``mean_abs_shap`` descending.
    kendall_tau : float
        Mean Kendall-tau rank correlation between the per-seed importance
        vectors (``NaN`` when fewer than two usable seeds).
    jaccard_topk : float
        Mean pairwise Jaccard overlap of the per-seed top-``k`` feature sets.
    top_k : int
        ``k`` used for the Jaccard overlap and the default plot.
    n_seeds : int
        Number of seeds that produced a usable explanation.
    method : str
        ``"tree"`` or ``"flagship"`` (echoed for labelling / filenames).
    per_seed : pandas.DataFrame
        Wide matrix of mean\\|SHAP\\| (index = feature, one column per seed); the
        raw material behind the summary statistics.
    pairwise_tau, pairwise_jaccard : list of float
        The individual pairwise statistics that were averaged (diagnostics).
    """

    table: pd.DataFrame
    kendall_tau: float
    jaccard_topk: float
    top_k: int
    n_seeds: int
    method: str
    per_seed: pd.DataFrame
    pairwise_tau: list[float] = field(default_factory=list)
    pairwise_jaccard: list[float] = field(default_factory=list)

    @property
    def scalars(self) -> dict[str, float]:
        """The two headline scalars ``{kendall_tau, jaccard_topk}``."""
        return {"kendall_tau": self.kendall_tau, "jaccard_topk": self.jaccard_topk}

    def top_features(self, k: int | None = None) -> list[str]:
        """Return the ``k`` most stable-on-average features (by mean\\|SHAP\\|)."""
        k = self.top_k if k is None else int(k)
        return self.table["feature"].head(k).tolist()


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
def _instantiate(model_factory: Callable[..., Any], seed: int) -> Any:
    """Build a fresh model, passing ``seed`` when the factory accepts an argument.

    A factory may be ``lambda s: LightGBMCorrector(seed=s)`` (seed-aware) or a
    zero-argument ``lambda: LinearScalingCorrector()`` (deterministic, varied
    only by the explanation-set bootstrap).  Both are supported.
    """
    try:
        return model_factory(seed)
    except TypeError:
        return model_factory()


def _temporal_split(df: pd.DataFrame, train_frac: float
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, explain) by date quantile, with positional fallback.

    Keeps the explanation set *out of sample* so the audit reflects how stable
    the attribution is on data the model did not train on.  Falls back to a
    positional split (and finally to explaining on the training rows) for tiny
    tables.
    """
    if "date" in df.columns and df["date"].notna().any():
        cut = df["date"].quantile(train_frac)
        train = df[df["date"] <= cut]
        explain = df[df["date"] > cut]
        if len(train) >= 5 and len(explain) >= 3:
            return train.reset_index(drop=True), explain.reset_index(drop=True)
    n = len(df)
    k = max(1, int(round(train_frac * n)))
    if k >= n:
        whole = df.reset_index(drop=True)
        return whole, whole
    return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)


def _bootstrap_rows(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Row bootstrap (resample with replacement), deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=len(df))
    return df.iloc[idx].reset_index(drop=True)


def _abs_importance(result: dict[str, Any]) -> pd.Series:
    """Global mean\\|SHAP\\| per feature from any explainer result dict.

    Handles both the :func:`tree_shap` / :func:`deep_shap` shape (``shap_values``)
    and the :func:`integrated_gradients` shape (``attributions``).
    """
    if "shap_values" in result:
        gi = global_importance(result)
        return pd.Series(gi["mean_abs_shap"].to_numpy(float),
                         index=gi["feature"].tolist())
    sv = np.asarray(result["attributions"], dtype=float)
    feats = list(result["features"])
    return pd.Series(np.abs(sv).mean(axis=0), index=feats)


def _explain_one_seed(model: Any, explain_df: pd.DataFrame, *, method: str,
                      max_samples: int, seed: int, flagship_explainer: str,
                      ig_steps: int, background: int) -> pd.Series:
    """Fit-agnostic single-seed global importance as a feature-indexed Series."""
    if method == "flagship":
        from .flagship_xai import deep_shap, integrated_gradients

        if flagship_explainer == "deep_shap":
            res = deep_shap(model, explain_df, background=background,
                            max_samples=max_samples, seed=seed)
        else:  # "ig" -- fast, deterministic, autograd through the flagship
            res = integrated_gradients(model, explain_df, baseline="mean",
                                       steps=ig_steps, max_samples=max_samples,
                                       seed=seed)
        return _abs_importance(res)

    # tree / boosting -> exact tree SHAP on the raw booster
    res = tree_shap(model, explain_df, max_samples=max_samples, seed=seed)
    return _abs_importance(res)


def _mean_pairwise_kendall(per_seed: pd.DataFrame) -> tuple[float, list[float]]:
    """Mean (and list) of pairwise Kendall-tau between seed importance columns."""
    from scipy.stats import kendalltau

    cols = list(per_seed.columns)
    taus: list[float] = []
    for a, b in combinations(cols, 2):
        va = per_seed[a].to_numpy(float)
        vb = per_seed[b].to_numpy(float)
        m = np.isfinite(va) & np.isfinite(vb)
        if m.sum() < 2 or np.std(va[m]) == 0 or np.std(vb[m]) == 0:
            continue
        with np.errstate(invalid="ignore"):
            t = kendalltau(va[m], vb[m]).correlation
        if np.isfinite(t):
            taus.append(float(t))
    return (float(np.mean(taus)) if taus else float("nan")), taus


def _mean_pairwise_jaccard(per_seed: pd.DataFrame, top_k: int
                           ) -> tuple[float, list[float]]:
    """Mean (and list) of pairwise Jaccard overlap of the per-seed top-k sets."""
    top_sets = []
    for c in per_seed.columns:
        s = per_seed[c].dropna().sort_values(ascending=False)
        top_sets.append(set(s.index[:top_k]))
    jacs: list[float] = []
    for A, B in combinations(top_sets, 2):
        union = A | B
        if union:
            jacs.append(len(A & B) / len(union))
    return (float(np.mean(jacs)) if jacs else float("nan")), jacs


# --------------------------------------------------------------------------- #
#  The audit                                                                  #
# --------------------------------------------------------------------------- #
def stability_across_seeds(model_factory: Callable[..., Any], df: pd.DataFrame,
                           n_seeds: int = 5, top_k: int = 10,
                           method: str = "tree", *, train_frac: float = 0.7,
                           max_samples: int = 2000, bootstrap: bool = True,
                           base_seed: int = 0, ig_steps: int = 32,
                           background: int = 100,
                           flagship_explainer: str = "ig") -> StabilityResult:
    """Audit the stability of a model's SHAP importances across re-fits / resamples.

    For each of ``n_seeds`` seeds the routine (i) builds a fresh model via
    ``model_factory`` (passing the seed when the factory accepts one), (ii) fits
    it on the in-sample portion, (iii) explains it on the held-out portion -- for
    ``method="tree"`` optionally bootstrap-resampling those rows -- and (iv)
    records the per-feature global mean\\|SHAP\\|.  The per-seed importance vectors
    are then compared with the mean Kendall-tau rank correlation and the mean
    Jaccard overlap of the top-``k`` driver sets, and summarised into per-feature
    confidence bands.  This converts a one-off SHAP figure into a reproducible
    trustworthiness statement (Slater et al., 2025).

    Parameters
    ----------
    model_factory : callable
        Returns a fresh, *unfitted* :class:`~sbc.models.base.BaseCorrector`.
        Called as ``model_factory(seed)`` when it accepts an argument (so torch /
        boosting RNGs vary), otherwise ``model_factory()`` (variation then comes
        from the explanation-set bootstrap alone).
    df : pandas.DataFrame
        A modelling table (see :mod:`sbc.schemas`); split internally into a
        training and an out-of-sample explanation portion.
    n_seeds : int, default 5
        Number of independent re-fit / re-explain repetitions.
    top_k : int, default 10
        Size of the headline feature set for the Jaccard overlap and the plot.
    method : {"tree", "flagship"}, default "tree"
        ``"tree"`` uses exact :func:`~sbc.explain.shap_analysis.tree_shap` on the
        booster; ``"flagship"`` uses the deep explainers
        (:func:`~sbc.explain.flagship_xai.integrated_gradients` by default, or
        :func:`~sbc.explain.flagship_xai.deep_shap`).
    train_frac : float, default 0.7
        Fraction of the (date-ordered) table used for training; the remainder is
        the explanation set.
    max_samples : int, default 2000
        Cap on explained rows per seed (passed to the explainer).
    bootstrap : bool, default True
        For ``method="tree"`` only, bootstrap-resample the explanation rows each
        seed so the audit also reflects explanation-set sampling variability.
        Ignored for the flagship, whose windowed inputs must stay contiguous.
    base_seed : int, default 0
        Seeds used are ``base_seed + i`` for ``i`` in ``range(n_seeds)``.
    ig_steps : int, default 32
        Riemann steps for the flagship integrated-gradients explainer.
    background : int, default 100
        Background-set size for the flagship ``deep_shap`` explainer.
    flagship_explainer : {"ig", "deep_shap"}, default "ig"
        Which deep explainer to use when ``method="flagship"``.

    Returns
    -------
    StabilityResult
        Holds the tidy per-feature ``table`` (``feature`` / ``mean_abs_shap`` /
        ``sd`` / ``cv`` / ``rank``) and the scalars ``kendall_tau`` /
        ``jaccard_topk`` (also via :pyattr:`StabilityResult.scalars`).

    Raises
    ------
    ValueError
        If ``n_seeds < 1`` or no seed produced a usable explanation.
    """
    from ..utils import seed_everything

    if int(n_seeds) < 1:
        raise ValueError("n_seeds must be >= 1")
    method = str(method).lower()
    if method not in ("tree", "flagship"):
        raise ValueError("method must be 'tree' or 'flagship'")

    train_df, explain_df = _temporal_split(df, float(train_frac))
    log.info("stability_across_seeds: method=%s n_seeds=%d | train=%d explain=%d rows",
             method, n_seeds, len(train_df), len(explain_df))

    columns: dict[str, pd.Series] = {}
    for i in range(int(n_seeds)):
        seed = int(base_seed) + i
        try:
            seed_everything(seed)
            model = _instantiate(model_factory, seed)
            model.fit(train_df, valid=explain_df)

            ex = explain_df
            if method == "tree" and bootstrap:
                ex = _bootstrap_rows(explain_df, seed)
            imp = _explain_one_seed(
                model, ex, method=method, max_samples=max_samples, seed=seed,
                flagship_explainer=flagship_explainer, ig_steps=ig_steps,
                background=background)
        except Exception as exc:  # pragma: no cover - keep the audit robust
            log.warning("stability_across_seeds: seed %d failed (%s); skipping",
                        seed, exc)
            continue
        columns[f"seed_{seed}"] = imp.astype(float)
        log.info("  seed %d: %d features explained (top=%s)",
                 seed, imp.notna().sum(), imp.idxmax() if imp.notna().any() else "n/a")

    if not columns:
        raise ValueError("no seed produced a usable explanation")

    # align all seeds on the feature union; an absent feature contributes 0
    per_seed = pd.DataFrame(columns)
    n_used = per_seed.shape[1]
    filled = per_seed.fillna(0.0)

    mean_abs = filled.mean(axis=1)
    sd = filled.std(axis=1, ddof=1) if n_used > 1 else pd.Series(0.0, index=filled.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = (sd / mean_abs.replace(0.0, np.nan)).fillna(0.0)

    table = pd.DataFrame({
        "feature": filled.index.to_numpy(),
        "mean_abs_shap": mean_abs.to_numpy(float),
        "sd": sd.to_numpy(float),
        "cv": cv.to_numpy(float),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1, dtype=int)

    kendall_tau, pw_tau = _mean_pairwise_kendall(filled)
    jaccard_topk, pw_jac = _mean_pairwise_jaccard(per_seed, int(top_k))
    if n_used < 2:
        log.warning("stability_across_seeds: only %d usable seed(s); the "
                    "stability scalars are undefined (NaN)", n_used)

    log.info("stability_across_seeds: kendall_tau=%.3f jaccard@%d=%.3f over %d seeds",
             kendall_tau, top_k, jaccard_topk, n_used)
    return StabilityResult(
        table=table, kendall_tau=kendall_tau, jaccard_topk=jaccard_topk,
        top_k=int(top_k), n_seeds=n_used, method=method, per_seed=per_seed,
        pairwise_tau=pw_tau, pairwise_jaccard=pw_jac)


# --------------------------------------------------------------------------- #
#  Plotting (Agg-safe, writes to results/figures)                             #
# --------------------------------------------------------------------------- #
def _figure_path(path: str | Path | None, default_name: str) -> Path:
    """Return a writable PNG path, defaulting under ``results/figures``."""
    if path is None:
        PATHS.figures.mkdir(parents=True, exist_ok=True)
        return PATHS.figures / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_stability_plot(result: StabilityResult, path: str | Path | None = None, *,
                        top_k: int | None = None, title: str | None = None) -> Path:
    """Write a top-``k`` importance bar chart with between-seed error bars.

    Bars are the mean mean\\|SHAP\\| across seeds; the error bars are the
    between-seed standard deviation (the confidence band).  The title carries the
    audit scalars so the figure is self-documenting.

    Parameters
    ----------
    result : StabilityResult
        Output of :func:`stability_across_seeds`.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to
        ``results/figures/shap_stability_<method>.png``).
    top_k : int, optional
        Number of features to display (defaults to ``result.top_k``).
    title : str, optional
        Override the auto-generated title.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    table = result.table if isinstance(result, StabilityResult) else result
    method = getattr(result, "method", "model")
    k = int(top_k if top_k is not None else getattr(result, "top_k", 10))
    out = _figure_path(path, f"shap_stability_{method}.png")

    t = table.head(k).iloc[::-1]  # largest bar on top
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(t) + 1.8))
    ax.barh(y, t["mean_abs_shap"].to_numpy(float),
            xerr=t["sd"].to_numpy(float), color="#1f77b4", ecolor="0.3",
            capsize=3, error_kw={"lw": 1.0})
    ax.set_yticks(y)
    ax.set_yticklabels(t["feature"].tolist())
    ax.set_xlabel("mean |SHAP|  (+/- SD across seeds)")

    if title is None:
        kt = getattr(result, "kendall_tau", float("nan"))
        jac = getattr(result, "jaccard_topk", float("nan"))
        n = getattr(result, "n_seeds", "?")
        title = (f"Attribution stability ({method}): "
                 f"Kendall-tau={kt:.2f}, Jaccard@{k}={jac:.2f}  ({n} seeds)")
    ax.set_title(title, fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote stability plot -> %s", out)
    return out


# --------------------------------------------------------------------------- #
#  Self-test (tiny LightGBM over 3 seeds; <3 min on synthetic)                #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.boosting import LightGBMCorrector
    from ..schemas import validate
    from ..synthetic import generate

    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(validate(df), scale="decadal")).reset_index(drop=True)
    print(f"[shap_stability] table: {len(df)} rows | {df['code'].nunique()} gauges "
          f"| {df['scale'].iloc[0]} scale")

    # tiny, fast booster; seed wired in so each re-fit genuinely differs
    def factory(seed: int) -> LightGBMCorrector:
        return LightGBMCorrector(
            params=dict(n_estimators=80, num_leaves=15, learning_rate=0.1),
            seed=seed)

    res = stability_across_seeds(factory, df, n_seeds=3, top_k=10, method="tree",
                                 max_samples=600, bootstrap=True)

    assert isinstance(res, StabilityResult)
    assert {"feature", "mean_abs_shap", "sd", "rank"} <= set(res.table.columns)
    assert res.table["rank"].tolist() == list(range(1, len(res.table) + 1))
    assert (res.table["sd"] >= 0).all()
    assert res.scalars.keys() == {"kendall_tau", "jaccard_topk"}
    assert np.isfinite(res.kendall_tau) and -1.0 <= res.kendall_tau <= 1.0
    assert 0.0 <= res.jaccard_topk <= 1.0

    print(f"[shap_stability] kendall_tau={res.kendall_tau:+.3f} | "
          f"jaccard@{res.top_k}={res.jaccard_topk:.3f} | seeds={res.n_seeds} "
          f"| features={len(res.table)}")
    top5 = res.table.head(5)[["feature", "mean_abs_shap", "sd"]]
    print("[shap_stability] top-5 stable features (mean |SHAP| +/- SD across seeds):")
    for _, r in top5.iterrows():
        print(f"    {r['feature']:<22s} {r['mean_abs_shap']:.4f} +/- {r['sd']:.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        png = save_stability_plot(res, Path(tmp) / "shap_stability_tree.png")
        size = png.stat().st_size
        assert png.exists() and size > 0
    print(f"[shap_stability] wrote {png.name} ({size} bytes) | SELF-TEST OK")


if __name__ == "__main__":
    _selftest()
