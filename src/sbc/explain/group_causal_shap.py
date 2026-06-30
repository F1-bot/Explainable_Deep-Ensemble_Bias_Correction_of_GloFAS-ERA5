"""Group-level, correlation-robust attribution for the *sbc* framework.

Per-feature SHAP is the right tool only when the inputs are (roughly)
independent.  The snow-influenced Central-Asian feature table is the opposite: a
single physical signal -- "there is a melting snowpack" -- is encoded redundantly
across snow-water-equivalent (``swe``), snowmelt (``smlt``), air temperature
(``t2m*``), their engineered lags / rolling moments and the GloFAS memory block.
When several collinear columns carry the *same* information, a Shapley game has
no principled way to choose between them: it *splits* the credit (and, with
opposite-signed local effects, may *cancel* it) differently on every re-fit.
That is exactly why the flagship's per-feature SHAP ranking is unstable across
seeds (Kendall-:math:`\\tau \\approx 0.27`) even though the *physics* the model
keys on does not change.

This module removes the within-block ambiguity by attributing at the level of
**physically meaningful feature groups** rather than individual columns:

``snow``
    The cryosphere / energy block -- ``swe``, ``smlt``, ``sf``, the temperature
    drivers (``t2m*``) and every engineered snow / melt / positive-degree-day /
    rain-on-snow descriptor.  This is the block the paper's "physically
    constrained" story is about.
``glofas_memory``
    The raw model's own recent behaviour -- ``f_log_qglofas*`` lags, causal
    rolling moments and rate-of-change.
``meteo``
    Liquid-water meteorology not specific to snow -- total precipitation
    (``tp``), soil moisture (``swvl1``) and similar.
``seasonality``
    The smooth day-of-year / decade-index harmonics (``f_doy*`` / ``f_decade*``).
``static``
    Per-gauge-constant catchment attributes (area, elevation, slope, glacier /
    snow fraction, aridity and collapsed climatology descriptors), discovered by
    their zero within-gauge variance via :func:`sbc.schemas.static_feature_columns`.

Because every member of a block carries the same signal, the *block's* total
attribution is well defined and stable even when the split between its members is
not.  Two complementary estimators are provided, both reusing the framework's
existing explainers (never re-implementing them):

* :func:`group_shap` -- group attributions.  Two engines, selected by ``method``:

  ``"aggregate"`` (the default for the tree models and the flagship) takes the
  additive per-feature attributions of an existing explainer -- exact tree SHAP
  (:func:`sbc.explain.shap_analysis.tree_shap`) for the boosters, integrated
  gradients (:func:`sbc.explain.flagship_xai.integrated_gradients`) for the
  flagship -- and *sums* them within each block.  Because the per-feature values
  are additive (they obey completeness / efficiency), the block sum is the
  block's joint contribution, with the within-block splitting / cancellation
  summed away.

  ``"coalition"`` plays the Shapley game directly at the group level: each block
  is a single player and the coalition value ``v(S)`` is computed by
  *marginalising* the complement over a background sample (the interventional /
  observational value function of Lundberg et al., 2020).  For the handful of
  blocks here all :math:`2^{G}` coalitions are enumerated exactly, so the result
  is the true group Shapley value -- the most defensible reading of "treat each
  block as one player".

* :func:`group_stability` -- the Slater-et-al.-style seed audit of
  :mod:`sbc.explain.shap_stability`, but on the *group* ranking, reporting the
  group-level Kendall-:math:`\\tau` / Jaccard alongside the per-feature numbers so
  the paper can show the instability is cured by grouping.

* :func:`robust_feature_importance` -- a correlation-*robust* per-feature score
  for the headline drivers, based on accumulated local effects (ALE, unbiased
  under input correlation) or on conditional permutation within strata of a
  correlated partner.

All heavy / optional dependencies (``scipy``, ``shap``, ``torch``,
``matplotlib``) are imported lazily inside the functions that need them; figures
are written on the headless Agg backend (never ``plt.show``) to
``results/figures``; feature lists and the static/dynamic split are always
discovered from the model / schema, never hard-coded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import PATHS
from ..schemas import (
    TARGET_COL,
    feature_columns,
    make_target,
    static_feature_columns,
)
from ..utils import get_logger
from .shap_analysis import _subsample, snow_features

log = get_logger(__name__)

__all__ = [
    "FEATURE_GROUPS",
    "GROUP_ORDER",
    "group_membership",
    "group_shap",
    "group_importance",
    "GroupStabilityResult",
    "group_stability",
    "robust_feature_importance",
    "save_group_bar",
]


# --------------------------------------------------------------------------- #
#  Group vocabulary and the discovery rules                                    #
# --------------------------------------------------------------------------- #
#: Canonical group order (the ``static`` block is appended last because it is
#: discovered by per-gauge variance rather than by name).
GROUP_ORDER: tuple[str, ...] = (
    "snow", "glofas_memory", "meteo", "seasonality", "static",
)

#: Human-readable description of each block (for docs / figure captions).
FEATURE_GROUPS: dict[str, str] = {
    "snow": "swe/smlt/sf, temperature (t2m*) and engineered snow/melt/pdd/ros descriptors",
    "glofas_memory": "f_log_qglofas* lags, causal rolling moments and rate-of-change",
    "meteo": "non-snow liquid-water meteorology (tp, swvl1, ...)",
    "seasonality": "smooth day-of-year / decade-index harmonics (f_doy*, f_decade*)",
    "static": "per-gauge-constant catchment attributes (area/elev/slope/glacier/aridity/climatology)",
}

#: Lower-cased substring tokens that mark a *dynamic* feature as snow-/cryosphere-
#: /energy-related.  Ambiguous very short tokens (e.g. ``sf``) are matched exactly
#: instead, see :func:`_classify_one`.
_SNOW_TOKENS: tuple[str, ...] = (
    "swe", "smlt", "scf", "snow", "snw", "melt", "pdd", "ros",
    "t2m", "temp", "tas", "freez", "thaw", "glac", "ice",
)
#: Tokens marking the GloFAS-memory block.
_GLOFAS_TOKENS: tuple[str, ...] = ("glofas", "qglofas")
#: Tokens marking the seasonality block.
_SEASON_TOKENS: tuple[str, ...] = ("doy", "decade")
#: Dynamic columns matched *exactly* into the snow block (guards short tokens).
_SNOW_EXACT: frozenset[str] = frozenset({"sf"})


def _classify_one(name: str) -> str:
    """Return the thematic (non-``static``) group of a single feature name.

    The precedence is ``glofas_memory`` -> ``seasonality`` -> ``snow`` ->
    ``meteo`` (the catch-all), chosen so the most specific block wins; the
    ``static`` block is assigned separately by :func:`group_membership` from the
    per-gauge variance.

    Parameters
    ----------
    name : str
        Feature column name.

    Returns
    -------
    str
        One of ``"glofas_memory"``, ``"seasonality"``, ``"snow"`` or ``"meteo"``.
    """
    low = str(name).lower()
    if any(tok in low for tok in _GLOFAS_TOKENS):
        return "glofas_memory"
    if any(tok in low for tok in _SEASON_TOKENS):
        return "seasonality"
    if low in _SNOW_EXACT or any(tok in low for tok in _SNOW_TOKENS):
        return "snow"
    return "meteo"


def group_membership(df: pd.DataFrame, features: list[str] | None = None,
                     groups: Mapping[str, Sequence[str]] | None = None
                     ) -> dict[str, str]:
    """Map every model-input feature of ``df`` to a physical block.

    Static catchment attributes (columns that are constant within each gauge, per
    :func:`sbc.schemas.static_feature_columns`) are assigned to the ``static``
    block first; the remaining dynamic columns are routed by name through
    :func:`_classify_one` into ``snow`` / ``glofas_memory`` / ``meteo`` /
    ``seasonality``.

    Parameters
    ----------
    df : pandas.DataFrame
        A modelling table (see :mod:`sbc.schemas`).  ``code`` is used to detect
        per-gauge-constant (static) columns.
    features : list of str, optional
        Restrict the mapping to these columns (defaults to all numeric model
        inputs via :func:`sbc.schemas.feature_columns`).  Any name not present in
        ``df`` is dropped.
    groups : mapping of str to sequence of str, optional
        Explicit override ``{group_name: [feature, ...]}``.  Listed features take
        the given group verbatim; every other feature is auto-classified.  Lets a
        caller hand-curate a block without losing the automatic routing of the
        rest.

    Returns
    -------
    dict of str to str
        ``{feature: group_name}`` for every resolved feature, insertion-ordered
        by ``features``.
    """
    feats = list(features) if features is not None else feature_columns(df)
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise ValueError("group_membership: no usable feature columns found")

    explicit: dict[str, str] = {}
    if groups:
        for gname, members in groups.items():
            for f in members:
                explicit[str(f)] = str(gname)

    static = set(static_feature_columns(df, feats))
    out: dict[str, str] = {}
    for f in feats:
        if f in explicit:
            out[f] = explicit[f]
        elif f in static:
            out[f] = "static"
        else:
            out[f] = _classify_one(f)
    return out


def _ordered_groups(membership: Mapping[str, str]) -> list[str]:
    """Present group names, in :data:`GROUP_ORDER` first then any extras."""
    present = set(membership.values())
    ordered = [g for g in GROUP_ORDER if g in present]
    ordered += [g for g in sorted(present) if g not in GROUP_ORDER]
    return ordered


# --------------------------------------------------------------------------- #
#  Per-feature -> group aggregation of an additive explainer                   #
# --------------------------------------------------------------------------- #
def _is_flagship(model: Any) -> bool:
    """Heuristic flagship probe (mirrors :mod:`sbc.explain.flagship_xai`)."""
    return (
        getattr(model, "net", None) is not None
        and hasattr(model, "_design_matrices")
        and bool(getattr(model, "dyn_cols", []))
    )


def _has_booster(model: Any) -> bool:
    """True when the corrector exposes a raw tree booster for exact tree SHAP."""
    return getattr(model, "model", None) is not None and not _is_flagship(model)


def _per_feature_attribution(model: Any, df: pd.DataFrame, *, engine: str,
                             max_samples: int, ig_steps: int, background: int,
                             seed: int) -> dict[str, Any]:
    """Run the appropriate per-feature explainer and normalise its output.

    Returns a dict with keys ``attr`` (signed ``(n, f)`` ndarray), ``features``,
    ``X`` and ``base_value`` regardless of which explainer produced it.
    """
    if engine == "tree":
        from .shap_analysis import tree_shap

        res = tree_shap(model, df, max_samples=max_samples, seed=seed)
        return {"attr": np.asarray(res["shap_values"], float),
                "features": list(res["features"]), "X": res["X"],
                "base_value": float(res["base_value"])}

    # "ig" -- integrated gradients (exact autograd for the flagship, vectorised
    # finite differences for any other corrector with predict_residual).
    from .flagship_xai import integrated_gradients

    res = integrated_gradients(model, df, baseline="mean", steps=int(ig_steps),
                               max_samples=max_samples, seed=seed)
    return {"attr": np.asarray(res["attributions"], float),
            "features": list(res["features"]), "X": res["X"],
            "base_value": float(res.get("base_value", 0.0))}


def _aggregate_to_groups(attr: np.ndarray, features: Sequence[str],
                         membership: Mapping[str, str], group_names: Sequence[str]
                         ) -> np.ndarray:
    """Sum signed per-feature attributions into ``(n, len(group_names))`` blocks."""
    gidx = {g: k for k, g in enumerate(group_names)}
    gv = np.zeros((attr.shape[0], len(group_names)), dtype=float)
    for j, f in enumerate(features):
        g = membership.get(f)
        if g in gidx:
            gv[:, gidx[g]] += attr[:, j]
    return gv


def _group_aggregate(model: Any, df: pd.DataFrame, groups, *, engine: str,
                     max_samples: int, ig_steps: int, background: int,
                     seed: int) -> dict[str, Any]:
    """Group attributions by summing an additive per-feature explainer."""
    pf = _per_feature_attribution(model, df, engine=engine, max_samples=max_samples,
                                  ig_steps=ig_steps, background=background, seed=seed)
    membership = group_membership(df, features=pf["features"], groups=groups)
    group_names = _ordered_groups(membership)
    gv = _aggregate_to_groups(pf["attr"], pf["features"], membership, group_names)
    log.info("group_shap(aggregate/%s): %d rows -> %d groups (%s)",
             engine, gv.shape[0], gv.shape[1], ", ".join(group_names))
    return {
        "group_values": gv,
        "groups": group_names,
        "membership": membership,
        "base_value": pf["base_value"],
        "X": pf["X"],
        "method": f"aggregate/{engine}",
        "feature_attributions": pf["attr"],
        "features": pf["features"],
    }


# --------------------------------------------------------------------------- #
#  Reduced-game group Shapley by interventional marginalisation                #
# --------------------------------------------------------------------------- #
def _predict_residual_frame(model: Any, Z: np.ndarray, feats: Sequence[str]
                            ) -> np.ndarray:
    """Evaluate ``model.predict_residual`` on a plain feature matrix."""
    frame = pd.DataFrame(np.asarray(Z, dtype=float), columns=list(feats))
    return np.asarray(model.predict_residual(frame), dtype=float)


def _group_coalition(model: Any, df: pd.DataFrame, groups, *, max_samples: int,
                     background: int, n_perm: int | None, seed: int
                     ) -> dict[str, Any]:
    """Exact (or sampled) group Shapley with an interventional value function.

    Each block is a single player; ``v(S)`` is the prediction with the features
    of the blocks in ``S`` held at their instance values and the complement
    *marginalised* over a background sample.  With ``G`` small, all :math:`2^{G}`
    coalitions are enumerated exactly so the returned group values are the true
    Shapley values and obey completeness: ``group_values.sum(1) + base_value ==
    predict_residual``.
    """
    if _is_flagship(model):
        raise ValueError(
            "method='coalition' uses a static feature matrix and is undefined "
            "for the windowed flagship; use method='aggregate' (integrated "
            "gradients) for RegimeProbNet")
    if not hasattr(model, "predict_residual"):
        raise AttributeError("group_shap(coalition) needs a predict_residual model")

    feats = list(getattr(model, "features", None) or feature_columns(df))
    feats = [f for f in feats if f in df.columns]
    membership = group_membership(df, features=feats, groups=groups)
    group_names = _ordered_groups(membership)
    G = len(group_names)
    gcols: dict[str, np.ndarray] = {
        g: np.array([i for i, f in enumerate(feats) if membership[f] == g], dtype=int)
        for g in group_names
    }

    dfx = df.reset_index(drop=True)
    n_fg = min(int(max_samples), len(dfx))
    rng = np.random.default_rng(seed)
    fg_idx = np.sort(rng.choice(len(dfx), size=n_fg, replace=False))
    bg_size = min(int(background), len(dfx))
    bg_idx = np.sort(rng.choice(len(dfx), size=bg_size, replace=False))
    X = dfx[feats].to_numpy(dtype=float)
    X_fg = X[fg_idx]                       # (n_fg, p)
    X_bg = X[bg_idx]                       # (B, p)
    B = X_bg.shape[0]

    bg_block = np.repeat(X_bg, n_fg, axis=0)           # (B*n_fg, p)
    fg_tile = np.tile(X_fg, (B, 1))                    # (B*n_fg, p)

    def value(member_mask: np.ndarray) -> np.ndarray:
        """Interventional ``v(S)`` for the group set encoded by ``member_mask``."""
        Z = np.where(member_mask[None, :], fg_tile, bg_block)
        pred = _predict_residual_frame(model, Z, feats)
        return pred.reshape(B, n_fg).mean(axis=0)      # (n_fg,)

    def cols_of(active: Sequence[int]) -> np.ndarray:
        """Boolean feature mask for the groups whose indices are ``active``."""
        m = np.zeros(len(feats), dtype=bool)
        for g in active:
            m[gcols[group_names[g]]] = True
        return m

    exact = (n_perm is None) and (G <= 12)
    if exact:
        v_cache: dict[int, np.ndarray] = {}
        for bits in range(1 << G):
            active = [g for g in range(G) if bits & (1 << g)]
            v_cache[bits] = value(cols_of(active))
        w = [math.factorial(s) * math.factorial(G - s - 1) / math.factorial(G)
             for s in range(G)]
        phi = np.zeros((n_fg, G), dtype=float)
        for g in range(G):
            bit = 1 << g
            others = [o for o in range(G) if o != g]
            for r in range(len(others) + 1):
                for combo in combinations(others, r):
                    mask = 0
                    for o in combo:
                        mask |= (1 << o)
                    phi[:, g] += w[r] * (v_cache[mask | bit] - v_cache[mask])
        base = float(np.mean(v_cache[0]))
    else:                                  # Monte-Carlo over group permutations
        n_perm = int(n_perm or 200)
        phi = np.zeros((n_fg, G), dtype=float)
        v_empty = value(cols_of([]))
        base = float(np.mean(v_empty))
        for _ in range(n_perm):
            order = rng.permutation(G)
            active: list[int] = []
            prev = v_empty
            for g in order:
                active.append(int(g))
                cur = value(cols_of(active))
                phi[:, g] += cur - prev
                prev = cur
        phi /= n_perm

    log.info("group_shap(coalition): %d fg x %d bg rows, %d groups, %s",
             n_fg, B, G, "exact 2^G" if exact else f"{n_perm} perms")
    return {
        "group_values": phi,
        "groups": group_names,
        "membership": membership,
        "base_value": base,
        "X": dfx.iloc[fg_idx][feats].astype(float).copy(),
        "method": "coalition",
        "feature_attributions": None,
        "features": feats,
    }


# --------------------------------------------------------------------------- #
#  Public group-attribution entry point                                        #
# --------------------------------------------------------------------------- #
def group_shap(model: Any, df: pd.DataFrame,
               groups: Mapping[str, Sequence[str]] | None = None,
               max_samples: int = 2000, *, method: str = "auto",
               background: int = 100, ig_steps: int = 32,
               n_perm: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Group-level Shapley / attribution values for a fitted corrector.

    Each physical block (see :func:`group_membership`) is treated as one player,
    so the within-block credit ambiguity that destabilises per-feature SHAP under
    the snow/temperature/GloFAS collinearity is removed.

    Parameters
    ----------
    model
        Fitted :class:`~sbc.models.base.BaseCorrector` exposing
        ``predict_residual`` (and, for the flagship, the windowed internals).
    df
        Modelling table to explain.
    groups : mapping of str to sequence of str, optional
        Explicit block override forwarded to :func:`group_membership`.
    max_samples : int, default 2000
        Cap on explained rows.
    method : {"auto", "aggregate", "coalition"}, default "auto"
        ``"aggregate"`` sums an additive per-feature explainer (exact tree SHAP
        for boosters, integrated gradients for the flagship / others) into the
        blocks; ``"coalition"`` plays the Shapley game directly at the group level
        with an interventional value function (tabular models only); ``"auto"``
        picks ``"aggregate"`` for the flagship and the boosters and
        ``"coalition"`` otherwise.
    background : int, default 100
        Background-sample size for the ``"coalition"`` marginalisation.
    ig_steps : int, default 32
        Riemann steps for the integrated-gradients per-feature explainer.
    n_perm : int, optional
        If given (or when there are more than 12 groups), the coalition Shapley is
        estimated by sampling this many group permutations instead of the exact
        :math:`2^{G}` enumeration.
    seed : int, default 0
        RNG seed for the subsample / background draws.

    Returns
    -------
    dict
        ``{"group_values": (n, G) ndarray, "groups": list[str],
        "membership": {feature: group}, "base_value": float, "X": DataFrame,
        "method": str, "feature_attributions": (n, f) ndarray | None,
        "features": list[str]}``.  ``group_values.sum(1) + base_value`` recovers
        ``predict_residual`` (completeness) for both engines.
    """
    method = str(method).lower()
    if method not in ("auto", "aggregate", "coalition"):
        raise ValueError("method must be 'auto', 'aggregate' or 'coalition'")

    if method == "coalition":
        return _group_coalition(model, df, groups, max_samples=max_samples,
                                background=background, n_perm=n_perm, seed=seed)

    if method == "auto" and not (_is_flagship(model) or _has_booster(model)):
        # a generic tabular corrector: the principled reduced-game is available
        return _group_coalition(model, df, groups, max_samples=max_samples,
                                background=background, n_perm=n_perm, seed=seed)

    engine = "tree" if _has_booster(model) else "ig"
    return _group_aggregate(model, df, groups, engine=engine, max_samples=max_samples,
                            ig_steps=ig_steps, background=background, seed=seed)


def group_importance(result: Mapping[str, Any]) -> pd.DataFrame:
    """Mean absolute group attribution per block, sorted descending.

    Parameters
    ----------
    result
        Output of :func:`group_shap`.

    Returns
    -------
    DataFrame
        Columns ``group``, ``mean_abs`` (mean ``|group attribution|`` over rows),
        ``share`` (fraction of the total) and ``rank`` (1 = most important).
    """
    gv = np.asarray(result["group_values"], dtype=float)
    groups = list(result["groups"])
    mean_abs = np.abs(gv).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    out = pd.DataFrame({
        "group": groups,
        "mean_abs": mean_abs,
        "share": mean_abs / total,
    }).sort_values("mean_abs", ascending=False, ignore_index=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


# --------------------------------------------------------------------------- #
#  Seed-stability audit at the group level                                     #
# --------------------------------------------------------------------------- #
@dataclass
class GroupStabilityResult:
    """Bundle returned by :func:`group_stability`.

    Attributes
    ----------
    table : pandas.DataFrame
        Per-group audit: ``group``, ``mean_abs`` (mean over seeds of the group
        importance), ``sd`` (between-seed SD), ``cv`` and ``rank``.
    kendall_tau : float
        Mean pairwise Kendall-:math:`\\tau` between the per-seed *group* rankings
        (the headline: should sit far above the per-feature value).
    jaccard_topk : float
        Mean pairwise Jaccard overlap of the per-seed top-``k`` group sets.
    feature_kendall_tau : float
        The same statistic computed on the *per-feature* rankings (when the
        aggregate engine exposed them), for the side-by-side contrast; ``NaN`` for
        the coalition engine, which produces no per-feature values.
    top_k, n_seeds, method : int, int, str
        Audit settings echoed for labelling.
    per_seed : pandas.DataFrame
        Wide matrix of group importances (index = group, one column per seed).
    pairwise_tau, pairwise_jaccard : list of float
        The individual pairwise statistics that were averaged.
    """

    table: pd.DataFrame
    kendall_tau: float
    jaccard_topk: float
    feature_kendall_tau: float
    top_k: int
    n_seeds: int
    method: str
    per_seed: pd.DataFrame
    pairwise_tau: list[float] = field(default_factory=list)
    pairwise_jaccard: list[float] = field(default_factory=list)

    @property
    def scalars(self) -> dict[str, float]:
        """Headline scalars ``{kendall_tau, jaccard_topk, feature_kendall_tau}``."""
        return {"kendall_tau": self.kendall_tau, "jaccard_topk": self.jaccard_topk,
                "feature_kendall_tau": self.feature_kendall_tau}


def _temporal_split(df: pd.DataFrame, train_frac: float
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date-quantile train/explain split with a positional fallback for tiny tables."""
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


def _instantiate(model_factory: Callable[..., Any], seed: int) -> Any:
    """Build a fresh model, passing ``seed`` when the factory accepts one."""
    try:
        return model_factory(seed)
    except TypeError:
        return model_factory()


def _mean_pairwise_kendall(per_seed: pd.DataFrame) -> tuple[float, list[float]]:
    """Mean (and list) of pairwise Kendall-:math:`\\tau` between seed columns."""
    from scipy.stats import kendalltau

    taus: list[float] = []
    for a, b in combinations(list(per_seed.columns), 2):
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
    top_sets = [set(per_seed[c].dropna().sort_values(ascending=False).index[:top_k])
                for c in per_seed.columns]
    jacs: list[float] = []
    for A, B in combinations(top_sets, 2):
        union = A | B
        if union:
            jacs.append(len(A & B) / len(union))
    return (float(np.mean(jacs)) if jacs else float("nan")), jacs


def group_stability(model_factory: Callable[..., Any], df: pd.DataFrame,
                    n_seeds: int = 5, top_k: int = 3, *, method: str = "auto",
                    train_frac: float = 0.7, max_samples: int = 1500,
                    base_seed: int = 0, ig_steps: int = 24, background: int = 80
                    ) -> GroupStabilityResult:
    """Audit the seed-stability of the *group* attribution ranking.

    For each of ``n_seeds`` seeds a fresh model is built via ``model_factory``,
    fitted on the in-sample portion and explained with :func:`group_shap` on the
    held-out portion; the per-seed group importances are then compared with the
    mean Kendall-:math:`\\tau` and the top-``k`` Jaccard overlap.  When the
    aggregate engine also returns per-feature attributions, the per-feature
    Kendall-:math:`\\tau` is computed from the *same* re-fits, so the figure can
    show -- at no extra cost -- that grouping is what cures the instability.

    Parameters
    ----------
    model_factory : callable
        Returns a fresh, unfitted corrector; called as ``model_factory(seed)``
        when it accepts an argument, otherwise ``model_factory()``.
    df : pandas.DataFrame
        Modelling table, split internally into train / explain portions.
    n_seeds : int, default 5
        Number of independent re-fit / re-explain repetitions.
    top_k : int, default 3
        Headline group-set size for the Jaccard overlap and the plot.
    method : {"auto", "aggregate", "coalition"}, default "auto"
        Forwarded to :func:`group_shap`.
    train_frac : float, default 0.7
        Fraction of the date-ordered table used for training.
    max_samples : int, default 1500
        Cap on explained rows per seed.
    base_seed : int, default 0
        Seeds used are ``base_seed + i`` for ``i in range(n_seeds)``.
    ig_steps, background : int
        Forwarded to the flagship / coalition engines of :func:`group_shap`.

    Returns
    -------
    GroupStabilityResult
        The per-group ``table`` and the scalars ``kendall_tau`` /
        ``jaccard_topk`` / ``feature_kendall_tau``.

    Raises
    ------
    ValueError
        If ``n_seeds < 1`` or no seed produced a usable explanation.
    """
    from ..utils import seed_everything

    if int(n_seeds) < 1:
        raise ValueError("n_seeds must be >= 1")

    train_df, explain_df = _temporal_split(df, float(train_frac))
    log.info("group_stability: n_seeds=%d method=%s | train=%d explain=%d rows",
             n_seeds, method, len(train_df), len(explain_df))

    group_cols: dict[str, pd.Series] = {}
    feat_cols: dict[str, pd.Series] = {}
    used_method = method
    for i in range(int(n_seeds)):
        seed = int(base_seed) + i
        try:
            seed_everything(seed)
            model = _instantiate(model_factory, seed)
            model.fit(train_df, valid=explain_df)
            res = group_shap(model, explain_df, method=method, max_samples=max_samples,
                             ig_steps=ig_steps, background=background, seed=seed)
        except Exception as exc:  # pragma: no cover - keep the audit robust
            log.warning("group_stability: seed %d failed (%s); skipping", seed, exc)
            continue
        used_method = res["method"]
        gi = group_importance(res)
        group_cols[f"seed_{seed}"] = pd.Series(gi["mean_abs"].to_numpy(float),
                                               index=gi["group"].tolist())
        fa = res.get("feature_attributions")
        if fa is not None:
            feat_cols[f"seed_{seed}"] = pd.Series(
                np.abs(np.asarray(fa, float)).mean(axis=0), index=res["features"])

    if not group_cols:
        raise ValueError("no seed produced a usable group explanation")

    per_seed = pd.DataFrame(group_cols)
    n_used = per_seed.shape[1]
    filled = per_seed.fillna(0.0)
    mean_abs = filled.mean(axis=1)
    sd = filled.std(axis=1, ddof=1) if n_used > 1 else pd.Series(0.0, index=filled.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = (sd / mean_abs.replace(0.0, np.nan)).fillna(0.0)

    table = pd.DataFrame({
        "group": filled.index.to_numpy(),
        "mean_abs": mean_abs.to_numpy(float),
        "sd": sd.to_numpy(float),
        "cv": cv.to_numpy(float),
    }).sort_values("mean_abs", ascending=False, ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1, dtype=int)

    kendall_tau, pw_tau = _mean_pairwise_kendall(filled)
    jaccard_topk, pw_jac = _mean_pairwise_jaccard(per_seed, int(top_k))
    feat_kendall = float("nan")
    if len(feat_cols) >= 2:
        feat_df = pd.DataFrame(feat_cols).fillna(0.0)
        feat_kendall, _ = _mean_pairwise_kendall(feat_df)

    log.info("group_stability: group kendall_tau=%.3f jaccard@%d=%.3f | "
             "feature kendall_tau=%.3f over %d seeds",
             kendall_tau, top_k, jaccard_topk, feat_kendall, n_used)
    return GroupStabilityResult(
        table=table, kendall_tau=kendall_tau, jaccard_topk=jaccard_topk,
        feature_kendall_tau=feat_kendall, top_k=int(top_k), n_seeds=n_used,
        method=str(used_method), per_seed=per_seed,
        pairwise_tau=pw_tau, pairwise_jaccard=pw_jac)


# --------------------------------------------------------------------------- #
#  Correlation-robust per-feature importance for the headline drivers          #
# --------------------------------------------------------------------------- #
def _candidate_features(model: Any, df: pd.DataFrame, features: Sequence[str] | None,
                        top: int) -> list[str]:
    """Resolve the per-feature shortlist, defaulting to present snow drivers."""
    if features is not None:
        cands = [f for f in features if f in df.columns]
    else:
        if _is_flagship(model):
            base = list(getattr(model, "dyn_cols", [])) + list(getattr(model, "stat_cols", []))
        else:
            base = list(getattr(model, "features", None) or feature_columns(df))
        base = [f for f in base if f in df.columns]
        cands = snow_features(base) or base
    return cands[: int(top)]


def robust_feature_importance(model: Any, df: pd.DataFrame,
                              features: Sequence[str] | None = None, *,
                              top: int = 8, method: str = "ale", bins: int = 20,
                              max_samples: int | None = 2000, seed: int = 0
                              ) -> pd.DataFrame:
    """Correlation-robust per-feature importance for the top drivers.

    Marginal SHAP / partial dependence extrapolate onto unrealistic feature
    combinations when the inputs are correlated and can credit the wrong driver.
    Two unbiased-under-correlation scores are offered:

    ``"ale"``
        The amplitude (max minus min) of the first-order accumulated-local-effects
        curve (Apley & Zhu, 2020) of :func:`sbc.explain.flagship_xai.ale`.  ALE
        accumulates only *local* effects within quantile bins, so it never
        evaluates the model off the data manifold.
    ``"conditional"``
        Conditional permutation importance: the feature is shuffled *within bins
        of its most-correlated partner* (Strobl et al., 2008), so the
        cross-feature correlation structure is preserved and only the feature's
        own conditional signal is destroyed.  Importance is the resulting rise in
        log-residual MSE.

    Parameters
    ----------
    model
        Fitted corrector with ``predict_residual``.
    df
        Modelling table.
    features : sequence of str, optional
        Shortlist to score; defaults to the present snow drivers (then any model
        feature), capped at ``top``.
    top : int, default 8
        Maximum number of features to score when ``features`` is ``None``.
    method : {"ale", "conditional"}, default "ale"
        Which correlation-robust estimator to use.
    bins : int, default 20
        Quantile bins for ALE / for the conditioning strata.
    max_samples : int, optional
        Row cap for the conditional-permutation MSE (``None`` = all rows).
    seed : int, default 0
        RNG seed for the conditional shuffles.

    Returns
    -------
    DataFrame
        Columns ``feature``, ``group``, ``importance`` and ``method``, sorted by
        importance descending.
    """
    method = str(method).lower()
    if method not in ("ale", "conditional"):
        raise ValueError("method must be 'ale' or 'conditional'")
    cands = _candidate_features(model, df, features, top)
    if not cands:
        raise ValueError("robust_feature_importance: no candidate features resolved")
    membership = group_membership(df, features=cands)

    if method == "ale":
        from .flagship_xai import ale

        rows = []
        for f in cands:
            try:
                a = ale(model, df, f, bins=bins, max_samples=max_samples, seed=seed)
                amp = float(np.nanmax(a["ale"]) - np.nanmin(a["ale"]))
            except Exception as exc:  # pragma: no cover - degenerate feature
                log.warning("ALE failed for %s (%s); importance=0", f, exc)
                amp = 0.0
            rows.append({"feature": f, "group": membership.get(f, "?"),
                         "importance": amp, "method": "ale"})
        out = pd.DataFrame(rows)
    else:
        out = _conditional_permutation(model, df, cands, membership, bins=bins,
                                       max_samples=max_samples, seed=seed)

    log.info("robust_feature_importance(%s): scored %d features", method, len(out))
    return out.sort_values("importance", ascending=False, ignore_index=True)


def _conditional_permutation(model: Any, df: pd.DataFrame, features: Sequence[str],
                             membership: Mapping[str, str], *, bins: int,
                             max_samples: int | None, seed: int) -> pd.DataFrame:
    """Conditional permutation importance (shuffle within a partner's strata)."""
    sub = _subsample(df, max_samples, seed) if max_samples is not None else df
    sub = sub.reset_index(drop=True)
    if TARGET_COL in sub.columns:
        y = sub[TARGET_COL].to_numpy(float)
    else:
        y = make_target(sub["q_obs"], sub["q_glofas"])

    def _mse(frame: pd.DataFrame) -> float:
        pred = np.asarray(model.predict_residual(frame), float)
        m = np.isfinite(pred) & np.isfinite(y)
        return float(np.mean((pred[m] - y[m]) ** 2)) if m.any() else np.nan

    base = _mse(sub)
    rng = np.random.default_rng(seed)
    pool = [f for f in (getattr(model, "features", None) or feature_columns(sub))
            if f in sub.columns]
    X = sub[pool].to_numpy(float) if pool else np.zeros((len(sub), 0))

    rows = []
    for f in features:
        partner = _most_correlated(sub, f, pool, X)
        shuffled = sub.copy()
        col = sub[f].to_numpy(float)
        if partner is None:
            shuffled[f] = rng.permutation(col)
        else:                              # permute within quantile strata of partner
            strata = _quantile_bins(sub[partner].to_numpy(float), bins)
            new = col.copy()
            for b in np.unique(strata):
                idx = np.where(strata == b)[0]
                if idx.size > 1:
                    new[idx] = col[rng.permutation(idx)]
            shuffled[f] = new
        rows.append({"feature": f, "group": membership.get(f, "?"),
                     "importance": float(_mse(shuffled) - base),
                     "method": "conditional"})
    return pd.DataFrame(rows)


def _quantile_bins(z: np.ndarray, bins: int) -> np.ndarray:
    """Integer bin index per value from up to ``bins`` quantile edges."""
    finite = z[np.isfinite(z)]
    if finite.size < 2:
        return np.zeros(len(z), dtype=int)
    edges = np.unique(np.nanquantile(finite, np.linspace(0, 1, int(bins) + 1)))
    if edges.size < 2:
        return np.zeros(len(z), dtype=int)
    return np.clip(np.searchsorted(edges, z, side="left"), 1, edges.size - 1)


def _most_correlated(df: pd.DataFrame, feature: str, pool: Sequence[str],
                     X: np.ndarray) -> str | None:
    """Return the pool feature most |Pearson|-correlated with ``feature``."""
    if feature not in pool or X.shape[1] == 0:
        return None
    j = list(pool).index(feature)
    z = X[:, j]
    best, best_r = None, 0.0
    for k, g in enumerate(pool):
        if g == feature:
            continue
        a, b = z, X[:, k]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            continue
        r = abs(float(np.corrcoef(a[m], b[m])[0, 1]))
        if np.isfinite(r) and r > best_r:
            best, best_r = g, r
    return best


# --------------------------------------------------------------------------- #
#  Plotting (Agg-safe, writes to results/figures)                              #
# --------------------------------------------------------------------------- #
def _figure_path(path: str | Path | None, default_name: str) -> Path:
    """Return a writable PNG path, defaulting under ``results/figures``."""
    if path is None:
        PATHS.figures.mkdir(parents=True, exist_ok=True)
        return PATHS.figures / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_group_bar(obj: Any, path: str | Path | None = None, *,
                   title: str | None = None) -> Path:
    """Write a horizontal bar chart of group importances to PNG.

    Accepts a :func:`group_shap` result dict, a :func:`group_importance` table,
    or a :class:`GroupStabilityResult` (in which case between-seed SD error bars
    and the audit scalars are drawn).

    Parameters
    ----------
    obj
        A ``group_shap`` result, a ``group_importance`` DataFrame or a
        :class:`GroupStabilityResult`.
    path : str or pathlib.Path, optional
        Destination PNG (defaults to ``results/figures/group_shap_<tag>.png``).
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

    err = None
    tag = "groups"
    if isinstance(obj, GroupStabilityResult):
        tbl = obj.table.copy()
        groups = tbl["group"].tolist()
        vals = tbl["mean_abs"].to_numpy(float)
        err = tbl["sd"].to_numpy(float)
        tag = obj.method.replace("/", "_")
        if title is None:
            title = (f"Group attribution stability ({obj.method}): "
                     f"Kendall-tau={obj.kendall_tau:.2f} "
                     f"(per-feature {obj.feature_kendall_tau:.2f}), "
                     f"Jaccard@{obj.top_k}={obj.jaccard_topk:.2f}")
    elif isinstance(obj, pd.DataFrame):
        tbl = obj
        groups = tbl["group"].tolist()
        col = "mean_abs" if "mean_abs" in tbl.columns else "importance"
        vals = tbl[col].to_numpy(float)
    else:                                  # group_shap result dict
        gi = group_importance(obj)
        groups = gi["group"].tolist()
        vals = gi["mean_abs"].to_numpy(float)
        tag = str(obj.get("method", "groups")).replace("/", "_")

    out = _figure_path(path, f"group_shap_{tag}.png")
    order = np.argsort(vals)               # smallest first -> largest on top
    groups = [groups[i] for i in order]
    vals = vals[order]
    err = err[order] if err is not None else None

    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(groups) + 1.8))
    ax.barh(y, vals, xerr=err, color="#1f77b4", ecolor="0.3", capsize=3,
            error_kw={"lw": 1.0})
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlabel("mean |group attribution|  (log-residual units)")
    ax.set_title(title or "Group-level feature attribution", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote group attribution bar -> %s", out)
    return out


# --------------------------------------------------------------------------- #
#  Self-test (tiny LightGBM + 3-epoch flagship on synthetic, <3 min)           #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models.boosting import LightGBMCorrector
    from ..models.regime_prob_net import RegimeProbNet
    from ..schemas import validate
    from ..synthetic import generate

    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(validate(df), scale="decadal")).reset_index(drop=True)
    cut = df["date"].quantile(0.7)
    train = df[df["date"] <= cut].copy()
    test = df[df["date"] > cut].copy()
    print(f"[group_causal_shap] rows={len(df)} gauges={df['code'].nunique()} "
          f"train={len(train)} test={len(test)}")

    # -- 1) discovered blocks ------------------------------------------------- #
    membership = group_membership(df)
    counts = pd.Series(membership).value_counts()
    print(f"[group_causal_shap] {len(membership)} features -> {counts.to_dict()}")

    # -- 2) tree group SHAP (auto -> aggregate/tree) -------------------------- #
    def factory(seed: int) -> LightGBMCorrector:
        return LightGBMCorrector(
            params=dict(n_estimators=80, num_leaves=15, learning_rate=0.1), seed=seed)

    tree = factory(0).fit(train, valid=test)
    res = group_shap(tree, test, max_samples=600, seed=0)
    gi = group_importance(res)
    print(f"[group_causal_shap] tree method={res['method']} group importances:")
    for _, r in gi.iterrows():
        print(f"    {r['group']:<14s} {r['mean_abs']:.4f}  ({r['share']*100:4.1f}%)")
    # tree SHAP is additive w.r.t. the *raw booster* output; predict_residual
    # additionally winsorises, so completeness is checked against the booster.
    raw_pred = np.asarray(tree.model.predict(res["X"]), float)
    completeness = float(np.abs(res["group_values"].sum(1) + res["base_value"]
                                - raw_pred).mean())
    print(f"[group_causal_shap] tree completeness |sum+base-booster|={completeness:.2e}")

    # -- 3) coalition (reduced-game) group Shapley on a small slice ----------- #
    coal = group_shap(tree, test, method="coalition", max_samples=150,
                      background=30, seed=0)
    coal_pred = np.asarray(tree.predict_residual(coal["X"]), float)
    coal_err = float(np.abs(coal["group_values"].sum(1) + coal["base_value"]
                            - coal_pred).mean())
    print(f"[group_causal_shap] coalition groups={coal['groups']} "
          f"completeness|sum+base-pred|={coal_err:.2e}")

    # -- 4) group-level seed stability vs per-feature (3 seeds) --------------- #
    gs = group_stability(factory, df, n_seeds=3, top_k=3, method="auto",
                         max_samples=600)
    print(f"[group_causal_shap] GROUP kendall_tau={gs.kendall_tau:+.3f} "
          f"(per-feature kendall_tau={gs.feature_kendall_tau:+.3f}, "
          f"baseline flagship ~0.27) | jaccard@{gs.top_k}={gs.jaccard_topk:.3f} "
          f"| seeds={gs.n_seeds}")

    # -- 5) flagship group SHAP via integrated gradients ---------------------- #
    flag = RegimeProbNet(K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                         epochs=3, batch_size=256, patience=5, lambda_gate=0.3,
                         lambda_phys=0.05, seed=0, verbose=False)
    flag.fit(train, valid=test)
    fres = group_shap(flag, test, max_samples=200, ig_steps=12, seed=0)
    fgi = group_importance(fres)
    top_grp = fgi["group"].iloc[0]
    print(f"[group_causal_shap] flagship method={fres['method']} "
          f"top_group={top_grp} importances={dict(zip(fgi['group'], fgi['mean_abs'].round(4)))}")

    # -- 6) correlation-robust per-feature (ALE amplitude) -------------------- #
    rb = robust_feature_importance(tree, test, top=6, method="ale", bins=12,
                                   max_samples=600)
    print("[group_causal_shap] ALE-robust per-feature importance (top drivers):")
    for _, r in rb.head(5).iterrows():
        print(f"    {r['feature']:<20s} [{r['group']:<13s}] {r['importance']:.4f}")

    # -- 7) figure ------------------------------------------------------------ #
    png = save_group_bar(gs)

    assert res["group_values"].shape[0] == len(res["X"])
    assert set(res["groups"]) <= set(membership.values())
    assert completeness < 1e-6 and coal_err < 1e-6
    assert np.isfinite(gs.kendall_tau) and -1.0 <= gs.kendall_tau <= 1.0
    assert fres["group_values"].shape[1] == len(fres["groups"])
    assert (rb["importance"] >= 0).all() and png.exists()
    print(f"[group_causal_shap] wrote {png.name} | SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
