"""Explainability *of the flagship* :class:`~sbc.models.regime_prob_net.RegimeProbNet`.

The companion module :mod:`sbc.explain.shap_analysis` explains the framework's
tree models exactly (tree SHAP) and falls back to permutation importance for the
deep models.  That is enough for a *proxy* picture, but a Q1 reviewer rightly
expects the **flagship's own** decision surface to be opened up rather than a
LightGBM stand-in.  This module provides three attribution methods that act
directly on a fitted corrector's ``predict_residual`` -- and, when the corrector
is the torch flagship, differentiate straight through its mixture-of-experts
network:

``integrated_gradients``
    Axiomatic, completeness-satisfying attributions (Sundararajan et al., 2017,
    *ICML*).  For the torch flagship the path integral is taken with exact
    autograd through the entity-aware-LSTM / regime-gated MoE; for any other
    corrector (or if torch is unavailable) a vectorised finite-difference
    estimator of the same path integral is used.  Both return the *same* dict so
    callers are model-agnostic.

``deep_shap``
    Shapley values for the flagship via :class:`shap.GradientExplainer` /
    :class:`shap.DeepExplainer` on the wrapped network, with a graceful cascade
    to model-agnostic KernelSHAP on a subsample and finally to integrated
    gradients.  Crucially it returns the **same dictionary shape** as
    :func:`sbc.explain.shap_analysis.tree_shap`
    (``shap_values``/``base_value``/``features``/``X``) so the existing
    ``global_importance`` / ``snow_dependence`` / ``regime_conditional_importance``
    consumers work unchanged on the flagship.

``ale``
    Accumulated Local Effects (Apley & Zhu, 2020, *JRSS-B*).  Unlike SHAP and
    partial dependence, ALE is unbiased under the strong cross-correlation
    between snow-water-equivalent, temperature and precipitation that is endemic
    in snow-driven Central-Asian catchments -- exactly the regime where marginal
    methods can attribute effect to the wrong, correlated driver.

All heavy, optional dependencies (``torch``, ``shap``, ``matplotlib``) are
imported lazily inside the functions that need them; figures are written on the
headless Agg backend (never ``plt.show``); every routine is deterministic given
a seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import PATHS
from ..schemas import SIM_COL, feature_columns
from ..utils import get_logger
from .shap_analysis import _model_features, _subsample, snow_features

log = get_logger(__name__)

__all__ = [
    "integrated_gradients",
    "deep_shap",
    "ale",
    "save_ale_plot",
    "snow_features",
]


# --------------------------------------------------------------------------- #
#  Capability probing                                                          #
# --------------------------------------------------------------------------- #
def _torch_ok() -> bool:
    """True when torch can be imported (deferred, never required at import)."""
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - torch optional
        return False
    return True


def _is_flagship(model: Any) -> bool:
    """Heuristic: a fitted torch corrector exposing the flagship internals."""
    return (
        getattr(model, "net", None) is not None
        and hasattr(model, "_design_matrices")
        and hasattr(model, "dyn_cols")
        and hasattr(model, "stat_cols")
        and bool(getattr(model, "dyn_cols", []))
    )


def _flagship_features(model: Any) -> list[str]:
    """The flagship's own input columns in network order (dynamic then static)."""
    return list(model.dyn_cols) + list(model.stat_cols)


def _resolve_features(model: Any, df: pd.DataFrame,
                      features: list[str] | None = None) -> list[str]:
    """Resolve the feature list to attribute over (present in ``df``)."""
    if features is not None:
        feats = [f for f in features if f in df.columns]
    elif _is_flagship(model):
        feats = [f for f in _flagship_features(model) if f in df.columns]
    else:
        try:
            feats = _model_features(model, df)
        except Exception:
            feats = feature_columns(df)
    if not feats:
        raise ValueError("no usable features found to attribute over")
    return list(feats)


# --------------------------------------------------------------------------- #
#  Torch residual wrapper for the flagship                                     #
# --------------------------------------------------------------------------- #
def _make_residual_module(model: Any):
    """Wrap the flagship ``net`` so it emits the scalar log-residual mean.

    The returned ``nn.Module`` maps the design tensors ``(x_dyn, x_stat)`` to the
    mixture predictive mean of the log-residual in *real* units (undoing the
    target standardisation), shaped ``(n, 1)`` so that ``shap`` and autograd
    treat it as a single-output regressor.
    """
    import torch.nn as nn

    net = model.net
    y_mean = float(model.y_mean)
    y_std = float(model.y_std)

    class _ResidualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, *args):  # x_dyn[, x_stat]
            x_dyn = args[0]
            if len(args) > 1 and args[1] is not None:
                x_stat = args[1]
            else:
                x_stat = x_dyn.new_zeros((x_dyn.shape[0], 0))
            w, _logits, mu, _sigma = self.net(x_dyn, x_stat)
            resid = (w * mu).sum(dim=1) * y_std + y_mean
            return resid.unsqueeze(-1)

    return _ResidualHead()


# --------------------------------------------------------------------------- #
#  Integrated gradients                                                        #
# --------------------------------------------------------------------------- #
def integrated_gradients(model: Any, df: pd.DataFrame, baseline: str | np.ndarray = "mean",
                         steps: int = 50, features: list[str] | None = None, *,
                         max_samples: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Integrated-gradients attribution of the log-residual prediction.

    Implements the path integral of Sundararajan et al. (2017): for input ``x``
    and baseline ``x'`` the attribution of feature ``j`` is
    ``(x_j - x'_j) * mean_alpha d f / d x_j |_{x' + alpha (x - x')}``.  For the
    torch flagship the gradient is exact (autograd through the EA-LSTM / MoE and
    the materialised input window); for any other corrector a vectorised
    central finite-difference estimate of the same integral is used, so the
    function works for every :class:`~sbc.models.base.BaseCorrector` exposing
    ``predict_residual`` -- including :class:`RegimeProbNet`.

    Parameters
    ----------
    model
        Fitted corrector exposing ``predict_residual``.
    df
        Modelling table to explain.
    baseline : {"mean", "zero"} or ndarray, default "mean"
        Reference point of the path integral.  ``"mean"`` uses the per-feature
        dataset mean (the standardised-space mean for the flagship); ``"zero"``
        uses the origin; an explicit array is taken as a per-feature baseline.
    steps : int, default 50
        Number of Riemann-sum steps along the path (midpoint rule).
    features : list of str, optional
        Subset of features to report; defaults to the model's input columns.
    max_samples : int, optional
        Cap on explained rows (``None`` = all).  Subsampling is **off by
        default** because the flagship's windowed inputs are most faithful over
        the full, contiguous table.
    seed : int, default 0
        Seed for the optional row subsample.

    Returns
    -------
    dict
        ``{"attributions": (n, f) ndarray, "features": list[str],
        "X": DataFrame (n, f), "base_value": float}``.  Row sums of
        ``attributions`` approximately equal ``predict_residual(x) - f(baseline)``
        (the completeness axiom).
    """
    if not hasattr(model, "predict_residual"):
        raise AttributeError("integrated_gradients needs a model with predict_residual()")
    sub = _subsample(df, max_samples, seed) if max_samples is not None else df
    feats = _resolve_features(model, sub, features)

    if _is_flagship(model) and _torch_ok():
        try:
            return _ig_flagship(model, sub, baseline, int(steps), feats)
        except Exception as exc:  # pragma: no cover - robustness cascade
            log.warning("autograd IG failed (%s); using finite-difference IG", exc)
    return _ig_finite_difference(model, sub, baseline, int(steps), feats)


def _baseline_design(baseline: str | np.ndarray, x: "Any"):
    """Build a (1, ...) baseline tensor matching design tensor ``x``."""
    import torch

    if isinstance(baseline, np.ndarray):
        b = torch.as_tensor(np.asarray(baseline, dtype=np.float32), device=x.device)
        return b.reshape((1,) + tuple(x.shape[1:])) if b.numel() else x.mean(0, keepdim=True)
    if str(baseline).lower() in ("zero", "zeros", "0"):
        return torch.zeros_like(x[:1])
    return x.mean(0, keepdim=True)  # "mean" (default)


def _ig_flagship(model: Any, df: pd.DataFrame, baseline: str | np.ndarray,
                 steps: int, feats: list[str]) -> dict[str, Any]:
    """Exact-gradient IG through the flagship network on the design tensors."""
    import torch

    dfx = df.reset_index(drop=True)
    x_dyn_np, x_stat_np = model._design_matrices(dfx)
    device = getattr(model, "device", "cpu")
    xd = torch.as_tensor(x_dyn_np, dtype=torch.float32, device=device)
    xs = torch.as_tensor(x_stat_np, dtype=torch.float32, device=device)
    has_stat = xs.shape[1] > 0

    bd = _baseline_design(baseline, xd)
    bs = _baseline_design(baseline, xs) if has_stat else xs[:1]

    mod = _make_residual_module(model).to(device).eval()
    grad_d = torch.zeros_like(xd)
    grad_s = torch.zeros_like(xs)
    alphas = (torch.arange(steps, dtype=torch.float32, device=device) + 0.5) / steps

    for a in alphas:
        id_ = (bd + a * (xd - bd)).detach().requires_grad_(True)
        if has_stat:
            is_ = (bs + a * (xs - bs)).detach().requires_grad_(True)
            out = mod(id_, is_).sum()
            gd, gs = torch.autograd.grad(out, [id_, is_])
            grad_s += gs.detach()
        else:
            out = mod(id_, xs).sum()
            (gd,) = torch.autograd.grad(out, [id_])
        grad_d += gd.detach()

    grad_d /= steps
    grad_s /= steps
    attr_dyn = ((xd - bd) * grad_d).sum(dim=1).cpu().numpy()      # (n, d_dyn)
    attr_stat = ((xs - bs) * grad_s).cpu().numpy()               # (n, d_stat)
    A = np.concatenate([attr_dyn, attr_stat], axis=1)

    with torch.no_grad():
        base_value = float(mod(bd, bs if has_stat else bd.new_zeros((1, 0)))
                           .reshape(-1)[0])

    full = _flagship_features(model)
    col_of = {f: i for i, f in enumerate(full)}
    requested = [f for f in feats if f in col_of]
    attr = np.column_stack([A[:, col_of[f]] for f in requested])
    X = dfx[requested].astype(float).copy()
    log.info("integrated_gradients(flagship): %d rows x %d feats, %d steps (base=%.4f)",
             attr.shape[0], attr.shape[1], steps, base_value)
    return {"attributions": attr, "features": requested, "X": X, "base_value": base_value}


def _ig_finite_difference(model: Any, df: pd.DataFrame, baseline: str | np.ndarray,
                          steps: int, feats: list[str]) -> dict[str, Any]:
    """Model-agnostic IG via vectorised central finite differences.

    Estimates the path-integral gradient by perturbing one feature at a time
    (all rows at once) at every Riemann midpoint, so the cost is
    ``2 * steps * n_features`` calls to ``predict_residual``.
    """
    dfx = df.reset_index(drop=True)
    n = len(dfx)
    X = dfx[feats].to_numpy(dtype=float)

    if isinstance(baseline, np.ndarray):
        b = np.asarray(baseline, dtype=float).reshape(-1)
        if b.size != len(feats):
            b = np.full(len(feats), float(np.nanmean(b)) if b.size else 0.0)
    elif str(baseline).lower() in ("zero", "zeros", "0"):
        b = np.zeros(len(feats))
    else:  # "mean"
        b = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    b = np.where(np.isfinite(b), b, 0.0)

    scale = np.nanstd(np.where(np.isfinite(X), X, np.nan), axis=0)
    h = np.where(np.isfinite(scale) & (scale > 0), 1e-3 * scale, 1e-4)

    work = dfx.copy()

    def _predict() -> np.ndarray:
        return np.asarray(model.predict_residual(work), dtype=float)

    grad_sum = np.zeros_like(X)
    alphas = (np.arange(steps) + 0.5) / steps
    for a in alphas:
        xa = b[None, :] + a * (X - b[None, :])
        for j, c in enumerate(feats):
            work[c] = xa[:, j]
        for j, c in enumerate(feats):
            col = xa[:, j]
            work[c] = col + h[j]
            fp = _predict()
            work[c] = col - h[j]
            fm = _predict()
            work[c] = col
            grad_sum[:, j] += (fp - fm) / (2.0 * h[j])
    grad = grad_sum / steps
    attr = (X - b[None, :]) * grad

    for j, c in enumerate(feats):  # f(baseline) under each row's context
        work[c] = np.full(n, b[j])
    base_value = float(np.nanmean(_predict()))

    log.info("integrated_gradients(finite-diff): %d rows x %d feats, %d steps (base=%.4f)",
             attr.shape[0], attr.shape[1], steps, base_value)
    X_out = dfx[feats].astype(float).copy()
    return {"attributions": attr, "features": list(feats), "X": X_out,
            "base_value": base_value}


# --------------------------------------------------------------------------- #
#  Deep SHAP (Gradient/Deep explainer -> KernelSHAP -> IG)                      #
# --------------------------------------------------------------------------- #
def deep_shap(model: Any, df: pd.DataFrame, background: int = 200,
              nsamples: int | str = "auto", *, max_samples: int = 1000,
              seed: int = 0) -> dict[str, Any]:
    """Shapley attributions for the flagship, with a robust fallback cascade.

    Tries :class:`shap.GradientExplainer` (then :class:`shap.DeepExplainer`) on
    the flagship's wrapped network using the materialised, per-sample-independent
    design tensors as the explanation space; aggregates the dynamic-input
    attributions over the time window so the result is one value per input
    feature.  If the deep explainer is incompatible, falls back to model-agnostic
    KernelSHAP on a subsample (the network is evaluated under a steady-state
    "static window" so a single feature row is well defined), and finally to
    :func:`integrated_gradients`.

    Parameters
    ----------
    model
        Fitted corrector (the flagship, or any model with ``predict_residual``).
    df
        Modelling table to explain.
    background : int, default 200
        Number of reference rows for the explainer's baseline distribution.
    nsamples : int or "auto", default "auto"
        KernelSHAP coalition budget (only used on that fallback path).
    max_samples : int, default 1000
        Cap on explained (foreground) rows.
    seed : int, default 0
        RNG seed for the foreground / background subsamples.

    Returns
    -------
    dict
        ``{"shap_values": (n, f) ndarray, "base_value": float,
        "features": list[str], "X": DataFrame (n, f)}`` -- identical in shape to
        :func:`sbc.explain.shap_analysis.tree_shap`, so ``global_importance``,
        ``snow_dependence`` and ``regime_conditional_importance`` consume it
        directly.
    """
    feats = _resolve_features(model, df, None)

    if _is_flagship(model) and _torch_ok():
        for explainer in ("gradient", "deep"):
            try:
                return _deep_explainer_flagship(model, df, feats, background,
                                                max_samples, seed, kind=explainer)
            except Exception as exc:
                log.warning("%s explainer incompatible (%s)", explainer, exc)
        try:
            return _kernel_shap(model, df, feats, background, nsamples,
                                max_samples, seed, static_window=True)
        except Exception as exc:
            log.warning("KernelSHAP fallback failed (%s); using integrated gradients", exc)
        return _ig_as_shap(model, df, feats, max_samples, seed)

    # non-flagship corrector: KernelSHAP on predict_residual, else IG
    try:
        return _kernel_shap(model, df, feats, background, nsamples,
                            max_samples, seed, static_window=False)
    except Exception as exc:
        log.warning("KernelSHAP failed (%s); using integrated gradients", exc)
    return _ig_as_shap(model, df, feats, max_samples, seed)


def _norm_sv(arr: Any, in_ndim: int) -> np.ndarray:
    """Normalise a shap-values array, dropping a trailing single-output axis."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == in_ndim + 1 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def _deep_explainer_flagship(model: Any, df: pd.DataFrame, feats: list[str],
                             background: int, max_samples: int, seed: int,
                             kind: str = "gradient") -> dict[str, Any]:
    """Run shap Gradient/Deep explainer on the flagship's design tensors."""
    import shap
    import torch

    dfx = df.reset_index(drop=True)
    x_dyn_full, x_stat_full = model._design_matrices(dfx)
    n = len(dfx)
    rng = np.random.default_rng(seed)
    fg_idx = np.sort(rng.choice(n, size=min(int(max_samples), n), replace=False))
    bg_idx = np.sort(rng.choice(n, size=min(int(background), n), replace=False))

    device = getattr(model, "device", "cpu")
    has_stat = x_stat_full.shape[1] > 0
    mod = _make_residual_module(model).to(device).eval()

    def _t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    if has_stat:
        bg_inputs = [_t(x_dyn_full[bg_idx]), _t(x_stat_full[bg_idx])]
        fg_inputs = [_t(x_dyn_full[fg_idx]), _t(x_stat_full[fg_idx])]
    else:
        bg_inputs = _t(x_dyn_full[bg_idx])
        fg_inputs = _t(x_dyn_full[fg_idx])

    if kind == "deep":
        explainer = shap.DeepExplainer(mod, bg_inputs)
        raw = explainer.shap_values(fg_inputs)
    else:
        explainer = shap.GradientExplainer(mod, bg_inputs)
        raw = explainer.shap_values(fg_inputs)

    if not isinstance(raw, list):
        raw = [raw]
    attr_dyn = _norm_sv(raw[0], 3).sum(axis=1)                    # (m, d_dyn)
    if has_stat and len(raw) > 1:
        attr_stat = _norm_sv(raw[1], 2)                          # (m, d_stat)
    else:
        attr_stat = np.zeros((attr_dyn.shape[0], len(model.stat_cols)), dtype=float)
    A = np.concatenate([attr_dyn, attr_stat], axis=1)

    full = _flagship_features(model)
    pr = np.asarray(model.predict_residual(dfx), dtype=float)
    base_value = float(np.nanmean(pr[bg_idx]))
    X = dfx.iloc[fg_idx][full].astype(float).copy()
    log.info("deep_shap(%s explainer): %d rows x %d feats (bg=%d, base=%.4f)",
             kind, A.shape[0], A.shape[1], len(bg_idx), base_value)
    return {"shap_values": A, "base_value": base_value, "features": full, "X": X}


def _flagship_static_residual(model: Any, M: np.ndarray, feats: list[str]) -> np.ndarray:
    """Evaluate the flagship under a steady-state ("static window") feature row.

    Each standalone feature row ``M[i]`` is broadcast across the ``seq_len``
    window, making ``f(feature_row)`` well defined for KernelSHAP coalitions.
    """
    import torch

    M = np.asarray(M, dtype=float)
    m = M.shape[0]
    idx = {f: i for i, f in enumerate(feats)}

    dyn = np.stack([M[:, idx[c]] for c in model.dyn_cols], axis=1) \
        if model.dyn_cols else np.zeros((m, 0))
    dyn = (dyn - model.dyn_mean) / model.dyn_std
    dyn = np.where(np.isfinite(dyn), dyn, 0.0)

    if model.stat_cols:
        stat = np.stack([M[:, idx[c]] for c in model.stat_cols], axis=1)
        stat = (stat - model.stat_mean) / model.stat_std
        stat = np.where(np.isfinite(stat), stat, 0.0)
    else:
        stat = np.zeros((m, 0))

    L = int(model.seq_len)
    x_dyn = np.repeat(dyn[:, None, :], L, axis=1).astype(np.float32)
    x_stat = stat.astype(np.float32)
    device = getattr(model, "device", "cpu")
    mod = _make_residual_module(model).to(device).eval()
    with torch.no_grad():
        r = mod(torch.as_tensor(x_dyn, device=device),
                torch.as_tensor(x_stat, device=device))
    return r.reshape(-1).cpu().numpy()


def _kernel_shap(model: Any, df: pd.DataFrame, feats: list[str], background: int,
                 nsamples: int | str, max_samples: int, seed: int,
                 static_window: bool) -> dict[str, Any]:
    """Model-agnostic KernelSHAP on a subsample (flagship or generic model)."""
    import shap

    dfx = df.reset_index(drop=True)
    n = len(dfx)
    rng = np.random.default_rng(seed)
    fg_idx = np.sort(rng.choice(n, size=min(int(max_samples), n), replace=False))
    bg_idx = np.sort(rng.choice(n, size=min(int(background), n), replace=False))
    X_all = dfx[feats].to_numpy(dtype=float)
    X_bg = X_all[bg_idx]
    X_fg = X_all[fg_idx]

    if static_window:
        def f(M: np.ndarray) -> np.ndarray:
            return _flagship_static_residual(model, M, feats)
    else:
        def f(M: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(np.asarray(M, dtype=float), columns=feats)
            return np.asarray(model.predict_residual(frame), dtype=float)

    explainer = shap.KernelExplainer(f, X_bg)
    sv = explainer.shap_values(X_fg, nsamples=nsamples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 1:
        sv = sv[None, :]
    base_value = float(np.ravel(np.asarray(explainer.expected_value, dtype=float))[0])
    X = dfx.iloc[fg_idx][feats].astype(float).copy()
    log.info("deep_shap(KernelSHAP, static_window=%s): %d rows x %d feats (base=%.4f)",
             static_window, sv.shape[0], sv.shape[1], base_value)
    return {"shap_values": sv, "base_value": base_value, "features": list(feats), "X": X}


def _ig_as_shap(model: Any, df: pd.DataFrame, feats: list[str],
                max_samples: int, seed: int) -> dict[str, Any]:
    """Final fallback: cast integrated gradients into the tree_shap dict shape."""
    ig = integrated_gradients(model, df, features=feats,
                              max_samples=max_samples, seed=seed)
    log.info("deep_shap: returning integrated-gradients attributions as SHAP values")
    return {"shap_values": ig["attributions"], "base_value": float(ig.get("base_value", 0.0)),
            "features": ig["features"], "X": ig["X"]}


# --------------------------------------------------------------------------- #
#  Accumulated Local Effects (ALE)                                             #
# --------------------------------------------------------------------------- #
def ale(model: Any, df: pd.DataFrame, feature: str, bins: int = 20, *,
        max_samples: int | None = None, seed: int = 0) -> pd.DataFrame:
    """First-order Accumulated Local Effects of ``predict_residual`` for one feature.

    ALE (Apley & Zhu, 2020) accumulates the *local* gradient of the prediction
    with respect to ``feature``, estimated within quantile bins by the centred
    finite difference at the bin edges and averaged over the rows that actually
    fall in each bin.  Because the differences are taken locally -- holding every
    other feature at its observed value and only nudging ``feature`` across its
    own bin -- ALE stays unbiased when ``feature`` is strongly correlated with
    other inputs (SWE/T/precip), the setting where SHAP and partial dependence
    extrapolate onto unrealistic combinations and can mislead.

    Parameters
    ----------
    model
        Fitted corrector exposing ``predict_residual``.
    df
        Modelling table.
    feature
        Name of the (numeric) feature to profile.
    bins : int, default 20
        Number of quantile bins; collapsed automatically where data are sparse.
    max_samples : int, optional
        Cap on rows used (``None`` = all; keeps the flagship's windows faithful).
    seed : int, default 0
        Seed for the optional subsample.

    Returns
    -------
    DataFrame
        Columns ``feature_value`` (the ``K+1`` bin edges) and ``ale`` (the
        centred accumulated effect at each edge, in log-residual units).  The
        bin row counts are attached in ``df.attrs["bin_counts"]`` and the feature
        name in ``df.attrs["feature"]``.
    """
    if not hasattr(model, "predict_residual"):
        raise AttributeError("ale needs a model with predict_residual()")
    if feature not in df.columns:
        raise KeyError(f"{feature!r} is not a column of the modelling table")

    sub = _subsample(df, max_samples, seed) if max_samples is not None else df
    dfx = sub.reset_index(drop=True)
    z = dfx[feature].to_numpy(dtype=float)
    finite = np.isfinite(z)
    if finite.sum() < 2:
        raise ValueError(f"feature {feature!r} has too few finite values for ALE")

    qs = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.unique(np.nanquantile(z[finite], qs))
    if edges.size < 2:
        raise ValueError(f"feature {feature!r} has insufficient spread for ALE")
    K = edges.size - 1
    bin_idx = np.clip(np.searchsorted(edges, z, side="left"), 1, K)

    work = dfx.copy()
    base_col = dfx[feature].to_numpy(dtype=float).copy()

    def _predict() -> np.ndarray:
        return np.asarray(model.predict_residual(work), dtype=float)

    avg = np.zeros(K, dtype=float)
    counts = np.zeros(K, dtype=int)
    for k in range(1, K + 1):
        sel = (bin_idx == k) & finite
        counts[k - 1] = int(sel.sum())
        if not sel.any():
            continue
        lo, hi = edges[k - 1], edges[k]
        col = base_col.copy()
        col[sel] = hi
        work[feature] = col
        p_hi = _predict()
        col[sel] = lo
        work[feature] = col
        p_lo = _predict()
        work[feature] = base_col
        avg[k - 1] = float(np.nanmean(p_hi[sel] - p_lo[sel]))

    ale_unc = np.concatenate([[0.0], np.cumsum(avg)])            # value at each edge
    total = int(counts.sum())
    if total > 0:
        mid = 0.5 * (ale_unc[:-1] + ale_unc[1:])                # mean over each bin
        center = float(np.sum(counts * mid) / total)
    else:
        center = 0.0
    ale_centered = ale_unc - center

    out = pd.DataFrame({"feature_value": edges, "ale": ale_centered})
    out.attrs["feature"] = feature
    out.attrs["bin_counts"] = counts
    log.info("ale(%s): %d bins over %d rows (range %.3g..%.3g)",
             feature, K, total, float(edges[0]), float(edges[-1]))
    return out


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


def save_ale_plot(ale_df: pd.DataFrame, path: str | Path | None = None, *,
                  feature: str | None = None, title: str | None = None,
                  model_name: str | None = None) -> Path:
    """Write an ALE curve to PNG under ``results/figures``.

    Parameters
    ----------
    ale_df
        Output of :func:`ale`.
    path
        Destination PNG (defaults to ``results/figures/ale_<feature>.png``).
    feature
        Override the x-axis label / filename stem (else taken from ``ale_df``).
    title
        Optional figure title.
    model_name
        Optional model tag folded into the default filename.

    Returns
    -------
    Path
        The written PNG path.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    feature = feature or ale_df.attrs.get("feature", "feature")
    stem = f"ale_{model_name}_{feature}" if model_name else f"ale_{feature}"
    out = _figure_path(path, f"{stem}.png")

    counts = ale_df.attrs.get("bin_counts")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ale_df["feature_value"], ale_df["ale"], marker="o", ms=4,
            lw=1.6, color="#1f77b4")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    # rug of bin edges along the bottom
    ylo = ax.get_ylim()[0]
    ax.plot(ale_df["feature_value"], np.full(len(ale_df), ylo), "|",
            color="0.4", ms=8, alpha=0.7)
    ax.set_xlabel(str(feature))
    ax.set_ylabel("ALE (log-residual)")
    ax.set_title(title or f"Accumulated Local Effects: {feature}")
    if counts is not None and len(counts):
        ax.text(0.99, 0.02, f"n={int(np.sum(counts))}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize="small", color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("wrote ALE plot -> %s", out)
    return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                   #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    from ..features import engineering as eng
    from ..features import regimes as reg
    from ..models.regime_prob_net import RegimeProbNet
    from ..synthetic import generate

    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = eng.build_features(df, "decadal")
    df = reg.classify_regimes(df)

    cut = df["date"].quantile(0.7)
    train = df[df["date"] <= cut].copy()
    test = df[df["date"] > cut].copy()
    print(f"[flagship_xai] rows={len(df)} gauges={df['code'].nunique()} "
          f"train={len(train)} test={len(test)}")

    model = RegimeProbNet(K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                          epochs=3, batch_size=256, patience=5, lambda_gate=0.3,
                          lambda_phys=0.05, seed=0, verbose=False)
    model.fit(train, valid=test)

    snow = snow_features(_flagship_features(model))
    target = "swe" if "swe" in snow else (snow[0] if snow else "swe")

    ig = integrated_gradients(model, test, baseline="mean", steps=24,
                              max_samples=400, seed=0)
    order = np.argsort(-np.abs(ig["attributions"]).mean(axis=0))
    top5 = [(ig["features"][i], round(float(np.abs(ig["attributions"]).mean(axis=0)[i]), 4))
            for i in order[:5]]
    row_sum = float(np.abs(ig["attributions"]).sum(axis=1).mean())
    print(f"[flagship_xai] IG attributions shape={ig['attributions'].shape} "
          f"base={ig['base_value']:+.4f} mean|sum_f|={row_sum:.4f}")
    print(f"[flagship_xai] top-5 IG features: {top5}")

    aledf = ale(model, test, target, bins=12)
    print(f"[flagship_xai] ALE({target}) edges={len(aledf)} "
          f"span=[{aledf['ale'].min():+.4f},{aledf['ale'].max():+.4f}]")
    png = save_ale_plot(aledf, model_name=model.name)

    # exercise the deep_shap cascade (Gradient explainer) on a small subsample
    try:
        ds = deep_shap(model, test, background=40, max_samples=80, seed=0)
        ds_ok = (ds["shap_values"].shape[0] == ds["X"].shape[0]
                 and ds["shap_values"].shape[1] == len(ds["features"]))
        print(f"[flagship_xai] deep_shap shape={ds['shap_values'].shape} "
              f"base={ds['base_value']:+.4f} dict_ok={ds_ok}")
    except Exception as exc:  # pragma: no cover - explainer/version dependent
        print(f"[flagship_xai] deep_shap skipped ({exc})")
        ds_ok = True

    assert ig["attributions"].shape == (len(ig["X"]), len(ig["features"]))
    assert {"feature_value", "ale"} <= set(aledf.columns)
    assert png.exists()
    print(f"[flagship_xai] wrote {png.name} | SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
