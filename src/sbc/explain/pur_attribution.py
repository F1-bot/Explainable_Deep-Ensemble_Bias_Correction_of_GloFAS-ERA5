"""Attribute-resolved diagnosis of *where* and *why* PUR generalisation fails.

The headline of the prediction-in-ungauged-regions (PUR) experiment is a single
weak number -- the flagship's median PUR KGE' drops far below its temporal-holdout
skill on the fully held-out Amu-Darya transfer domain.  A Q1 reviewer will ask the
obvious follow-up: *which* ungauged catchments break the correction, and what is it
about them that the model cannot transfer to?  This module turns that weak number
into science by regressing the *per-gauge* PUR failure of any model onto the static
HydroATLAS catchment attributes, and by showing that the deep ensemble's own
between-seed disagreement is an a-priori flag for the very gauges it then gets
wrong.

Two analyses are provided:

``attribute_regression``
    Reads a per-gauge result table (``results/tables/per_gauge_*.parquet``) and the
    processed static-attribute table, joins them by gauge ``code``, and regresses a
    per-gauge *failure* target -- the correction skill gain ``delta_kge`` and the
    residual volume bias ``|PBIAS|`` by default -- on a small, physically meaningful
    set of catchment attributes (glacier fraction, drainage area, mean elevation,
    snow fraction, aridity).  It reports, per attribute, the univariate Spearman and
    Pearson association with failure *and* a standardised multivariate ridge
    coefficient, so the paper can state which catchment properties predict where the
    correction fails to transfer.  Attribute names are resolved through documented
    aliases (synthetic ``glacier_frac`` / real HydroATLAS ``gl_fr`` etc.) so the
    same call runs on the real and the synthetic tables; nothing is hard-coded.

``epistemic_vs_error``
    Correlates the deep ensemble's *between-seed* (epistemic) variance with the
    per-gauge error and, when an out-of-support flag is supplied, quantifies how
    well that disagreement separates the transfer (PUR) gauges from the in-support
    ones (rank-correlation + AUC).  A positive correlation is the evidence that the
    ensemble *knows what it does not know*: member disagreement rises precisely on
    the out-of-support catchments, so it can serve as an operational, label-free
    reliability flag.  :func:`between_seed_variance` derives that per-gauge variance
    from a fitted :class:`~sbc.models.robust.DeepEnsembleCorrector` without re-running
    the validation matrix.

All heavy / optional imports (scipy, matplotlib, torch via the ensemble) are kept
inside the functions that need them; figures are written on the headless Agg
backend (never ``plt.show``); tables go to ``results/tables`` and figures to
``results/figures`` with a ``pur_`` prefix.  The skill helpers reuse
:mod:`sbc.validation.metrics` and never hard-code a feature list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Attribute vocabulary (canonical name <- documented aliases)                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttributeSpec:
    """One catchment attribute used as a PUR-failure predictor.

    Parameters
    ----------
    name : str
        Canonical attribute name used in the output tables and figures.
    aliases : tuple of str
        Candidate source-column names, tried in order; the first present in the
        static table is used.  This is what lets one call run unchanged on the
        synthetic table (``glacier_frac`` / ``elev_m`` / ``snow_frac``) and the
        real HydroATLAS table (``gl_fr`` / ``h_mean`` / ``fs_ann_chelsa``).
    transform : str
        ``"none"`` or ``"log10"`` (applied to the raw column before regression);
        heavy-tailed attributes such as drainage area are log-scaled so the linear
        association is meaningful.
    label : str
        Human-readable axis label for the figures.
    """

    name: str
    aliases: tuple[str, ...]
    transform: str = "none"
    label: str = ""


#: default predictor set -- the physically interpretable HydroATLAS attributes the
#: snow-hydrology literature links to GloFAS transferability.  Resolved through the
#: aliases so synthetic and real static tables both work.
DEFAULT_ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec("glacier_frac", ("glacier_frac", "gl_fr", "gla_pc_sse"),
                  "none", "glacier fraction [-]"),
    AttributeSpec("area_km2", ("area_km2", "area", "up_area", "uparea_km2"),
                  "log10", "log10 drainage area [km2]"),
    AttributeSpec("elevation", ("elev_m", "h_mean", "elevation", "ele_mt_sav"),
                  "none", "mean elevation [m]"),
    AttributeSpec("snow_frac", ("snow_frac", "fs_ann_chelsa", "scd", "snw_pc_syr"),
                  "none", "snow fraction / cover [-]"),
    AttributeSpec("aridity", ("aridity", "ai_ann_cgiar", "ai_ann_bio_chelsa", "ari_ix_sav"),
                  "none", "aridity index [-]"),
)

#: failure-target definitions: ``good`` is +1 when *larger is better* (so a negative
#: association means the attribute predicts failure) and -1 when larger is worse.
TARGET_DEFS: dict[str, dict[str, Any]] = {
    "delta_kge": {"good": +1, "label": "delta-KGE' (corrected - raw)"},
    "kge": {"good": +1, "label": "corrected KGE'"},
    "abs_pbias": {"good": -1, "label": "|PBIAS| corrected [%]"},
    "d_abs_pbias": {"good": +1, "label": "|PBIAS| reduction [%]"},
}

#: minimum finite gauge pairs an attribute needs to report an association
MIN_GAUGES: int = 4

__all__ = [
    "AttributeSpec",
    "DEFAULT_ATTRIBUTES",
    "TARGET_DEFS",
    "resolve_attributes",
    "gauge_failure_table",
    "attribute_regression",
    "between_seed_variance",
    "epistemic_vs_error",
    "plot_attribute_regression",
    "plot_attribute_scatter",
    "plot_epistemic_vs_error",
    "build_all",
]


# --------------------------------------------------------------------------- #
#  Static-attribute resolution                                                #
# --------------------------------------------------------------------------- #
def _as_specs(attributes: Sequence[Any] | None) -> tuple[AttributeSpec, ...]:
    """Normalise the ``attributes`` argument to a tuple of :class:`AttributeSpec`.

    Accepts ``None`` (use :data:`DEFAULT_ATTRIBUTES`), a sequence of
    :class:`AttributeSpec`, or a sequence of canonical names selecting a subset of
    the defaults.
    """
    if attributes is None:
        return DEFAULT_ATTRIBUTES
    specs: list[AttributeSpec] = []
    by_name = {s.name: s for s in DEFAULT_ATTRIBUTES}
    for a in attributes:
        if isinstance(a, AttributeSpec):
            specs.append(a)
        elif isinstance(a, str) and a in by_name:
            specs.append(by_name[a])
        elif isinstance(a, str):
            specs.append(AttributeSpec(a, (a,), "none", a))
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported attribute spec {a!r}")
    return tuple(specs)


def _apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    """Apply ``"none"`` / ``"log10"`` to a raw attribute column."""
    x = np.asarray(x, float)
    if transform == "log10":
        return np.log10(np.clip(x, 1e-6, None))
    return x


def resolve_attributes(static: pd.DataFrame, attributes: Sequence[Any] | None = None,
                       *, code_col: str = "code") -> pd.DataFrame:
    """Resolve catchment attributes from a static table by alias, one row per gauge.

    Parameters
    ----------
    static : pandas.DataFrame
        Processed static-attribute table with a gauge ``code`` column and the
        HydroATLAS (or synthetic) attribute columns.
    attributes : sequence, optional
        Predictor specification (see :func:`_as_specs`); defaults to
        :data:`DEFAULT_ATTRIBUTES`.
    code_col : str, default ``"code"``
        Name of the gauge-identifier column.

    Returns
    -------
    pandas.DataFrame
        Columns ``code`` plus one *canonical* column per resolved attribute (after
        its transform).  Attributes whose aliases are all absent are skipped.  The
        resolution map (canonical -> source column / transform / label) is stored in
        ``df.attrs["resolved"]``.
    """
    if code_col not in static.columns:
        raise KeyError(f"static table missing the gauge id column {code_col!r}")
    specs = _as_specs(attributes)
    out = pd.DataFrame({"code": static[code_col].astype(str).to_numpy()})
    resolved: dict[str, dict[str, str]] = {}
    for spec in specs:
        src = next((c for c in spec.aliases if c in static.columns), None)
        if src is None:
            log.debug("resolve_attributes: no alias of %s present (%s); skipping",
                      spec.name, spec.aliases)
            continue
        col = pd.to_numeric(static[src], errors="coerce").to_numpy(float)
        out[spec.name] = _apply_transform(col, spec.transform)
        resolved[spec.name] = {"source": src, "transform": spec.transform,
                               "label": spec.label or spec.name}
    out = out.dropna(axis=0, how="all", subset=[c for c in out.columns if c != "code"])
    out.attrs["resolved"] = resolved
    log.info("resolve_attributes: %d/%d attributes resolved (%s)",
             len(resolved), len(specs), ", ".join(resolved))
    return out


# --------------------------------------------------------------------------- #
#  Per-gauge failure table                                                    #
# --------------------------------------------------------------------------- #
def gauge_failure_table(per_gauge: pd.DataFrame, static: pd.DataFrame, *,
                        model: str, split: str, attributes: Sequence[Any] | None = None,
                        ) -> pd.DataFrame:
    """Join per-gauge skill (one model/split) with resolved catchment attributes.

    Parameters
    ----------
    per_gauge : pandas.DataFrame
        A ``per_gauge_<tag>`` table (see :func:`sbc.validation.cv.run_matrix`) with
        ``model`` / ``split`` / ``code`` / ``basin`` and the ``kge`` / ``kge_raw`` /
        ``pbias`` / ``pbias_raw`` columns.
    static : pandas.DataFrame
        Static-attribute table (see :func:`resolve_attributes`).
    model, split : str
        The model name and validation split to analyse (e.g. ``"regimeprobnet"`` /
        ``"pur"``).
    attributes : sequence, optional
        Predictor specification; defaults to :data:`DEFAULT_ATTRIBUTES`.

    Returns
    -------
    pandas.DataFrame
        One row per gauge, carrying the skill columns, the derived failure targets
        (``delta_kge``, ``kge``, ``abs_pbias``, ``d_abs_pbias``) and the resolved
        attribute columns.  The attribute resolution map is preserved in
        ``df.attrs["resolved"]``.
    """
    need = {"model", "split", "code", "kge", "kge_raw", "pbias", "pbias_raw"}
    miss = need - set(per_gauge.columns)
    if miss:
        raise KeyError(f"per_gauge table missing columns {sorted(miss)}")
    sub = per_gauge[(per_gauge["model"] == model) & (per_gauge["split"] == split)].copy()
    if sub.empty:
        log.warning("gauge_failure_table: no rows for model=%s split=%s", model, split)
        return pd.DataFrame()
    sub["code"] = sub["code"].astype(str)

    metric_cols = [c for c in ("kge", "kge_raw", "nse", "nse_raw", "pbias", "pbias_raw",
                               "lognse", "crps", "crps_raw", "peak_timing_err", "n")
                   if c in sub.columns]
    # several folds (LOBO) can contribute the same gauge -> per-gauge median
    keep_first = [c for c in ("basin", "domain") if c in sub.columns]
    agg = {c: "median" for c in metric_cols}
    agg.update({c: "first" for c in keep_first})
    g = sub.groupby("code", as_index=False).agg(agg)

    g["delta_kge"] = g["kge"] - g["kge_raw"]
    g["abs_pbias"] = g["pbias"].abs()
    g["abs_pbias_raw"] = g["pbias_raw"].abs()
    g["d_abs_pbias"] = g["abs_pbias_raw"] - g["abs_pbias"]
    g["model"] = model
    g["split"] = split

    attr = resolve_attributes(static, attributes)
    merged = g.merge(attr, on="code", how="inner")
    merged.attrs["resolved"] = attr.attrs.get("resolved", {})
    if merged.empty:
        log.warning("gauge_failure_table: zero gauges after joining static attributes "
                    "(code overlap=%d)", len(set(g["code"]) & set(attr["code"])))
    else:
        log.info("gauge_failure_table: model=%s split=%s -> %d gauges with attributes",
                 model, split, len(merged))
    return merged


# --------------------------------------------------------------------------- #
#  Statistics helpers                                                         #
# --------------------------------------------------------------------------- #
def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Spearman and Pearson correlation with NaN handling and small-n guards."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = {"n": int(x.size), "spearman_r": np.nan, "spearman_p": np.nan,
           "pearson_r": np.nan, "pearson_p": np.nan}
    if x.size < MIN_GAUGES or x.std() == 0 or y.std() == 0:
        return out
    from scipy.stats import pearsonr, spearmanr

    with np.errstate(invalid="ignore", divide="ignore"):
        sr = spearmanr(x, y)
        pr = pearsonr(x, y)
    out["spearman_r"] = float(sr.correlation if hasattr(sr, "correlation") else sr[0])
    out["spearman_p"] = float(sr.pvalue if hasattr(sr, "pvalue") else sr[1])
    out["pearson_r"] = float(pr[0])
    out["pearson_p"] = float(pr[1])
    return out


def _ridge_standardised(X: np.ndarray, y: np.ndarray, alpha: float = 1.0
                        ) -> tuple[np.ndarray, float, float, int]:
    """Standardised ridge coefficients and R^2 of a multivariate fit.

    Both predictors and target are z-scored, so the returned coefficients are
    directly comparable in magnitude (effect of a 1-SD change in each attribute on
    the target, in target SDs).  Ridge (small ``alpha``) tames the strong
    glacier/elevation/snow collinearity endemic to mountain catchments and keeps the
    fit defined when gauges are few.

    Returns
    -------
    (beta, r2, adj_r2, n) : tuple
        Standardised coefficients (``NaN`` where a predictor is constant), the
        coefficient of determination, the adjusted R^2 and the number of gauges used.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    n, p = X.shape
    beta = np.full(p, np.nan)
    if n < 3 or y.std() == 0:
        return beta, np.nan, np.nan, n
    sd = X.std(axis=0)
    good = sd > 0
    Xz = np.zeros_like(X)
    Xz[:, good] = (X[:, good] - X[:, good].mean(axis=0)) / sd[good]
    yz = (y - y.mean()) / y.std()
    Xg = Xz[:, good]
    A = Xg.T @ Xg + alpha * np.eye(Xg.shape[1])
    bg = np.linalg.solve(A, Xg.T @ yz)
    beta[good] = bg
    yhat = Xg @ bg
    ss_res = float(np.sum((yz - yhat) ** 2))
    ss_tot = float(np.sum((yz - yz.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj = (1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
           if np.isfinite(r2) and n - p - 1 > 0 else np.nan)
    return beta, float(r2), float(adj) if adj is not None else np.nan, n


# --------------------------------------------------------------------------- #
#  Attribute regression (the headline analysis)                               #
# --------------------------------------------------------------------------- #
def attribute_regression(per_gauge: pd.DataFrame, static: pd.DataFrame, *,
                         model: str = "regimeprobnet", split: str = "pur",
                         targets: Sequence[str] = ("delta_kge", "abs_pbias"),
                         attributes: Sequence[Any] | None = None,
                         ridge_alpha: float = 1.0) -> pd.DataFrame:
    """Regress per-gauge PUR failure on catchment attributes and rank the predictors.

    For each requested *failure target* the routine (i) computes the univariate
    Spearman and Pearson association of every resolved attribute with the target and
    (ii) fits one standardised multivariate ridge model of the target on all
    attributes jointly.  It returns a tidy table that answers the paper's question:
    which catchment properties predict where the correction fails to transfer to the
    ungauged domain?

    Parameters
    ----------
    per_gauge : pandas.DataFrame
        A ``per_gauge_<tag>`` result table (see :func:`gauge_failure_table`).
    static : pandas.DataFrame
        Static-attribute table (see :func:`resolve_attributes`).
    model : str, default ``"regimeprobnet"``
        Model whose per-gauge failure is explained.
    split : str, default ``"pur"``
        Validation split to analyse (``"pur"`` is the headline; ``"temporal"`` gives
        an in-support contrast).
    targets : sequence of str, default ``("delta_kge", "abs_pbias")``
        Failure targets from :data:`TARGET_DEFS` -- the correction skill gain and the
        residual volume bias by default.
    attributes : sequence, optional
        Predictor specification; defaults to :data:`DEFAULT_ATTRIBUTES`.
    ridge_alpha : float, default 1.0
        Ridge penalty on the z-scored multivariate fit.

    Returns
    -------
    pandas.DataFrame
        One row per (target, attribute) with ``n_gauges``, the attribute mean/SD, the
        univariate ``spearman_r`` / ``spearman_p`` / ``pearson_r`` / ``pearson_p``,
        the standardised multivariate ``coef_std``, a within-target ``rank`` by
        absolute Spearman association, a ``predicts`` flag (``"failure"`` /
        ``"success"``) and the model-level ``r2`` / ``adj_r2``.  The per-gauge failure
        table is attached as ``df.attrs["failure_table"]`` and the resolution map as
        ``df.attrs["resolved"]``.  Empty if the join yields no gauges.
    """
    ft = gauge_failure_table(per_gauge, static, model=model, split=split,
                             attributes=attributes)
    if ft.empty:
        return pd.DataFrame()
    resolved = ft.attrs.get("resolved", {})
    attr_cols = [c for c in resolved if c in ft.columns]
    # drop attributes that are constant across the available gauges (uninformative)
    attr_cols = [c for c in attr_cols if np.nanstd(ft[c].to_numpy(float)) > 0]
    if not attr_cols:
        log.warning("attribute_regression: no varying attributes for model=%s split=%s",
                    model, split)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for target in targets:
        if target not in ft.columns:
            log.warning("attribute_regression: target %r absent; skipping", target)
            continue
        y = ft[target].to_numpy(float)
        good_sign = TARGET_DEFS.get(target, {}).get("good", 0)
        X = ft[attr_cols].to_numpy(float)
        beta, r2, adj, n_used = _ridge_standardised(X, y, alpha=ridge_alpha)
        for j, col in enumerate(attr_cols):
            xj = ft[col].to_numpy(float)
            c = _safe_corr(xj, y)
            sr = c["spearman_r"]
            predicts = "n/a"
            if good_sign and np.isfinite(sr):
                predicts = "failure" if good_sign * sr < 0 else "success"
            rows.append({
                "model": model, "split": split, "target": target,
                "attribute": col, "label": resolved.get(col, {}).get("label", col),
                "source": resolved.get(col, {}).get("source", col),
                "n_gauges": c["n"], "attr_mean": float(np.nanmean(xj)),
                "attr_sd": float(np.nanstd(xj)),
                "spearman_r": sr, "spearman_p": c["spearman_p"],
                "pearson_r": c["pearson_r"], "pearson_p": c["pearson_p"],
                "coef_std": float(beta[j]), "predicts": predicts,
                "r2": r2, "adj_r2": adj, "n_model": n_used,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman"] = out["spearman_r"].abs()
    out["rank"] = (out.groupby("target")["abs_spearman"]
                   .rank(ascending=False, method="min").astype("Int64"))
    out = out.sort_values(["target", "rank"]).reset_index(drop=True)
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].round(4)
    out.attrs["failure_table"] = ft
    out.attrs["resolved"] = resolved
    return out


# --------------------------------------------------------------------------- #
#  Epistemic (between-seed) variance vs error                                 #
# --------------------------------------------------------------------------- #
def between_seed_variance(model: Any, df: pd.DataFrame, *, group: str = "code"
                          ) -> pd.Series:
    """Per-gauge between-seed (epistemic) variance of a fitted deep ensemble.

    For a :class:`~sbc.models.robust.DeepEnsembleCorrector` the members' log-residual
    predictions are stacked and their population variance (the *disagreement* between
    independently seeded members) is taken per row, then averaged per gauge.  Any
    other model exposing :meth:`predict_variance` falls back to its total predictive
    variance.

    Parameters
    ----------
    model : object
        A fitted ensemble exposing ``members_`` (preferred) or ``predict_variance``.
    df : pandas.DataFrame
        Rows to score; must carry the ``group`` column.
    group : str, default ``"code"``
        Aggregation key (gauge id).

    Returns
    -------
    pandas.Series
        Mean epistemic variance per gauge, indexed by ``group`` and named
        ``"epistemic_var"``.
    """
    members = list(getattr(model, "members_", []) or [])
    if len(members) >= 2:
        preds = np.column_stack(
            [np.asarray(m.predict_residual(df), float).ravel() for m in members])
        epi = np.nanvar(preds, axis=1)  # ddof=0 between-member variance
    else:
        pv = getattr(model, "predict_variance", None)
        if not callable(pv):
            raise AttributeError(
                "model exposes neither >=2 members_ nor predict_variance; "
                "cannot derive between-seed variance")
        epi = np.asarray(pv(df), float).ravel()
        log.warning("between_seed_variance: <2 members; using total predict_variance")
    frame = pd.DataFrame({group: np.asarray(df[group]), "epistemic_var": epi})
    return frame.groupby(group)["epistemic_var"].mean().rename("epistemic_var")


def _to_code_series(obj: Any, name: str, codes: Sequence[Any] | None) -> pd.Series:
    """Coerce an array / Series / DataFrame to a code-indexed numeric Series."""
    if isinstance(obj, pd.Series):
        s = obj.astype(float)
    elif isinstance(obj, pd.DataFrame):
        if "code" in obj.columns:
            val = next(c for c in obj.columns if c != "code")
            s = obj.set_index("code")[val].astype(float)
        else:
            s = obj.iloc[:, 0].astype(float)
    else:
        arr = np.asarray(obj, float).ravel()
        idx = list(codes) if codes is not None else list(range(arr.size))
        s = pd.Series(arr, index=idx)
    s = s.rename(name)
    s.index = s.index.astype(str)
    s.index.name = "code"
    return s


def epistemic_vs_error(deepens_variance: Any, error: Any, *,
                       codes: Sequence[Any] | None = None,
                       labels: Any = None, log_variance: bool = True
                       ) -> dict[str, Any]:
    """Correlate deep-ensemble between-seed variance with per-gauge error.

    Tests the claim that ensemble *disagreement* flags out-of-support gauges: if
    member variance rises on the catchments the ensemble then gets wrong, the
    variance is a usable, label-free reliability signal.  When an out-of-support
    ``labels`` mask is supplied the routine additionally reports how well the
    variance ranks those gauges apart (AUC, group means).

    Parameters
    ----------
    deepens_variance : array-like, Series or DataFrame
        Per-gauge between-seed variance (see :func:`between_seed_variance`).  A
        ``code``-indexed Series is aligned by gauge; a plain array is aligned
        positionally (optionally via ``codes``).
    error : array-like, Series or DataFrame
        Per-gauge error to correlate against (e.g. ``1 - KGE'`` or ``|PBIAS|``).
    codes : sequence, optional
        Gauge ids used to align plain-array inputs.
    labels : array-like or Series, optional
        Boolean / 0-1 out-of-support flag per gauge (e.g. ``split == "pur"``); enables
        the separation diagnostics.
    log_variance : bool, default True
        Take ``log10`` of the variance for the Pearson correlation and the figure
        (the rank-based Spearman is transform-invariant either way).

    Returns
    -------
    dict
        ``{"n", "spearman_r", "spearman_p", "pearson_r", "pearson_p",
        "auc_out_of_support", "var_mean_in", "var_mean_out", "log_variance",
        "frame"}`` where ``frame`` is the aligned per-gauge DataFrame
        (``code`` / ``variance`` / ``error`` / optionally ``out_of_support``).
    """
    v = _to_code_series(deepens_variance, "variance", codes)
    e = _to_code_series(error, "error", codes)
    frame = pd.concat([v, e], axis=1, join="inner").reset_index()
    if labels is not None:
        lab = _to_code_series(labels, "out_of_support", codes)
        frame = frame.merge(lab.reset_index(), on="code", how="left")
        frame["out_of_support"] = frame["out_of_support"].fillna(0).astype(float)

    frame = frame[np.isfinite(frame["variance"]) & np.isfinite(frame["error"])].copy()
    var = frame["variance"].to_numpy(float)
    err = frame["error"].to_numpy(float)
    var_for_pearson = np.log10(np.clip(var, 1e-12, None)) if log_variance else var

    sp = _safe_corr(var, err)               # Spearman is rank-based (transform-free)
    pe = _safe_corr(var_for_pearson, err)   # Pearson on (log-)variance

    out: dict[str, Any] = {
        "n": int(frame.shape[0]),
        "spearman_r": sp["spearman_r"], "spearman_p": sp["spearman_p"],
        "pearson_r": pe["pearson_r"], "pearson_p": pe["pearson_p"],
        "log_variance": bool(log_variance),
        "auc_out_of_support": np.nan, "var_mean_in": np.nan, "var_mean_out": np.nan,
        "frame": frame,
    }
    if "out_of_support" in frame.columns and frame["out_of_support"].nunique() == 2:
        y = frame["out_of_support"].to_numpy(float)
        out["var_mean_in"] = float(np.nanmean(var[y == 0]))
        out["var_mean_out"] = float(np.nanmean(var[y == 1]))
        try:
            from sklearn.metrics import roc_auc_score

            out["auc_out_of_support"] = float(roc_auc_score(y, var))
        except Exception:  # pragma: no cover - sklearn optional / degenerate labels
            n_pos = float((y == 1).sum())
            n_neg = float((y == 0).sum())
            if n_pos and n_neg:
                ranks = pd.Series(var).rank().to_numpy()
                auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                out["auc_out_of_support"] = float(auc)
    log.info("epistemic_vs_error: n=%d spearman_r=%.3f (p=%.3g) auc_oos=%s",
             out["n"], out["spearman_r"], out["spearman_p"],
             "n/a" if not np.isfinite(out["auc_out_of_support"])
             else f"{out['auc_out_of_support']:.3f}")
    return out


# --------------------------------------------------------------------------- #
#  Plotting (Agg-safe)                                                        #
# --------------------------------------------------------------------------- #
def _new_axes(figsize: tuple[float, float], nrows: int = 1):
    """Return ``(fig, axes)`` on the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(nrows, 1, figsize=figsize, squeeze=False)


def _save(fig, path: str | Path) -> Path:
    """Tight-layout, save and close a figure; return its path."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figure -> %s", out)
    return out


def _stars(p: float) -> str:
    """Significance stars for a p-value."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def plot_attribute_regression(table: pd.DataFrame, path: str | Path) -> Path:
    """Horizontal bar chart of per-attribute association with each failure target.

    Bars are the univariate Spearman correlation; red marks attributes that predict
    *failure* (worse skill / larger bias as the attribute grows), blue those that
    predict success.  Significance stars and the standardised multivariate
    coefficient annotate each bar; the panel title carries the model R^2.
    """
    targets = list(dict.fromkeys(table["target"]))
    fig, axes = _new_axes((7.5, 0.55 * table["attribute"].nunique() * len(targets) + 1.6),
                          nrows=len(targets))
    for ax, target in zip(axes[:, 0], targets):
        t = table[table["target"] == target].sort_values("spearman_r")
        y = np.arange(len(t))
        colors = ["#d62728" if p == "failure" else "#1f77b4"
                  for p in t["predicts"]]
        ax.barh(y, t["spearman_r"].to_numpy(float), color=colors)
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(t["label"].tolist())
        ax.set_xlim(-1.05, 1.05)
        for yi, (_, r) in zip(y, t.iterrows()):
            sr = r["spearman_r"]
            txt = f"{_stars(r['spearman_p'])}  b*={r['coef_std']:+.2f}"
            ax.text(sr + (0.03 if sr >= 0 else -0.03), yi, txt,
                    va="center", ha="left" if sr >= 0 else "right", fontsize="x-small")
        r2 = t["r2"].iloc[0] if len(t) else np.nan
        ax.set_title(f"{TARGET_DEFS.get(target, {}).get('label', target)}"
                     f"   (multivariate R^2={r2:.2f}, n={int(t['n_gauges'].max())})",
                     fontsize="small")
    axes[-1, 0].set_xlabel("Spearman rank correlation of attribute with target")
    model = str(table["model"].iloc[0])
    split = str(table["split"].iloc[0])
    fig.suptitle(f"What predicts {split.upper()} failure of {model}", fontsize="medium")
    return _save(fig, path)


def plot_attribute_scatter(failure_table: pd.DataFrame, path: str | Path, *,
                           target: str, attribute: str) -> Path:
    """Scatter of one attribute vs a failure target, with OLS line and Spearman r."""
    t = failure_table[[attribute, target, "code"]].dropna()
    x = t[attribute].to_numpy(float)
    y = t[target].to_numpy(float)
    fig, axes = _new_axes((6.0, 4.6))
    ax = axes[0, 0]
    ax.scatter(x, y, s=40, color="#1f77b4", edgecolor="white", zorder=3)
    if x.size >= 2 and x.std() > 0:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="#d62728", lw=1.4)
    c = _safe_corr(x, y)
    ax.set_xlabel(attribute)
    ax.set_ylabel(TARGET_DEFS.get(target, {}).get("label", target))
    ax.set_title(f"{attribute} vs {target}: "
                 f"Spearman r={c['spearman_r']:+.2f}{_stars(c['spearman_p'])} "
                 f"(n={c['n']})", fontsize="small")
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    return _save(fig, path)


def plot_epistemic_vs_error(result: dict[str, Any], path: str | Path) -> Path:
    """Scatter of between-seed variance vs per-gauge error (PUR gauges highlighted)."""
    frame = result["frame"]
    var = frame["variance"].to_numpy(float)
    err = frame["error"].to_numpy(float)
    fig, axes = _new_axes((6.4, 4.8))
    ax = axes[0, 0]
    if "out_of_support" in frame.columns:
        oos = frame["out_of_support"].to_numpy(float) == 1
        ax.scatter(var[~oos], err[~oos], s=42, color="#1f77b4",
                   edgecolor="white", label="in-support", zorder=3)
        ax.scatter(var[oos], err[oos], s=52, color="#d62728", marker="^",
                   edgecolor="white", label="out-of-support (PUR)", zorder=4)
        ax.legend(fontsize="small")
    else:
        ax.scatter(var, err, s=42, color="#1f77b4", edgecolor="white", zorder=3)
    if result.get("log_variance", True) and np.all(var > 0):
        ax.set_xscale("log")
    ax.set_xlabel("deep-ensemble between-seed variance (log-residual)")
    ax.set_ylabel("per-gauge error")
    auc = result.get("auc_out_of_support", np.nan)
    auc_txt = "" if not np.isfinite(auc) else f" | AUC(out-of-support)={auc:.2f}"
    ax.set_title(f"Disagreement flags out-of-support gauges: "
                 f"Spearman r={result['spearman_r']:+.2f}"
                 f"{_stars(result['spearman_p'])} (n={result['n']}){auc_txt}",
                 fontsize="small")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def build_all(tag: str = "v2_real_decadal", *, model: str = "regimeprobnet",
              split: str = "pur", contrast_split: str | None = "temporal",
              targets: Sequence[str] = ("delta_kge", "abs_pbias"),
              per_gauge: pd.DataFrame | None = None, static: pd.DataFrame | None = None,
              attributes: Sequence[Any] | None = None,
              deepens_variance: Any = None, deepens_model: Any = None,
              model_table: pd.DataFrame | None = None, error_metric: str = "one_minus_kge",
              figures_dir: str | Path | None = None,
              tables_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the PUR-attribution analyses for ``tag`` and write tables and figures.

    The attribute regression is cheap (it only reads the per-gauge and static tables)
    and always runs.  The epistemic-vs-error analysis needs a per-gauge between-seed
    variance: pass a precomputed ``deepens_variance`` Series, or a fitted
    ``deepens_model`` + ``model_table`` to derive it via
    :func:`between_seed_variance`; otherwise that part is skipped (logged) so the
    default call stays light.

    Parameters
    ----------
    tag : str
        Result tag; reads ``results/tables/per_gauge_<tag>.parquet`` when
        ``per_gauge`` is ``None``.
    model, split : str
        Model and headline split to explain.
    contrast_split : str or None
        Optional second split (default ``"temporal"``) regressed for contrast.
    targets : sequence of str
        Failure targets (see :data:`TARGET_DEFS`).
    per_gauge, static : pandas.DataFrame, optional
        Pre-loaded tables (default read from the canonical locations).
    attributes : sequence, optional
        Predictor specification.
    deepens_variance : array-like / Series, optional
        Precomputed per-gauge between-seed variance.
    deepens_model, model_table : optional
        A fitted ensemble and the modelling table to derive the variance from.
    error_metric : {"one_minus_kge", "abs_pbias"}
        Per-gauge error used in the epistemic analysis.
    figures_dir, tables_dir : path-like, optional
        Output directories (default ``results/figures`` / ``results/tables``).

    Returns
    -------
    dict
        ``{"tables": {...}, "figures": {...}, "regression": DataFrame,
        "epistemic": dict | None}``.
    """
    from ..utils import save_table

    figures_dir = Path(figures_dir) if figures_dir is not None else PATHS.figures
    tables_dir = Path(tables_dir) if tables_dir is not None else PATHS.tables
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables_out: dict[str, Path] = {}
    figures_out: dict[str, Path] = {}

    def _emit(frame: pd.DataFrame, key: str, stem: str) -> None:
        # ``.attrs`` can hold a nested DataFrame (failure_table) that parquet cannot
        # JSON-serialise, so persist a metadata-free copy.
        f = frame.copy()
        f.attrs = {}
        tables_out[key] = save_table(f, tables_dir / f"{stem}_{tag}.parquet",
                                     csv_mirror=True)

    if per_gauge is None:
        per_gauge = pd.read_parquet(PATHS.tables / f"per_gauge_{tag}.parquet")
        log.info("build_all: loaded per_gauge_%s (%d rows)", tag, len(per_gauge))
    if static is None:
        static = pd.read_parquet(PATHS.processed / "static_attributes.parquet")
        log.info("build_all: loaded static_attributes (%d gauges)", len(static))

    # -- 1. attribute regression on the headline split ---------------------- #
    reg = attribute_regression(per_gauge, static, model=model, split=split,
                               targets=targets, attributes=attributes)
    if not reg.empty:
        _emit(reg, "regression", "pur_attr_regression")
        figures_out["regression"] = plot_attribute_regression(
            reg, figures_dir / f"pur_attr_regression_{tag}.png")
        ft = reg.attrs.get("failure_table")
        if ft is not None and not ft.empty:
            _emit(ft, "failure_table", "pur_failure_gauges")
            top = reg[reg["target"] == targets[0]].sort_values("rank").iloc[0]
            figures_out["scatter"] = plot_attribute_scatter(
                ft, figures_dir / f"pur_attr_scatter_{tag}.png",
                target=str(targets[0]), attribute=str(top["attribute"]))

    # -- 1b. optional contrast split (e.g. temporal) ------------------------ #
    if contrast_split and contrast_split != split:
        reg_c = attribute_regression(per_gauge, static, model=model,
                                     split=contrast_split, targets=targets,
                                     attributes=attributes)
        if not reg_c.empty:
            _emit(reg_c, "regression_contrast", f"pur_attr_regression_{contrast_split}")

    # -- 2. epistemic variance vs error (needs a variance source) ----------- #
    epi: dict[str, Any] | None = None
    if deepens_variance is None and deepens_model is not None and model_table is not None:
        try:
            deepens_variance = between_seed_variance(deepens_model, model_table)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("build_all: between_seed_variance failed: %s", exc)
    if deepens_variance is not None:
        de = per_gauge[per_gauge["model"] == "deepens"].copy()
        if de.empty:
            de = per_gauge.copy()
        de["code"] = de["code"].astype(str)
        de = de.groupby("code", as_index=False).agg(
            kge=("kge", "median"), pbias=("pbias", "median"),
            is_pur=("split", lambda s: float((s == "pur").any())))
        if error_metric == "abs_pbias":
            de["error"] = de["pbias"].abs()
        else:
            de["error"] = 1.0 - de["kge"]
        epi = epistemic_vs_error(
            deepens_variance, de.set_index("code")["error"],
            labels=de.set_index("code")["is_pur"])
        _emit(epi["frame"], "epistemic", "pur_epistemic")
        figures_out["epistemic"] = plot_epistemic_vs_error(
            epi, figures_dir / f"pur_epistemic_vs_error_{tag}.png")
    else:
        log.info("build_all: no deepens variance supplied; skipping epistemic analysis")

    log.info("build_all(%s): %d tables, %d figures", tag, len(tables_out), len(figures_out))
    return {"tables": tables_out, "figures": figures_out, "regression": reg,
            "epistemic": epi}


# --------------------------------------------------------------------------- #
#  Self-test (tiny synthetic per-gauge + static; <3 min)                      #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..schemas import validate
    from ..synthetic import generate
    from ..validation import cv

    # -- small synthetic modelling table -> features -> regimes ------------- #
    raw = generate(scale="decadal", years=8, n_basins=3,
                   gauges_per_basin=(3, 4), seed=7)
    df = classify_regimes(build_features(validate(raw), scale="decadal")).reset_index(drop=True)
    print(f"[pur_attr] table: {len(df)} rows | {df['code'].nunique()} gauges | "
          f"domains={sorted(df['domain'].unique())}")

    # synthetic static-attribute table: per-gauge-constant columns ----------- #
    static_cols = ["area_km2", "elev_m", "slope_deg", "snow_frac", "glacier_frac", "aridity"]
    static = df.groupby("code", as_index=False)[static_cols].first()

    # synthetic per-gauge result table from a fast deterministic corrector --- #
    from ..models.quantile_mapping import LinearScalingCorrector

    pg = cv.run_matrix(lambda: LinearScalingCorrector(), df, "scaling",
                       temporal=True, lobo=False, pur=True, test_frac=0.3)
    print(f"[pur_attr] per_gauge: {len(pg)} rows | splits={sorted(pg['split'].unique())} "
          f"| pur gauges={pg[pg.split=='pur']['code'].nunique()}")

    # -- (1) attribute regression on the PUR split -------------------------- #
    reg = attribute_regression(pg, static, model="scaling", split="pur",
                               targets=("delta_kge", "abs_pbias"))
    assert not reg.empty, "attribute_regression returned empty"
    assert {"attribute", "spearman_r", "coef_std", "predicts", "rank", "r2"}.issubset(reg.columns)
    assert set(reg["target"]) == {"delta_kge", "abs_pbias"}, "targets mismatch"
    assert reg["spearman_r"].notna().any(), "no association computed"
    assert (reg["n_gauges"] >= MIN_GAUGES).any(), "too few gauges joined"
    print("[pur_attr] PUR attribute ranking (delta_kge):")
    view = reg[reg.target == "delta_kge"][["attribute", "spearman_r", "spearman_p",
                                           "coef_std", "predicts", "rank"]]
    print(view.to_string(index=False))
    top = reg[reg.target == "delta_kge"].sort_values("rank").iloc[0]
    print(f"[pur_attr] strongest delta_kge predictor: {top['attribute']} "
          f"(Spearman r={top['spearman_r']:+.2f}, predicts {top['predicts']}, "
          f"multivariate R^2={top['r2']:.2f})")

    # -- (2) epistemic variance vs error: real tiny deep ensemble ----------- #
    epi_ok = "synthetic-array fallback"
    var_series = None
    try:
        from ..models.robust import DeepEnsembleCorrector
        from ..validation.splits import pur_split

        ptr, _ = pur_split(df)
        ens = DeepEnsembleCorrector(
            base="ealstm", n_members=2, pur_robust=False,
            member_kwargs=dict(max_epochs=2, hidden_size=12, seq_length=4,
                               batch_size=1024, patience=2),
            seed=0).fit(df[ptr].reset_index(drop=True))
        var_series = between_seed_variance(ens, df)
        assert var_series.notna().any() and (var_series >= 0).all()
        # per-gauge error of the ensemble's corrected discharge
        from ..validation.metrics import evaluate_by_group

        preds = ens.predict(df)
        ev = evaluate_by_group(df.assign(q_pred=preds), "q_obs", "q_pred")
        err = pd.Series((1.0 - ev["kge"]).to_numpy(), index=ev["code"].astype(str))
        is_pur = df.groupby("code")["domain"].first().eq("transfer").astype(float)
        epi = epistemic_vs_error(var_series, err, labels=is_pur)
        epi_ok = (f"real deepens: var_in={epi['var_mean_in']:.2e} "
                  f"var_out={epi['var_mean_out']:.2e} auc={epi['auc_out_of_support']:.2f}")
    except Exception as exc:  # torch/ensemble optional -> synthetic arrays
        log.warning("[pur_attr] deep-ensemble path skipped (%s); using synthetic arrays", exc)
        rng = np.random.default_rng(0)
        codes = [f"S{i:03d}" for i in range(20)]
        oos = np.array([0] * 14 + [1] * 6, float)
        var = np.exp(rng.normal(-6 + 1.5 * oos, 0.4))       # higher on out-of-support
        err = 0.1 + 0.8 * oos + rng.normal(0, 0.05, 20)      # higher on out-of-support
        epi = epistemic_vs_error(var, err, codes=codes,
                                 labels=pd.Series(oos, index=codes))

    assert {"spearman_r", "frame", "auc_out_of_support"}.issubset(epi)
    assert np.isfinite(epi["spearman_r"]), "epistemic correlation not computed"
    print(f"[pur_attr] epistemic_vs_error ({epi_ok}): "
          f"Spearman r={epi['spearman_r']:+.2f} (p={epi['spearman_p']:.3g}), "
          f"n={epi['n']}, AUC(out-of-support)={epi['auc_out_of_support']:.2f}")

    # -- (3) figures + orchestration into a scratch dir --------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        out = build_all("synthetic_decadal_quick", model="scaling", split="pur",
                        contrast_split="temporal", per_gauge=pg, static=static,
                        deepens_variance=var_series,
                        figures_dir=Path(tmp) / "fig", tables_dir=Path(tmp) / "tab")
        n_fig = len(list((Path(tmp) / "fig").glob("*.png")))
        n_tab = len(list((Path(tmp) / "tab").glob("*.parquet")))
    assert not out["regression"].empty, "build_all produced no regression"
    assert n_fig >= 2 and n_tab >= 2, f"unexpected outputs: {n_fig} figs, {n_tab} tabs"
    print(f"[pur_attr] build_all wrote {n_fig} figures + {n_tab} tables to scratch "
          f"| figures={list(out['figures'])}")
    print("[pur_attr] SELF-TEST OK")


if __name__ == "__main__":
    _selftest()
