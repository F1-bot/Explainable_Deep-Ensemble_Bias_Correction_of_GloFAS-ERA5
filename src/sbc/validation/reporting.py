"""Reviewer-grade reporting helpers: dual KGE, skill-class tables, monotonicity.

KGE' (the modified Kling-Gupta efficiency of Kling et al., 2012) is the
framework's headline deterministic score and already lives in
:func:`sbc.validation.metrics.kge_prime`.  Several of the baselines this paper is
benchmarked against -- notably the GloFAS-LSTM corrector of Hunt et al. (2022,
*HESS* 26:5449) -- instead report the *original* KGE of Gupta et al. (2009) and
classify each gauge with the Knoben et al. (2019, *HESS* 23:4323) thresholds:
a model is **skilful** at a gauge when it beats the mean-flow benchmark
(``KGE > -0.41``) and **highly skilful** when ``KGE > 0.707``.  To make the two
literatures directly comparable this module adds

* :func:`kge_2009` -- the 2009 KGE (variability term = standard-deviation ratio
  ``alpha``, *not* the coefficient-of-variation ratio ``gamma`` of KGE'); and
* :func:`dual_kge` -- both efficiencies side by side plus the Knoben skill flags,
  so a single call yields the numbers needed to position against KGE2009 papers.

On top of the per-pair scores it provides the two *rhetorical* artefacts a Q1
reviewer expects rather than a single pooled median:

* :func:`skill_class_table` -- per ``model x split`` counts of gauges that are
  skilful / highly skilful (Hunt's idiom *"highly skilful at k of N; NSE>0.9 at
  m of N"*) **and** the per-gauge KGE distribution (quantiles), which is far
  stronger evidence than one basin-averaged number; and
* :func:`monotonicity_violation_rate` -- the concrete *"the physics constraint
  works"* number: the fraction of melt-regime samples at which the predicted
  log-residual is locally *decreasing* in snow-water-equivalent / snowmelt
  (a finite-difference estimate of ``d residual / d feature < 0``), to contrast
  the soft-penalty flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet`
  with the hard-constrained ``probnet_hardmono`` variant.

Everything is pure, NaN-aware NumPy/pandas; it reuses
:mod:`sbc.validation.metrics` and the :mod:`sbc.schemas` conventions and never
hard-codes a feature list.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..schemas import REGIME_COL
from ..utils import get_logger
from . import metrics as M

log = get_logger(__name__)

__all__ = [
    "SKILFUL_KGE",
    "HIGHLY_SKILFUL_KGE",
    "DEFAULT_KGE_QUANTILES",
    "kge_2009",
    "dual_kge",
    "skill_class",
    "skill_class_table",
    "monotonicity_violation_rate",
]

#: Knoben et al. (2019) skill thresholds.  ``-0.41`` is the KGE of the mean-flow
#: benchmark (``1 - sqrt(2) ~= -0.4142``); a model is *skilful* only above it, so
#: the historic ``KGE > 0`` line systematically over-states skill.
SKILFUL_KGE: float = -0.41
#: Above this a gauge is *highly skilful* (Hunt et al., 2022 / Knoben et al.).
HIGHLY_SKILFUL_KGE: float = 0.707
#: Quantiles reported for the per-gauge KGE distribution.
DEFAULT_KGE_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


# --------------------------------------------------------------------------- #
#  KGE (2009) and the dual report                                             #
# --------------------------------------------------------------------------- #
def kge_2009(obs, sim) -> dict[str, float]:
    """Original Kling-Gupta efficiency (Gupta et al., 2009) and its components.

    Identical to the modified :func:`sbc.validation.metrics.kge_prime` except for
    the *variability* term: the 2009 formulation uses the ratio of standard
    deviations ``alpha = sigma_sim / sigma_obs`` whereas KGE' (2012) uses the
    ratio of coefficients of variation ``gamma = CV_sim / CV_obs``.  The two
    coincide only when the bias ratio ``beta`` is one.

    ``KGE = 1 - sqrt[(r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2]``

    Parameters
    ----------
    obs, sim : array_like
        Observed and simulated/corrected series (NaNs are dropped pairwise).

    Returns
    -------
    dict
        ``kge`` (the 2009 efficiency), ``r`` (Pearson correlation), ``beta``
        (mean ratio ``mu_sim / mu_obs``) and ``alpha`` (std ratio).  All ``NaN``
        when fewer than three finite pairs or ``sigma_obs == 0``.
    """
    obs, sim = M._clean(obs, sim)
    if obs.size < 3 or obs.std() == 0:
        return {"kge": np.nan, "r": np.nan, "beta": np.nan, "alpha": np.nan}
    r = float(np.corrcoef(obs, sim)[0, 1])
    mo, ms = float(obs.mean()), float(sim.mean())
    beta = ms / mo if mo != 0 else np.nan
    alpha = float(sim.std() / obs.std())
    kge = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"kge": float(kge), "r": r, "beta": float(beta), "alpha": alpha}


def skill_class(kge: float) -> str:
    """Map a scalar KGE to a Knoben skill label.

    Parameters
    ----------
    kge : float
        A KGE value (2009 or 2012 -- the thresholds are formulation-agnostic).

    Returns
    -------
    str
        ``"highly_skilful"`` (``> 0.707``), ``"skilful"`` (``> -0.41``),
        ``"unskilful"`` (finite but at/below the mean-flow benchmark) or
        ``"undefined"`` (non-finite).
    """
    if not np.isfinite(kge):
        return "undefined"
    if kge > HIGHLY_SKILFUL_KGE:
        return "highly_skilful"
    if kge > SKILFUL_KGE:
        return "skilful"
    return "unskilful"


def dual_kge(obs, sim) -> dict[str, float]:
    """Both Kling-Gupta efficiencies (2009 *and* 2012) plus Knoben skill flags.

    Computes the original KGE (:func:`kge_2009`) and the modified KGE'
    (:func:`sbc.validation.metrics.kge_prime`) on the same pair so the paper can
    be quoted in either convention, and tags the 2009 value with the Knoben
    ``skilful`` / ``highly_skilful`` flags used by Hunt et al. (2022).

    Parameters
    ----------
    obs, sim : array_like
        Observed and simulated/corrected series.

    Returns
    -------
    dict
        ``kge2009`` / ``kge2009_r`` / ``kge2009_beta`` / ``kge2009_alpha`` (2009),
        ``kge2012`` / ``kge2012_r`` / ``kge2012_beta`` / ``kge2012_gamma`` (2012,
        from ``kge_prime``), the boolean ``skilful`` / ``highly_skilful`` flags
        (judged on the 2009 KGE, Hunt's convention) and the string ``skill_class``.
    """
    k09 = kge_2009(obs, sim)
    k12 = M.kge_prime(obs, sim)
    out = {
        "kge2009": k09["kge"], "kge2009_r": k09["r"],
        "kge2009_beta": k09["beta"], "kge2009_alpha": k09["alpha"],
        "kge2012": k12["kge"], "kge2012_r": k12["r"],
        "kge2012_beta": k12["beta"], "kge2012_gamma": k12["gamma"],
    }
    out["skilful"] = bool(np.isfinite(k09["kge"]) and k09["kge"] > SKILFUL_KGE)
    out["highly_skilful"] = bool(np.isfinite(k09["kge"]) and k09["kge"] > HIGHLY_SKILFUL_KGE)
    out["skill_class"] = skill_class(k09["kge"])
    return out


# --------------------------------------------------------------------------- #
#  Skill-class table (counts + per-gauge distribution per model x split)      #
# --------------------------------------------------------------------------- #
def skill_class_table(per_gauge: pd.DataFrame, kge_col: str = "kge", *,
                      group_cols: Sequence[str] = ("model", "split"),
                      nse_col: str = "nse", nse_threshold: float = 0.9,
                      quantiles: Sequence[float] = DEFAULT_KGE_QUANTILES,
                      code_col: str = "code") -> pd.DataFrame:
    """Per ``model x split`` skill-class counts and the per-gauge KGE distribution.

    Consumes a tidy *per-gauge* results table (e.g. the output of
    :func:`sbc.validation.cv.run_matrix` / :func:`~sbc.validation.cv.compare`,
    one row per gauge) and, for every group, reports how many gauges are
    *skilful* (``KGE > -0.41``) and *highly skilful* (``KGE > 0.707``) -- Hunt's
    *"highly skilful at k of N"* idiom -- alongside the quantiles of the per-gauge
    KGE, which carries far more information than a single pooled median.  When an
    NSE column is present it additionally counts gauges with ``NSE > nse_threshold``
    (Hunt's *"NSE>0.9 at m of N"*).

    Parameters
    ----------
    per_gauge : pandas.DataFrame
        Per-gauge results; must contain ``kge_col`` and, ideally, the grouping
        and ``code`` columns.
    kge_col : str, default ``"kge"``
        Column holding the per-gauge KGE to classify.  Pass ``"kge2009"`` to
        reproduce Hunt's original-KGE classification exactly.
    group_cols : sequence of str, default ``("model", "split")``
        Grouping columns; those absent from ``per_gauge`` are silently ignored
        (an empty intersection collapses to a single overall group).
    nse_col : str, default ``"nse"``
        Optional NSE column for the ``NSE > nse_threshold`` count.
    nse_threshold : float, default 0.9
        NSE bar for the "good fit" count.
    quantiles : sequence of float, default :data:`DEFAULT_KGE_QUANTILES`
        Quantiles of the per-gauge KGE distribution to report (columns
        ``kge_q{pct}``).
    code_col : str, default ``"code"``
        Gauge id column; used only to report ``n_unique_gauges``.

    Returns
    -------
    pandas.DataFrame
        One row per group with the group keys, ``n_gauges`` (gauges with a finite
        KGE, the ``N`` of "k of N"), ``n_skilful`` / ``n_highly_skilful`` and
        their fractions, ``n_nse_above`` (when ``nse_col`` present), the KGE
        ``min`` / ``mean`` / ``max`` and ``kge_q{pct}`` quantile columns, and a
        ready-to-quote ``summary`` string.
    """
    if kge_col not in per_gauge.columns:
        raise KeyError(f"per_gauge has no KGE column {kge_col!r}; "
                       f"columns are {list(per_gauge.columns)}")

    present = [c for c in group_cols if c in per_gauge.columns]
    has_nse = nse_col in per_gauge.columns
    qs = list(quantiles)

    if present:
        groups: Iterable[tuple[Any, pd.DataFrame]] = per_gauge.groupby(present, sort=True)
    else:
        groups = [((), per_gauge)]

    rows: list[dict[str, Any]] = []
    for keys, g in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec: dict[str, Any] = dict(zip(present, keys))

        k = g[kge_col].to_numpy(float)
        finite = k[np.isfinite(k)]
        n = int(finite.size)
        rec["n_gauges"] = n
        if code_col in g.columns:
            rec["n_unique_gauges"] = int(g[code_col].nunique())

        n_sk = int((finite > SKILFUL_KGE).sum())
        n_hsk = int((finite > HIGHLY_SKILFUL_KGE).sum())
        rec["n_skilful"] = n_sk
        rec["n_highly_skilful"] = n_hsk
        rec["frac_skilful"] = float(n_sk / n) if n else np.nan
        rec["frac_highly_skilful"] = float(n_hsk / n) if n else np.nan

        n_nse = None
        if has_nse:
            ns = g[nse_col].to_numpy(float)
            ns = ns[np.isfinite(ns)]
            n_nse = int((ns > nse_threshold).sum())
            rec["n_nse_above"] = n_nse
            rec["nse_threshold"] = float(nse_threshold)

        if n:
            rec["kge_min"] = float(np.min(finite))
            rec["kge_mean"] = float(np.mean(finite))
            rec["kge_max"] = float(np.max(finite))
            for q in qs:
                rec[f"kge_q{int(round(q * 100))}"] = float(np.quantile(finite, q))
        else:
            rec["kge_min"] = rec["kge_mean"] = rec["kge_max"] = np.nan
            for q in qs:
                rec[f"kge_q{int(round(q * 100))}"] = np.nan

        summary = (f"highly skilful at {n_hsk} of {n}; "
                   f"skilful at {n_sk} of {n}")
        if n_nse is not None:
            summary += f"; NSE>{nse_threshold:g} at {n_nse} of {n}"
        rec["summary"] = summary
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty and present:
        out = out.sort_values(present).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
#  Monotonicity-violation rate (the "physics constraint works" number)        #
# --------------------------------------------------------------------------- #
def _regime_labels(df: pd.DataFrame) -> np.ndarray:
    """String regime label per row: use the ``regime`` column or classify on the fly."""
    if REGIME_COL in df.columns:
        return df[REGIME_COL].astype(str).to_numpy()
    try:  # classify_regimes is leakage-free (rule-based on the row's own forcings)
        from ..features.regimes import classify_regimes
        return classify_regimes(df)[REGIME_COL].astype(str).to_numpy()
    except Exception as exc:  # pragma: no cover - features module optional
        log.warning("could not classify regimes (%s); treating all rows as melt", exc)
        return np.full(len(df), "melt_freshet", dtype=object)


def monotonicity_violation_rate(
    model: Any, df: pd.DataFrame,
    features: str | Sequence[str] = ("swe", "smlt"),
    regime: str | Iterable[str] = "melt_freshet", *,
    rel_step: float = 0.05, h: float | None = None, tol: float = 1e-6,
) -> dict[str, Any]:
    """Fraction of melt-regime samples that violate the snow monotonicity constraint.

    Physically, during snow-/ice-melt the corrected discharge must be
    non-decreasing in snow-water-equivalent (``swe``) and in snowmelt (``smlt``).
    Because :func:`sbc.schemas.back_transform` is strictly increasing in the
    predicted log-residual, this is equivalent to
    ``d residual / d feature >= 0``, the exact quantity the flagship's soft
    penalty pushes towards and the ``probnet_hardmono`` variant enforces by
    construction.  This routine estimates that derivative by a **central finite
    difference** -- perturbing one raw feature column at a time and re-running
    ``predict_residual`` -- and reports the share of melt-regime rows at which it
    is negative.  A near-zero number is the concrete evidence that the physics
    constraint holds out of sample.

    Parameters
    ----------
    model
        Any fitted :class:`~sbc.models.base.BaseCorrector` exposing
        ``predict_residual`` (the flagship, ``probnet_hardmono``, a tree model,
        ...).
    df : pandas.DataFrame
        Modelling table to probe; must carry the perturbed ``features`` and,
        ideally, a ``regime`` column (otherwise regimes are classified on the
        fly via :func:`sbc.features.regimes.classify_regimes`).
    features : str or sequence of str, default ``("swe", "smlt")``
        Raw feature(s) the prediction must be non-decreasing in.  Features absent
        from ``df`` are skipped with a warning.
    regime : str or iterable of str, default ``"melt_freshet"``
        Regime label(s) over which the constraint is asserted.  Pass e.g.
        ``("melt_freshet", "glacier_melt", "rain_on_snow")`` for all melt-driven
        regimes.
    rel_step : float, default 0.05
        Finite-difference step as a fraction of each feature's standard deviation
        (used when ``h`` is ``None``).
    h : float, optional
        Absolute step overriding ``rel_step`` for every feature.
    tol : float, default 1e-6
        A derivative below ``-tol`` counts as a violation; the small slack guards
        against float32 round-off in the deep models.

    Returns
    -------
    dict
        ``regime`` (the resolved regime set), ``features`` (those actually
        probed), ``n_melt`` (melt rows found), ``n_eval`` (finite gradient
        evaluations pooled over rows x features), ``violation_rate`` (pooled
        fraction with ``grad < -tol``), ``mean_grad`` / ``min_grad`` (pooled
        gradient summaries) and ``per_feature`` -- a mapping from feature name to
        its own ``n`` / ``violation_rate`` / ``mean_grad`` / ``min_grad`` /
        ``step``.
    """
    if not hasattr(model, "predict_residual"):
        raise AttributeError("monotonicity_violation_rate needs a model with predict_residual()")

    feats_req = (features,) if isinstance(features, str) else list(features)
    feats = [f for f in feats_req if f in df.columns]
    missing = [f for f in feats_req if f not in df.columns]
    if missing:
        log.warning("monotonicity: features %s absent from the table; skipped", missing)
    regimes = {regime} if isinstance(regime, str) else set(regime)

    work = df.reset_index(drop=True).copy()
    labels = _regime_labels(work)
    melt = np.isin(labels, list(regimes))
    n_melt = int(melt.sum())

    base = {
        "regime": sorted(regimes),
        "features": feats,
        "n_melt": n_melt,
        "n_eval": 0,
        "violation_rate": np.nan,
        "mean_grad": np.nan,
        "min_grad": np.nan,
        "per_feature": {},
    }
    if not feats or n_melt == 0:
        if not feats:
            log.warning("monotonicity: no usable feature columns; nothing to evaluate")
        else:
            log.warning("monotonicity: no rows in regime %s", sorted(regimes))
        return base

    per_feature: dict[str, dict[str, float]] = {}
    pooled: list[np.ndarray] = []
    for f in feats:
        col = work[f].to_numpy(float).copy()
        finite_col = col[np.isfinite(col)]
        scale = float(np.std(finite_col)) if finite_col.size else 0.0
        step = float(h) if h is not None else (rel_step * scale if scale > 0 else 1e-3)

        work[f] = col + step
        fp = np.asarray(model.predict_residual(work), float)
        work[f] = col - step
        fm = np.asarray(model.predict_residual(work), float)
        work[f] = col  # restore before touching the next feature

        grad = (fp - fm) / (2.0 * step)
        gm = grad[melt]
        gm = gm[np.isfinite(gm)]
        n = int(gm.size)
        n_viol = int((gm < -tol).sum())
        per_feature[f] = {
            "n": n,
            "violation_rate": float(n_viol / n) if n else np.nan,
            "mean_grad": float(np.mean(gm)) if n else np.nan,
            "min_grad": float(np.min(gm)) if n else np.nan,
            "step": step,
        }
        pooled.append(gm)

    allg = np.concatenate(pooled) if pooled else np.zeros(0)
    n_eval = int(allg.size)
    base.update({
        "n_eval": n_eval,
        "violation_rate": float((allg < -tol).sum() / n_eval) if n_eval else np.nan,
        "mean_grad": float(np.mean(allg)) if n_eval else np.nan,
        "min_grad": float(np.min(allg)) if n_eval else np.nan,
        "per_feature": per_feature,
    })
    log.info("monotonicity(%s): %d melt rows, violation_rate=%.4f over %d evals",
             sorted(regimes), n_melt, base["violation_rate"], n_eval)
    return base


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
def _per_gauge_kge(obs: np.ndarray, pred: np.ndarray, codes: np.ndarray,
                   model_name: str, split_name: str) -> pd.DataFrame:
    """Tiny tidy per-gauge KGE table (2009 + 2012 + NSE) for the skill-class demo."""
    rows = []
    for code in pd.unique(codes):
        m = codes == code
        rows.append({
            "model": model_name, "split": split_name, "code": str(code),
            "kge2009": kge_2009(obs[m], pred[m])["kge"],
            "kge": M.kge_prime(obs[m], pred[m])["kge"],
            "nse": M.nse(obs[m], pred[m]),
        })
    return pd.DataFrame(rows)


def _selftest() -> None:  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..models import get_model, load_all
    from ..models.regime_prob_net import RegimeProbNet
    from ..schemas import OBS_COL, SIM_COL, validate
    from ..synthetic import generate
    from .splits import temporal_split

    load_all()

    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(df, scale="decadal"))
    df = validate(df).reset_index(drop=True)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train = df[tr_mask].reset_index(drop=True)
    test = df[te_mask].reset_index(drop=True)
    obs = test[OBS_COL].to_numpy(float)
    raw = test[SIM_COL].to_numpy(float)
    codes = test["code"].to_numpy()
    print(f"[reporting] gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    # ----- a tiny LightGBM corrector (falls back to flagship if lgbm absent) -- #
    lgbm_pred = None
    try:
        lgbm = get_model("lgbm")(seed=0).fit(train)
        lgbm_pred = lgbm.predict(test)
    except Exception as exc:
        print(f"[reporting] lgbm unavailable ({exc}); continuing with flagship only")

    # ----- a tiny soft-penalty flagship -------------------------------------- #
    flag = RegimeProbNet(K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                         epochs=3, batch_size=256, patience=5, lambda_gate=0.3,
                         lambda_phys=0.1, seed=0, verbose=False)
    flag.fit(train, valid=test)
    flag_pred = flag.predict(test)

    # ===== 1) KGE2009 vs KGE'2012 (pooled, raw vs corrected) ================= #
    d_raw = dual_kge(obs, raw)
    d_flag = dual_kge(obs, flag_pred)
    print(f"[reporting] raw      KGE2009={d_raw['kge2009']:+.3f} "
          f"KGE'2012={d_raw['kge2012']:+.3f} class={d_raw['skill_class']}")
    print(f"[reporting] flagship KGE2009={d_flag['kge2009']:+.3f} "
          f"KGE'2012={d_flag['kge2012']:+.3f} class={d_flag['skill_class']} "
          f"(skilful={d_flag['skilful']}, highly={d_flag['highly_skilful']})")

    # ===== 2) skill-class table (per model x split, KGE2009 classification) == #
    frames = [_per_gauge_kge(obs, raw, codes, "raw_glofas", "temporal")]
    if lgbm_pred is not None:
        frames.append(_per_gauge_kge(obs, lgbm_pred, codes, "lgbm", "temporal"))
    frames.append(_per_gauge_kge(obs, flag_pred, codes, "flagship", "temporal"))
    per_gauge = pd.concat(frames, ignore_index=True)
    sct = skill_class_table(per_gauge, kge_col="kge2009")
    cols = ["model", "split", "n_gauges", "n_skilful", "n_highly_skilful",
            "n_nse_above", "kge_q10", "kge_q50", "kge_q90"]
    print("[reporting] skill-class table (classified on KGE2009):")
    print(sct[[c for c in cols if c in sct.columns]].to_string(index=False))
    for _, r in sct.iterrows():
        print(f"[reporting]   {r['model']:<11s}: {r['summary']}")

    # ===== 3) monotonicity-violation rate (soft flagship vs an unconstrained tree)
    mv_flag = monotonicity_violation_rate(flag, test, features=("swe", "smlt"),
                                          regime="melt_freshet")
    print(f"[reporting] flagship melt-monotonicity: n_melt={mv_flag['n_melt']} "
          f"violation_rate={mv_flag['violation_rate']:.4f} "
          f"mean_grad={mv_flag['mean_grad']:+.4g}")
    for f, st in mv_flag["per_feature"].items():
        print(f"[reporting]   d(resid)/d({f}): viol={st['violation_rate']:.4f} "
              f"mean={st['mean_grad']:+.4g} min={st['min_grad']:+.4g} (n={st['n']})")
    if lgbm_pred is not None:
        mv_lgbm = monotonicity_violation_rate(lgbm, test, features=("swe", "smlt"),
                                              regime="melt_freshet")
        print(f"[reporting] lgbm (unconstrained) melt-monotonicity: "
              f"violation_rate={mv_lgbm['violation_rate']:.4f}")

    # ----- assertions -------------------------------------------------------- #
    assert set(["kge2009", "kge2012", "skilful", "highly_skilful"]).issubset(d_flag)
    assert sct.shape[0] == len(frames) and "summary" in sct.columns
    assert (sct["n_skilful"] >= sct["n_highly_skilful"]).all()
    assert 0.0 <= mv_flag["violation_rate"] <= 1.0 and mv_flag["n_melt"] >= 1
    assert set(mv_flag["per_feature"]) == {"swe", "smlt"}
    print("[reporting] SELF-TEST OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
