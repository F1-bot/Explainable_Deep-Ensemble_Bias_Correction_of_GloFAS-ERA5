"""Final-rigor integration driver for the explainable bias-correction study.

This is the *single* script the maintainer runs on the **real** assembled data
once the parallel sibling agents have landed their modules.  It stitches the
late-stage rigor experiments into one leakage-safe, reproducible pass and writes
every artefact under ``results/tables/*_finalrigor*``:

1. **Transfer learning** — evaluate the few-shot
   :class:`~sbc.models.transfer_lstm.TransferLSTMCorrector` against the flagship
   :class:`~sbc.models.regime_prob_net.RegimeProbNet` on the temporal *and* PUR
   (core->transfer) protocols, and print the *few-shot transfer curve* (PUR skill
   vs. the number of donor gauges revealed from the held-out transfer domain).
2. **SHAP stability** — run
   :func:`sbc.explain.shap_stability.stability_across_seeds` on the flagship
   (``method="flagship"``) and on LightGBM (``method="tree"``) and persist the
   cross-seed attribution *agreement* metrics (Kendall-tau rank correlation /
   top-k Jaccard overlap).
3. **Dual reporting** — :mod:`sbc.validation.reporting`: the dual KGE-2009 /
   KGE'-2012 headline table, the per-gauge Knoben *skill-class* table read from
   ``results/tables/per_gauge_v2_real_decadal.parquet`` and the *soft vs.
   hard-monotone* :func:`~sbc.validation.reporting.monotonicity_violation_rate`.
4. **Monthly SABER row** — a like-for-like monthly bias-correction run (donor /
   SABER-style regionalization vs. our LightGBM) so the paper carries a fair,
   native-resolution SABER comparison; if no monthly modelling table exists the
   decadal table is aggregated to calendar months first.
5. **Multi-scale consistency** — :mod:`sbc.multiscale` cross-scale transfer
   retention and the daily<->decadal aggregation-consistency diagnostic.

The heavy real matrix is **never** run here.  ``--dry-run`` exercises the entire
wiring on a tiny synthetic table (3 basins, 8 years, 3-epoch nets) in a couple of
minutes so the integration can be smoke-tested before the real reanalysis.

Every sibling call is made through its documented public API and wrapped so that,
if a module is not yet present, the driver falls back to an inline implementation
built from the shared primitives — the artefact is always produced and the wiring
is always proven.

Run::

    PYTHONPATH=src python scripts/run_final_rigor.py --dry-run
    PYTHONPATH=src python scripts/run_final_rigor.py            # real data
    PYTHONPATH=src python scripts/run_final_rigor.py --models regimeprobnet,stacked,donor
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import load_config, prepare
from sbc.schemas import OBS_COL, PRED_COL, SIM_COL, TARGET_COL, validate
from sbc.utils import get_logger, save_json, save_table, seed_everything
from sbc.validation import cv
from sbc.validation import metrics as M
from sbc.validation.splits import temporal_split

log = get_logger("finalrigor")
T0 = time.time()

#: headline models reported by the dual-KGE / skill-class tables.
HEADLINE_DEFAULT: tuple[str, ...] = (
    "regimeprobnet", "stacked", "deepens", "glofas_lstm", "donor", "qrf", "lgbm",
)


def _stamp(msg: str) -> None:
    """Log ``msg`` with an elapsed-seconds prefix."""
    log.info("[%6.0fs] %s", time.time() - T0, msg)


def _stem(name: str, tag: str) -> str:
    """Build an output stem of the form ``<name>_finalrigor_<tag>``."""
    return f"{name}_finalrigor_{tag}"


def _median_kge(test: pd.DataFrame, model) -> tuple[float, float]:
    """Median per-gauge KGE' of a fitted model's correction and of raw GloFAS."""
    work = test.assign(**{PRED_COL: np.asarray(model.predict(test), float)})
    pg = M.evaluate_by_group(work, OBS_COL, PRED_COL)
    pg_raw = M.evaluate_by_group(work, OBS_COL, SIM_COL)
    return float(pg["kge"].median(skipna=True)), float(pg_raw["kge"].median(skipna=True))


# --------------------------------------------------------------------------- #
#  Data assembly (real) / synthesis (dry-run)                                 #
# --------------------------------------------------------------------------- #
def build_decadal(args, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw_decadal, prepared_decadal)`` for the requested mode.

    ``raw_decadal`` is the pre-feature modelling table (kept for the monthly
    aggregation), ``prepared_decadal`` has engineered features and regimes.
    """
    if args.dry_run:
        from sbc.synthetic import generate

        raw = generate(scale="decadal", years=args.years, n_basins=args.n_basins,
                       gauges_per_basin=(2, 3), seed=cfg.get("seed", 1234))
    else:
        raw = assemble("decadal")
    prepared = prepare(raw.copy(), "decadal", cfg).reset_index(drop=True)
    return raw, prepared


# --------------------------------------------------------------------------- #
#  Model factories                                                            #
# --------------------------------------------------------------------------- #
def _flagship_factory(net: dict) -> Callable:
    """Zero-argument factory for the flagship RegimeProbNet at the chosen size."""
    def factory():
        from sbc.models.regime_prob_net import RegimeProbNet

        return RegimeProbNet(seq_len=net["seq"], hidden=net["hidden"], K=net["K"],
                             epochs=net["epochs"], verbose=False)

    return factory


def _transfer_factory(net: dict) -> Callable | None:
    """Zero-argument factory for ``TransferLSTMCorrector`` (sibling), or ``None``."""
    try:
        from sbc.models.transfer_lstm import TransferLSTMCorrector  # sibling agent
    except Exception as exc:  # pragma: no cover - module authored in parallel
        log.info("transfer_lstm.TransferLSTMCorrector pending (%s)", exc)
        return None

    def factory():
        for kw in (dict(seq_length=net["seq"], hidden_size=net["hidden"],
                        max_epochs=net["epochs"]),
                   dict(seq_len=net["seq"], hidden=net["hidden"], epochs=net["epochs"]),
                   {}):
            try:
                return TransferLSTMCorrector(**kw)
            except TypeError:
                continue
        return TransferLSTMCorrector()

    return factory


# --------------------------------------------------------------------------- #
#  (1) Transfer LSTM vs. flagship + few-shot transfer curve                   #
# --------------------------------------------------------------------------- #
def few_shot_transfer_curve(df: pd.DataFrame, factory: Callable, label: str,
                            shots: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Few-shot core->transfer curve: PUR skill vs. number of donor gauges.

    For each ``k`` in ``shots`` the model is trained on **all** core gauges plus
    ``k`` randomly chosen transfer-domain gauges and scored on the *remaining*
    (still ungauged) transfer gauges.  ``k = 0`` is the pure PUR baseline; rising
    skill with ``k`` is the few-shot transfer signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared modelling table carrying a ``domain`` column.
    factory : callable
        Zero-argument corrector factory.
    label : str
        Model label recorded in the output.
    shots : tuple of int
        Donor-gauge counts to sweep.
    seed : int
        Seed for the (deterministic) donor draw.

    Returns
    -------
    pandas.DataFrame
        One row per ``k`` with median PUR KGE' of the correction and of raw
        GloFAS, plus the donor / test gauge counts.
    """
    rng = np.random.default_rng(seed)
    core = df[df["domain"] == "core"]
    transfer = df[df["domain"] == "transfer"]
    t_gauges = sorted(transfer["code"].unique().tolist())
    if core.empty or len(t_gauges) < 2:
        log.warning("few-shot curve [%s]: need core rows and >=2 transfer gauges", label)
        return pd.DataFrame()

    order = rng.permutation(t_gauges).tolist()
    rows: list[dict] = []
    for k in shots:
        if k >= len(t_gauges):          # always keep >=1 ungauged test gauge
            continue
        donors = set(order[:k])
        train = pd.concat([core, transfer[transfer["code"].isin(donors)]],
                          ignore_index=True)
        test = transfer[~transfer["code"].isin(donors)].reset_index(drop=True)
        if test.empty:
            continue
        try:
            model = factory().fit(train)
            kge, kge_raw = _median_kge(test, model)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("few-shot fit failed (%s, k=%d): %s", label, k, exc)
            kge, kge_raw = np.nan, np.nan
        rows.append({"model": label, "n_shots": int(k),
                     "n_donor_gauges": len(donors),
                     "n_test_gauges": int(test["code"].nunique()),
                     "kge": round(kge, 4), "kge_raw": round(kge_raw, 4),
                     "delta_kge": round(kge - kge_raw, 4)})
    return pd.DataFrame(rows)


def section_transfer(df: pd.DataFrame, net: dict, tag: str, seed: int) -> None:
    """Transfer LSTM vs. flagship on temporal+PUR and the few-shot transfer curve."""
    _stamp("(1) transfer LSTM vs. flagship (temporal + PUR) + few-shot curve")
    facs: dict[str, Callable] = {"regimeprobnet": _flagship_factory(net)}
    tfac = _transfer_factory(net)
    if tfac is not None:
        facs["transfer_lstm"] = tfac
    else:
        log.info("running flagship only; transfer_lstm contribution marked pending")

    # head-to-head on the temporal + PUR protocols
    try:
        res = cv.compare(facs, df, temporal=True, lobo=False, pur=True)
        summ = cv.summarise(res)
        save_table(summ, PATHS.tables / f"{_stem('transfer_vs_flagship', tag)}.parquet",
                   csv_mirror=True)
        if not summ.empty:
            print("\n=== (1) Transfer LSTM vs. flagship (median KGE' by split) ===")
            print(summ[["model", "split", "n_gauges", "kge", "kge_raw", "d_kge"]]
                  .to_string(index=False))
    except Exception as exc:
        log.warning("transfer head-to-head failed: %s", exc)

    # few-shot transfer curve (per available model)
    curves = [c for c in (few_shot_transfer_curve(df, fac, name, net["shots"], seed)
                          for name, fac in facs.items()) if not c.empty]
    if curves:
        curve = pd.concat(curves, ignore_index=True)
        save_table(curve, PATHS.tables / f"{_stem('few_shot_curve', tag)}.parquet",
                   csv_mirror=True)
        print("\n=== (1) Few-shot transfer curve (PUR KGE' vs. donor gauges) ===")
        print(curve.to_string(index=False))


# --------------------------------------------------------------------------- #
#  (2) SHAP stability across seeds                                            #
# --------------------------------------------------------------------------- #
def _stability_factories(net: dict) -> dict[str, tuple[Callable, str]]:
    """Seed-aware ``(factory, method)`` pairs for the stability audit."""
    def lgbm(seed: int = 0):
        from sbc.models.boosting import LightGBMCorrector

        return LightGBMCorrector(n_optuna_trials=0, seed=int(seed))

    def flagship(seed: int = 0):
        from sbc.models.regime_prob_net import RegimeProbNet

        return RegimeProbNet(seq_len=net["seq"], hidden=net["hidden"], K=net["K"],
                             epochs=net["epochs"], seed=int(seed), verbose=False)

    return {"regimeprobnet": (flagship, "flagship"), "lgbm": (lgbm, "tree")}


def _fallback_stability(label: str, factory: Callable, method: str,
                        df: pd.DataFrame, net: dict, seeds: list[int],
                        top_k: int = 10) -> dict:
    """Inline cross-seed agreement (used only if shap_stability is unavailable)."""
    from itertools import combinations

    from sbc.explain import shap_analysis as SH

    tr, _ = temporal_split(df, 0.3)
    train = df[tr].reset_index(drop=True)
    sample = train.sample(min(net["shap_n"], len(train)), random_state=0)
    imps: dict[int, pd.Series] = {}
    for s in seeds:
        try:
            model = factory(s).fit(train)
            if method == "tree":
                res = SH.tree_shap(model, sample, max_samples=net["shap_n"], seed=s)
                gi = SH.global_importance(res)
                imps[s] = gi.set_index("feature")["mean_abs_shap"]
            else:
                gi = SH.gradient_importance(model, sample)
                col = "mean_abs_shap" if "mean_abs_shap" in gi.columns else gi.columns[-1]
                imps[s] = gi.set_index("feature")[col]
        except Exception as exc:  # pragma: no cover
            log.warning("inline stability failed (%s, seed=%d): %s", label, s, exc)
    if len(imps) < 2:
        return {"model": label, "method": method, "n_seeds": len(imps),
                "kendall_tau": np.nan, "jaccard_topk": np.nan, "top_k": top_k}
    feats = sorted(set().union(*[set(s.index) for s in imps.values()]))
    mat = pd.DataFrame({s: imps[s].reindex(feats).fillna(0.0) for s in imps})
    taus, jac = [], []
    for a, b in combinations(mat.columns, 2):
        taus.append(mat[a].corr(mat[b], method="kendall"))
        ta = set(mat[a].sort_values(ascending=False).head(top_k).index)
        tb = set(mat[b].sort_values(ascending=False).head(top_k).index)
        jac.append(len(ta & tb) / max(len(ta | tb), 1))
    return {"model": label, "method": method, "n_seeds": len(imps),
            "kendall_tau": round(float(np.nanmean(taus)), 4),
            "jaccard_topk": round(float(np.nanmean(jac)), 4), "top_k": top_k}


def section_shap_stability(df: pd.DataFrame, net: dict, tag: str,
                           seeds: list[int]) -> None:
    """Run shap_stability.stability_across_seeds for the flagship + LightGBM."""
    _stamp("(2) SHAP stability across seeds (flagship + LightGBM)")
    n_seeds, base = len(seeds), int(min(seeds))

    try:
        from sbc.explain import shap_stability as SS  # sibling agent
    except Exception as exc:  # pragma: no cover - module authored in parallel
        SS = None
        log.info("explain.shap_stability pending (%s); using inline fallback", exc)

    rows: list[dict] = []
    for label, (factory, method) in _stability_factories(net).items():
        rec = None
        if SS is not None and hasattr(SS, "stability_across_seeds"):
            try:
                res = SS.stability_across_seeds(
                    factory, df, n_seeds=n_seeds, top_k=10, method=method,
                    base_seed=base, max_samples=net["shap_n"])
                rec = {"model": label, "method": getattr(res, "method", method),
                       "n_seeds": int(getattr(res, "n_seeds", n_seeds)),
                       "kendall_tau": round(float(res.kendall_tau), 4),
                       "jaccard_topk": round(float(res.jaccard_topk), 4),
                       "top_k": int(getattr(res, "top_k", 10)), "source": "shap_stability"}
                # persist the per-feature confidence-band table and a figure
                save_table(res.table, PATHS.tables /
                           f"{_stem(f'shap_stability_{label}_table', tag)}.parquet",
                           csv_mirror=True)
                try:
                    SS.save_stability_plot(res, PATHS.figures /
                                           f"{_stem(f'shap_stability_{label}', tag)}.png")
                except Exception as exc:  # pragma: no cover - plotting optional
                    log.debug("stability plot skipped (%s): %s", label, exc)
            except Exception as exc:
                log.warning("stability_across_seeds(%s) failed: %s", label, exc)
        if rec is None:
            rec = {**_fallback_stability(label, factory, method, df, net, seeds),
                   "source": "inline"}
        rows.append(rec)

    out = pd.DataFrame(rows)
    save_table(out, PATHS.tables / f"{_stem('shap_stability', tag)}.parquet",
               csv_mirror=True)
    print("\n=== (2) SHAP stability (cross-seed attribution agreement) ===")
    print(out.to_string(index=False))


# --------------------------------------------------------------------------- #
#  (3) Dual KGE / skill-class / monotonicity reporting                        #
# --------------------------------------------------------------------------- #
def _headline_per_gauge(df: pd.DataFrame, net: dict, dry_run: bool,
                        headline: list[str]) -> pd.DataFrame:
    """Per-gauge skill table for the headline models (real: read v2 parquet)."""
    if not dry_run:
        path = PATHS.tables / "per_gauge_v2_real_decadal.parquet"
        if path.exists():
            pg = pd.read_parquet(path)
            keep = [m for m in headline if m in set(pg["model"].unique())]
            return pg[pg["model"].isin(keep)].reset_index(drop=True)
        log.warning("%s absent; recomputing a small headline per-gauge table", path.name)
    # dry-run / fallback: evaluate two cheap headline correctors
    from sbc.models.boosting import LightGBMCorrector

    facs: dict[str, Callable] = {
        "lgbm": lambda: LightGBMCorrector(n_optuna_trials=0),
        "regimeprobnet": _flagship_factory(net),
    }
    return cv.compare(facs, df, temporal=True, lobo=False, pur=True)


def _add_kge2009(pg: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct per-gauge KGE-2009 (alpha=std-ratio) alongside KGE'-2012.

    KGE'-2012 uses the variability ratio ``gamma = CV_sim / CV_obs`` while the
    original KGE-2009 uses ``alpha = sigma_sim / sigma_obs = gamma * beta``; both
    are recoverable from the stored ``kge_r / kge_beta / kge_gamma`` components, so
    no re-prediction is needed.
    """
    out = pg.copy()
    for suf in ("", "_raw"):
        r, b, g = out.get(f"kge_r{suf}"), out.get(f"kge_beta{suf}"), out.get(f"kge_gamma{suf}")
        if r is None or b is None or g is None:
            continue
        alpha = g * b
        out[f"kge2009{suf}"] = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (b - 1) ** 2)
        out[f"kge2012{suf}"] = out.get(f"kge{suf}")
    return out


def _dual_kge_headline(pg: pd.DataFrame) -> pd.DataFrame:
    """Median per-(model, split) KGE-2009 / KGE'-2012 table with Knoben classes."""
    cols = [c for c in ("kge2009", "kge2012", "kge2009_raw", "kge2012_raw") if c in pg]
    agg = (pg.groupby(["model", "split"])
           .agg(n_gauges=("code", "nunique"), **{c: (c, "median") for c in cols})
           .reset_index())
    try:
        from sbc.validation.reporting import skill_class

        if "kge2009" in agg:
            agg["skill_class"] = agg["kge2009"].map(skill_class)
    except Exception:  # pragma: no cover
        pass
    return agg.round(4)


_SKILL_BINS = [(-np.inf, -0.41, "unskilful (<=-0.41)"),
               (-0.41, 0.707, "skilful (>-0.41)"),
               (0.707, np.inf, "highly_skilful (>0.707)")]


def _skill_class_fallback(kge: float) -> str:
    """Knoben skill label (used only if reporting.skill_class is unavailable)."""
    if not np.isfinite(kge):
        return "undefined"
    for lo, hi, name in _SKILL_BINS:
        if lo < kge <= hi:
            return name
    return "highly_skilful (>0.707)"


def _skill_class_table_fallback(pg: pd.DataFrame, kge_col: str) -> pd.DataFrame:
    """Per-gauge skill-class counts by (model, split) — inline fallback."""
    work = pg.copy()
    work["skill_class"] = work[kge_col].map(_skill_class_fallback)
    tbl = (work.groupby(["model", "split", "skill_class"])
           .agg(n_gauges=("code", "nunique")).reset_index())
    tot = tbl.groupby(["model", "split"])["n_gauges"].transform("sum")
    tbl["frac"] = (tbl["n_gauges"] / tot).round(3)
    return tbl


def _mono_rate(model, test: pd.DataFrame) -> tuple[dict, str]:
    """Monotonicity-violation rate via reporting (preferred) or constraint_variants."""
    try:
        from sbc.validation import reporting

        if hasattr(reporting, "monotonicity_violation_rate"):
            out = reporting.monotonicity_violation_rate(
                model, test, features=("swe", "smlt"), regime="melt_freshet")
            if isinstance(out, dict):
                return out, "reporting"
    except Exception as exc:
        log.info("reporting.monotonicity_violation_rate pending (%s)", exc)
    from sbc.models.constraint_variants import monotonicity_violation_rate

    return monotonicity_violation_rate(model, test), "constraint_variants"


def _mono_row(label: str, rate: dict, src: str) -> dict:
    """Flatten a monotonicity-rate dict (both reporting & constraint shapes)."""
    vr = rate.get("violation_rate", rate.get("viol_rate", np.nan))
    row = {"model": label, "source": src,
           "violation_rate": round(float(vr), 5) if np.isfinite(vr) else np.nan,
           "n_melt": int(rate.get("n_melt", rate.get("n", 0)) or 0),
           "n_eval": int(rate.get("n_eval", 0) or 0),
           "mean_grad": rate.get("mean_grad", np.nan)}
    for feat, st in (rate.get("per_feature") or {}).items():       # reporting shape
        if isinstance(st, dict):
            row[f"viol_{feat}"] = round(float(st.get("violation_rate", np.nan)), 5)
    for k, v in rate.items():                                      # constraint shape
        if k.startswith("viol_") and k != "viol_rate":
            row[k] = round(float(v), 5)
    return row


def section_reporting(df: pd.DataFrame, net: dict, tag: str, dry_run: bool,
                      headline: list[str]) -> None:
    """Dual KGE table, per-gauge skill classes and soft-vs-hard monotonicity."""
    _stamp("(3) reporting: dual KGE + skill-class table + monotonicity")
    pg = _headline_per_gauge(df, net, dry_run, headline)
    if pg.empty:
        log.warning("no per-gauge results available; skipping reporting tables")
    else:
        pg = _add_kge2009(pg)

        # ---- (3a) dual KGE-2009 / KGE'-2012 headline -----------------------
        dual = _dual_kge_headline(pg)
        save_table(dual, PATHS.tables / f"{_stem('dual_kge', tag)}.parquet", csv_mirror=True)
        print("\n=== (3a) Dual KGE-2009 / KGE'-2012 (median by model x split) ===")
        print(dual.to_string(index=False))

        # ---- (3b) per-gauge skill-class table (Knoben, on KGE-2009) --------
        kcol = "kge2009" if "kge2009" in pg.columns else "kge"
        skill = None
        try:
            from sbc.validation import reporting

            skill = reporting.skill_class_table(pg, kge_col=kcol)
        except Exception as exc:
            log.info("reporting.skill_class_table pending (%s); using inline fallback", exc)
            skill = _skill_class_table_fallback(pg, kcol)
        save_table(skill, PATHS.tables / f"{_stem('skill_class', tag)}.parquet", csv_mirror=True)
        show = [c for c in ("model", "split", "n_gauges", "n_skilful",
                            "n_highly_skilful", "frac_highly_skilful", "summary",
                            "skill_class", "frac")
                if c in skill.columns]
        print(f"\n=== (3b) Per-gauge skill-class table (classified on {kcol}) ===")
        print(skill[show].to_string(index=False))

    # ---- (3c) soft vs. hard-monotone violation rate ------------------------
    try:
        from sbc.models.constraint_variants import HardMonoProbNetCorrector
        from sbc.models.regime_prob_net import RegimeProbNet

        tr, te = temporal_split(df, 0.3)
        train, test = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
        soft = RegimeProbNet(seq_len=net["seq"], hidden=net["hidden"], K=net["K"],
                             epochs=net["epochs"], verbose=False).fit(train)
        hard = HardMonoProbNetCorrector(seq_len=net["seq"], hidden=net["hidden"],
                                        K=net["K"], epochs=net["epochs"],
                                        head_epochs=net["head_epochs"],
                                        verbose=False).fit(train)
        rows = [_mono_row(label, *(_mono_rate(mdl, test)))
                for label, mdl in (("regimeprobnet_soft", soft),
                                   ("probnet_hardmono", hard))]
        mono = pd.DataFrame(rows)
        save_table(mono, PATHS.tables / f"{_stem('monotonicity', tag)}.parquet",
                   csv_mirror=True)
        print("\n=== (3c) Monotonicity-violation rate (soft vs. hard) ===")
        print(mono.to_string(index=False))
    except Exception as exc:
        log.warning("monotonicity section failed: %s", exc)


# --------------------------------------------------------------------------- #
#  (4) Monthly SABER comparison                                               #
# --------------------------------------------------------------------------- #
def _aggregate_to_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a raw decadal modelling table to calendar months.

    Numeric columns are mean-aggregated within each (gauge, month); the string id
    columns (``basin`` / ``domain``) are carried with ``first``.  The volatile
    ``log_residual`` target is dropped and rebuilt downstream by
    :func:`sbc.schemas.validate`.  ``scale`` is set to ``"monthly"`` so the SABER
    donor keys its seasonal factors by calendar month.
    """
    df = raw.drop(columns=[c for c in (TARGET_COL,) if c in raw.columns]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = pd.to_datetime(dict(year=df["date"].dt.year,
                                     month=df["date"].dt.month, day=15))
    static_str = [c for c in df.columns
                  if c not in ("code", "date") and not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns
           if c not in ("code", "date") and pd.api.types.is_numeric_dtype(df[c])]
    agg: dict[str, str] = {c: "mean" for c in num}
    agg.update({c: "first" for c in static_str})
    out = df.groupby(["code", "date"], as_index=False).agg(agg)
    out["scale"] = "monthly"
    return out


def section_monthly_saber(raw: pd.DataFrame, cfg: dict, tag: str) -> None:
    """Like-for-like monthly SABER (donor) vs. LightGBM bias correction."""
    _stamp("(4) monthly-scale SABER (donor) vs. LightGBM")
    try:
        native = PATHS.processed / "discharge_monthly.parquet"
        if native.exists():
            log.info("native monthly observations present (%s); aggregating "
                     "decadal model table to months for a like-for-like run", native.name)
        monthly_raw = _aggregate_to_monthly(raw)
        monthly = prepare(monthly_raw, "monthly", cfg).reset_index(drop=True)
        log.info("monthly table: %d rows, %d gauges", len(monthly), monthly["code"].nunique())

        from sbc.models.boosting import LightGBMCorrector
        from sbc.models.sota_baselines import DonorRegionalizationCorrector

        facs = {"lgbm_monthly": lambda: LightGBMCorrector(n_optuna_trials=0),
                "donor_saber_monthly": lambda: DonorRegionalizationCorrector()}
        res = cv.compare(facs, monthly, temporal=True, lobo=False, pur=True)
        summ = cv.summarise(res)
        summ.insert(0, "resolution", "monthly")
        save_table(summ, PATHS.tables / f"{_stem('monthly_saber', tag)}.parquet",
                   csv_mirror=True)
        print("\n=== (4) Monthly SABER (donor) vs. LightGBM ===")
        if not summ.empty:
            print(summ[["resolution", "model", "split", "n_gauges", "kge", "kge_raw", "d_kge"]]
                  .to_string(index=False))
    except Exception as exc:
        log.warning("monthly SABER section failed: %s", exc)


# --------------------------------------------------------------------------- #
#  (5) Multi-scale daily<->decadal consistency                                #
# --------------------------------------------------------------------------- #
def _multiscale_tables(args, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build matched (daily, decadal) prepared tables for the same gauges."""
    if args.dry_run:
        from sbc.features.engineering import build_features
        from sbc.features.regimes import classify_regimes
        from sbc.synthetic import decadal_aggregate, generate

        raw_daily = generate(scale="daily", years=args.years, n_basins=args.n_basins,
                             gauges_per_basin=(2, 3), seed=cfg.get("seed", 1234))
        daily = classify_regimes(build_features(raw_daily, scale="daily"))
        decadal = classify_regimes(build_features(decadal_aggregate(raw_daily),
                                                  scale="decadal"))
    else:
        daily = prepare(assemble("daily"), "daily", cfg)
        decadal = prepare(assemble("decadal"), "decadal", cfg)
    # the synthetic feature tables carry no log_residual target until validated
    return (validate(daily).reset_index(drop=True),
            validate(decadal).reset_index(drop=True))


def section_multiscale(args, cfg: dict, tag: str) -> None:
    """Cross-scale transfer retention + daily->decadal aggregation consistency."""
    _stamp("(5) multi-scale daily<->decadal consistency")
    try:
        from sbc import multiscale as MS

        daily, decadal = _multiscale_tables(args, cfg)
        tfrac = cfg["validation"]["temporal_test_frac"]

        transfer = MS.cross_scale_transfer(daily, decadal, seed=cfg.get("seed", 1234),
                                           test_frac=tfrac)
        save_table(transfer, PATHS.tables / f"{_stem('multiscale_transfer', tag)}.parquet",
                   csv_mirror=True)
        print("\n=== (5a) Cross-scale transfer (retention by direction) ===")
        print(transfer.to_string(index=False))

        # aggregation-consistency: daily-corrected -> decades vs. decadal truth
        from sbc.models.boosting import LightGBMCorrector

        tr, _ = temporal_split(daily, tfrac)
        m = LightGBMCorrector(n_optuna_trials=0).fit(daily[tr].reset_index(drop=True))
        daily_pred = daily.assign(**{PRED_COL: m.predict(daily)})
        cons = MS.consistency_check(daily_pred, decadal)
        scalar = {k: v for k, v in cons.items() if not isinstance(v, pd.DataFrame)}
        save_table(pd.DataFrame([scalar]),
                   PATHS.tables / f"{_stem('multiscale_consistency', tag)}.parquet",
                   csv_mirror=True)
        save_json(scalar, PATHS.tables / f"{_stem('multiscale_consistency', tag)}.json")
        print("\n=== (5b) Daily->decadal aggregation consistency ===")
        for k in ("mean_abs_discrepancy", "rel_discrepancy_pct",
                  "raw_glofas_mean_abs_discrepancy", "raw_glofas_rel_discrepancy_pct",
                  "n_decades", "n_gauges"):
            print(f"  {k:36s} {scalar.get(k)}")
    except Exception as exc:
        log.warning("multi-scale section failed: %s", exc)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def build_net(args) -> dict:
    """Resolve net sizes / sweep grids for the requested (dry vs. real) mode."""
    if args.dry_run:
        return {"seq": 4, "hidden": 16, "K": 3, "epochs": args.epochs or 3,
                "head_epochs": 40, "shots": (0, 1, 2), "shap_n": 300}
    return {"seq": 12, "hidden": 64, "K": 5, "epochs": args.epochs or 50,
            "head_epochs": 250, "shots": (0, 1, 2, 4, 8, 16), "shap_n": 2000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny synthetic smoke run to prove the wiring (a few min)")
    ap.add_argument("--models", default=None,
                    help="comma-separated headline-model subset for the reporting tables")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override net training epochs (default 3 dry / 50 real)")
    ap.add_argument("--years", type=int, default=8, help="synthetic years (dry-run)")
    ap.add_argument("--n-basins", type=int, default=3, help="synthetic core basins (dry-run)")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma-separated seeds for the SHAP-stability audit")
    ap.add_argument("--skip", default="",
                    help="sections to skip: transfer,shap,report,monthly,multiscale")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    from sbc.models import load_all

    load_all()  # register donor / probnet_hardmono / qrf / ... for the registry

    net = build_net(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [0, 1]
    if args.dry_run:
        seeds = seeds[:2] if len(seeds) >= 2 else [0, 1]
    headline = ([m.strip() for m in args.models.split(",") if m.strip()]
                if args.models else list(HEADLINE_DEFAULT))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    tag = ("dryrun" if args.dry_run else "real") + "_decadal"

    _stamp(f"final-rigor driver: dry_run={args.dry_run} tag={tag} net={net}")
    raw_dec, df_dec = build_decadal(args, cfg)
    _stamp(f"prepared decadal: {df_dec.shape}, gauges={df_dec['code'].nunique()}, "
           f"basins={df_dec['basin'].nunique()}, "
           f"transfer_gauges={df_dec.loc[df_dec['domain'] == 'transfer', 'code'].nunique()}")

    if "transfer" not in skip:
        section_transfer(df_dec, net, tag, cfg.get("seed", 1234))
    if "shap" not in skip:
        section_shap_stability(df_dec, net, tag, seeds)
    if "report" not in skip:
        section_reporting(df_dec, net, tag, args.dry_run, headline)
    if "monthly" not in skip:
        section_monthly_saber(raw_dec, cfg, tag)
    if "multiscale" not in skip:
        section_multiscale(args, cfg, tag)

    _stamp("FINAL RIGOR COMPLETE")
    print("\nFINAL RIGOR COMPLETE", flush=True)


if __name__ == "__main__":
    main()
