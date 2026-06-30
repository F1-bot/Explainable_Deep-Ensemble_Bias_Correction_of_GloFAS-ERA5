"""Consolidate all real runs into ONE canonical, traceable, honestly-scoped
results table: per-gauge medians with bootstrap CIs, win-fraction vs raw,
skilful / highly-skilful counts (Hunt idiom), dual KGE2009/KGE'2012, and a
monthly-scale SABER-donor comparison row. Closes the 'tables-vs-claims drift',
'PUR with CIs' and 'skill-class' blockers from the verification round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.utils import get_logger, save_table, seed_everything

log = get_logger("consolidate")
SKILFUL = 1.0 - np.sqrt(2.0)   # -0.414 (Knoben/Hunt 'skilful')
HIGHLY = np.sqrt(2.0) / 2.0    # 0.707 ('highly skilful')


def _boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def build_canonical():
    """Merge the original real run + the v2 run into one per-gauge table."""
    real = pd.read_parquet(PATHS.tables / "per_gauge_real_decadal.parquet")
    v2 = pd.read_parquet(PATHS.tables / "per_gauge_v2_real_decadal.parquet")
    # original run is canonical for its models (it also has LOBO); take from v2
    # only the models that are NOT in the original run.
    only_v2 = set(v2["model"]) - set(real["model"])
    canon = pd.concat([real, v2[v2["model"].isin(only_v2)]], ignore_index=True)
    save_table(canon, PATHS.tables / "per_gauge_canonical_decadal.parquet", csv_mirror=True)
    log.info("canonical per-gauge: %d rows, models=%s",
             len(canon), sorted(canon["model"].unique()))
    return canon


def summarise_canonical(canon):
    rows = []
    for (model, split), g in canon.groupby(["model", "split"]):
        kge = g["kge"].dropna()
        d = (g["kge"] - g["kge_raw"]).dropna()
        lo, hi = _boot_ci(kge.values)
        rows.append({
            "model": model, "split": split, "n_gauges": int(g["code"].nunique()),
            "kge_raw_median": round(float(g["kge_raw"].median()), 3),
            "kge_median": round(float(kge.median()), 3),
            "kge_ci_lo": round(lo, 3), "kge_ci_hi": round(hi, 3),
            "d_kge_median": round(float(d.median()), 3),
            "win_frac_vs_raw": round(float((g["kge"] > g["kge_raw"]).mean()), 3),
            "skilful_k": int((kge > SKILFUL).sum()),
            "highly_skilful_k": int((kge > HIGHLY).sum()),
            "nse_neg_k": int((g["nse"] < 0).sum()) if "nse" in g else np.nan,
            "crps_median": round(float(g["crps"].median()), 3) if "crps" in g and g["crps"].notna().any() else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["split", "kge_median"], ascending=[True, False])
    save_table(out, PATHS.tables / "summary_canonical_decadal.parquet", csv_mirror=True)
    log.info("CANONICAL SUMMARY:\n%s", out.to_string(index=False))
    return out


def dual_kge_headline(cfg):
    """KGE2009 vs KGE'2012 for raw GloFAS and one refit headline (lgbm)."""
    from sbc.data.assemble import assemble
    from sbc.experiment import prepare
    from sbc.validation.reporting import dual_kge
    from sbc.validation.splits import temporal_split
    from sbc.schemas import OBS_COL, SIM_COL, back_transform
    df = prepare(assemble("decadal"), "decadal", cfg).reset_index(drop=True)
    tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
    test = df[te]
    rows = [{"series": "raw_glofas", **dual_kge(test[OBS_COL].values, test[SIM_COL].values)}]
    try:
        from sbc.models.boosting import LightGBMCorrector
        m = LightGBMCorrector(n_optuna_trials=0).fit(df[tr])
        pred = back_transform(test[SIM_COL].values, m.predict_residual(test))
        rows.append({"series": "lgbm_corrected", **dual_kge(test[OBS_COL].values, pred)})
    except Exception as exc:
        log.warning("dual-kge lgbm: %s", exc)
    out = pd.DataFrame(rows)
    save_table(out, PATHS.tables / "dual_kge_real_decadal.parquet", csv_mirror=True)
    log.info("DUAL KGE (2009 vs 2012):\n%s", out.to_string(index=False))


def monthly_saber(cfg):
    """Aggregate decadal->monthly and run the SABER-style donor + lgbm + raw."""
    from sbc.data.assemble import assemble
    from sbc.experiment import prepare
    from sbc.validation import cv
    df = prepare(assemble("decadal"), "decadal", cfg)
    static_cols = [c for c in df.columns if df.groupby("code")[c].nunique(dropna=False).max() <= 1
                   and c not in ("code", "basin", "domain", "scale")]
    keep_static = [c for c in static_cols if df[c].dtype.kind in "fc"]
    g = df.assign(ym=df["date"].dt.to_period("M"))
    agg = {"q_obs": "mean", "q_glofas": "mean", "basin": "first", "domain": "first"}
    agg.update({c: "first" for c in keep_static})
    m = g.groupby(["code", "ym"], as_index=False).agg(agg)
    m["date"] = m["ym"].dt.to_timestamp(); m = m.drop(columns="ym"); m["scale"] = "monthly"
    from sbc.schemas import make_target
    m["log_residual"] = make_target(m["q_obs"], m["q_glofas"])
    from sbc.models.sota_baselines import DonorRegionalizationCorrector
    from sbc.models.boosting import LightGBMCorrector
    facs = {"donor_saber": lambda: DonorRegionalizationCorrector(k=5),
            "lgbm": lambda: LightGBMCorrector(n_optuna_trials=0)}
    res = cv.compare(facs, m, temporal=True, lobo=False, pur=True)
    summ = cv.summarise(res); summ.insert(0, "scale", "monthly")
    save_table(summ, PATHS.tables / "monthly_saber_real.parquet", csv_mirror=True)
    log.info("MONTHLY (SABER like-for-like, vs published SABER 0.47):\n%s", summ.to_string(index=False))


def main():
    cfg_path = PATHS.configs / "default.yaml"
    import yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    canon = build_canonical()
    summarise_canonical(canon)
    for fn in (dual_kge_headline, monthly_saber):
        try:
            fn(cfg)
        except Exception as exc:
            log.warning("%s failed: %s", fn.__name__, exc)
    print("CONSOLIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
