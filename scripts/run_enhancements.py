"""Substance-strengthening experiments on the real decadal data.

Implements the tractable, high-value items from the improvement roadmap on a
single temporal(+PUR) split (never the 4 h LOBO matrix):

* P3  statistical rigour   — paired Wilcoxon + bootstrap CI + win-fraction;
* P4  component ablations  — snow / static / residual-target / regime-gating /
                             physics-penalty (sbc.ablation);
* P1  stacked ensemble     — the missing headline contribution;
* P7  deep ensemble        — multi-seed flagship for PUR robustness;
* P9  calibration          — reliability / PIT / coverage / sharpness / CRPSS;
* P12 gate-vs-regime       — proves the MoE experts specialise by regime.

Run::  PYTHONPATH=src python scripts/run_enhancements.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import build_factories, load_config, prepare
from sbc.utils import get_logger, save_table, seed_everything
from sbc.validation import cv

log = get_logger("enh")
T0 = time.time()
TAG = "real_decadal"


def _stamp(msg):
    log.info("[%5.0fs] %s", time.time() - T0, msg)


def significance(df_pg: pd.DataFrame, ref="regimeprobnet", n_boot=2000, seed=0):
    """P3: per-split paired Wilcoxon of `ref` vs each model + bootstrap median CI."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    rows = []
    for split, g in df_pg.groupby("split"):
        wide = g.pivot_table(index="code", columns="model", values="kge")
        models = [m for m in wide.columns if m != ref]
        for m in models:
            pair = wide[[ref, m]].dropna()
            if len(pair) < 5:
                continue
            try:
                p = wilcoxon(pair[ref], pair[m]).pvalue
            except Exception:
                p = np.nan
            win = float((pair[ref] > pair[m]).mean())
            rows.append({"split": split, "ref": ref, "vs": m, "n": len(pair),
                         "median_ref": float(pair[ref].median()),
                         "median_vs": float(pair[m].median()),
                         "median_diff": float((pair[ref] - pair[m]).median()),
                         "win_fraction": round(win, 3),
                         "wilcoxon_p": round(float(p), 4)})
        # bootstrap median-KGE CI per model
        for m in wide.columns:
            v = wide[m].dropna().values
            if len(v) < 5:
                continue
            boot = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
            rows.append({"split": split, "ref": "(CI)", "vs": m, "n": len(v),
                         "median_vs": round(float(np.median(v)), 3),
                         "ci_lo": round(float(np.percentile(boot, 2.5)), 3),
                         "ci_hi": round(float(np.percentile(boot, 97.5)), 3)})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    _stamp("loading + preparing real decadal data")
    df = prepare(assemble("decadal"), "decadal", cfg).reset_index(drop=True)
    _stamp(f"prepared {df.shape}, gauges={df.code.nunique()}")

    # ---- P3 significance (cheap, from existing per-gauge table) -------------
    try:
        pg = pd.read_parquet(PATHS.tables / f"per_gauge_{TAG}.parquet")
        sig = significance(pg)
        save_table(sig, PATHS.tables / f"significance_{TAG}.parquet", csv_mirror=True)
        _stamp("P3 significance saved")
    except Exception as exc:
        log.warning("P3 failed: %s", exc)

    # ---- P4 component ablations --------------------------------------------
    try:
        from sbc.ablation import run_ablations
        abl = run_ablations(df, scale="decadal", splits=("temporal", "pur"),
                            seed=cfg["seed"],
                            flagship_kwargs=dict(epochs=30, hidden=64, seq_len=12,
                                                 expert_hidden=32, gate_hidden=32))
        save_table(abl, PATHS.tables / f"ablation_{TAG}.parquet", csv_mirror=True)
        _plot_ablation(abl)
        _stamp("P4 ablations saved")
    except Exception as exc:
        log.warning("P4 failed: %s", exc)

    # ---- P1 + P7 stacked ensemble + deep ensemble --------------------------
    try:
        from sbc.models.robust import DeepEnsembleCorrector
        from sbc.models.boosting import LightGBMCorrector, CatBoostCorrector
        from sbc.models.regime_prob_net import RegimeProbNet
        from sbc.models.ensemble import StackedEnsemble

        def _stacked():
            return StackedEnsemble(
                [LightGBMCorrector(n_optuna_trials=0), CatBoostCorrector(n_optuna_trials=0),
                 RegimeProbNet(seq_len=12, hidden=64, epochs=25)], meta="nnls")

        def _deepens():
            return DeepEnsembleCorrector(base="probnet", n_members=3,
                                         member_kwargs=dict(epochs=25, hidden=64, seq_len=12))

        res = cv.compare({"stacked": _stacked, "deepens": _deepens}, df,
                         temporal=True, lobo=False, pur=True)
        save_table(res, PATHS.tables / f"per_gauge_enhanced_{TAG}.parquet", csv_mirror=True)
        save_table(cv.summarise(res), PATHS.tables / f"summary_enhanced_{TAG}.parquet",
                   csv_mirror=True)
        _stamp("P1+P7 stacked/deepens saved\n" + cv.summarise(res).to_string(index=False))
    except Exception as exc:
        log.warning("P1/P7 failed: %s", exc)

    # ---- P9 calibration + P12 gate-vs-regime -------------------------------
    try:
        _calibration_and_gate(df, cfg)
        _stamp("P9+P12 calibration/gate saved")
    except Exception as exc:
        log.warning("P9/P12 failed: %s", exc)

    _stamp("ENHANCEMENTS COMPLETE")


def _plot_ablation(abl: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = abl[abl.split == "temporal"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(sub["study"], sub["delta_kge"], color="teal")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ΔKGE′ = full − ablated  (positive ⇒ component helps)")
    ax.set_title("Component ablations (real decadal, temporal holdout)")
    fig.tight_layout(); fig.savefig(PATHS.figures / f"ablation_{TAG}.png", dpi=160)
    plt.close(fig)


def _calibration_and_gate(df, cfg):
    from sbc.models.regime_prob_net import RegimeProbNet
    from sbc.validation import calibration as C
    from sbc.validation.metrics import crps_ensemble, crps_gaussian
    from sbc.validation.splits import temporal_split
    from sbc.schemas import OBS_COL, make_target

    tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
    train, test = df[tr], df[te]
    model = RegimeProbNet(seq_len=12, hidden=64, epochs=40).fit(train)

    levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    rows = []
    for name, part in [("temporal", test), ("pur", df[df.domain == "transfer"])]:
        if len(part) < 20:
            continue
        q = model.predict_discharge_quantiles(part, tuple(levels))
        obs = part[OBS_COL].to_numpy(float)
        crps_m = crps_ensemble(obs, q)
        # climatological Gaussian reference in residual space -> discharge units
        mu0, sd0 = float(make_target(train[OBS_COL], train["q_glofas"]).mean()), \
            float(make_target(train[OBS_COL], train["q_glofas"]).std())
        crps_ref = crps_gaussian(make_target(obs, part["q_glofas"]),
                                 np.full(len(obs), mu0), np.full(len(obs), sd0))
        summ = C.calibration_summary(obs, q, levels, crps_model=crps_m, crps_ref=crps_ref)
        summ.update({"split": name, "n": len(obs)})
        rows.append(summ)
        try:
            C.save_reliability_diagram(obs, q, levels,
                                       path=PATHS.figures / f"calibration_reliability_{name}.png")
            C.save_pit_histogram(obs, q, levels,
                                 path=PATHS.figures / f"calibration_pit_{name}.png")
        except Exception as exc:
            log.debug("calib plot skipped (%s): %s", name, exc)
    if rows:
        save_table(pd.DataFrame(rows), PATHS.tables / f"calibration_{TAG}.parquet",
                   csv_mirror=True)

    # P12 gate-vs-regime alignment
    try:
        gw = model.gate_weights(test) if hasattr(model, "gate_weights") else None
        if gw is not None and "regime" in test.columns:
            expert = np.asarray(gw).argmax(axis=1)
            conf = pd.crosstab(test["regime"].values, expert,
                               rownames=["regime"], colnames=["expert"])
            save_table(conf.reset_index(), PATHS.tables / f"gate_alignment_{TAG}.parquet",
                       csv_mirror=True)
    except Exception as exc:
        log.warning("gate alignment skipped: %s", exc)


if __name__ == "__main__":
    main()
