"""v2 extras on real decadal data: teleconnection gain + flagship SHAP/IG + ALE."""
from __future__ import annotations

import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import load_config, prepare
from sbc.utils import get_logger, save_table, seed_everything
from sbc.validation import cv
from sbc.validation.splits import temporal_split

log = get_logger("extras")


def main():
    cfg = load_config()
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    raw = assemble("decadal")
    df0 = prepare(raw.copy(), "decadal", cfg)

    # ---- teleconnection skill comparison (LightGBM, fast) -------------------
    frames = []
    from sbc.models.boosting import LightGBMCorrector
    configs = [("no_tele", df0)]
    try:
        from sbc.data.teleconnections import add_teleconnections
        try:
            rawt = add_teleconnections(raw.copy())
        except TypeError:
            rawt = add_teleconnections(raw.copy(), "decadal")
        configs.append(("with_tele", prepare(rawt, "decadal", cfg)))
        log.info("teleconnection features added: %d new cols",
                 prepare(rawt, "decadal", cfg).shape[1] - df0.shape[1])
    except Exception as exc:
        log.warning("teleconnections unavailable: %s", exc)
    for name, d in configs:
        res = cv.run_matrix(lambda: LightGBMCorrector(n_optuna_trials=0), d,
                            f"lgbm_{name}", temporal=True, lobo=False, pur=True)
        s = cv.summarise(res)
        s.insert(0, "config", name)
        frames.append(s)
    tele = pd.concat(frames, ignore_index=True)
    save_table(tele, PATHS.tables / "teleconnection_comparison.parquet", csv_mirror=True)
    log.info("teleconnection comparison:\n%s",
             tele[["config", "split", "kge", "kge_raw"]].to_string(index=False))

    # ---- SHAP / IG / ALE on the FLAGSHIP (not a LightGBM proxy) -------------
    from sbc.explain import flagship_xai as FX, shap_analysis as SH
    from sbc.models.regime_prob_net import RegimeProbNet

    df0 = df0.reset_index(drop=True)
    tr, te = temporal_split(df0, cfg["validation"]["temporal_test_frac"])
    flag = RegimeProbNet(seq_len=12, hidden=64, epochs=50).fit(df0[tr])
    test = df0[te].reset_index(drop=True)
    sample = test.sample(min(1500, len(test)), random_state=0).reset_index(drop=True)

    try:
        ds = FX.deep_shap(flag, sample, background=200)
    except Exception as exc:
        log.warning("deep_shap failed (%s); falling back to integrated_gradients", exc)
        ig = FX.integrated_gradients(flag, sample)
        ds = {"shap_values": ig["attributions"], "base_value": ig.get("base_value", 0.0),
              "features": ig["features"], "X": ig["X"]}
    gi = SH.global_importance(ds)
    save_table(gi, PATHS.tables / "flagship_shap_importance.parquet", csv_mirror=True)
    log.info("FLAGSHIP global attribution (top 12):\n%s", gi.head(12).to_string(index=False))
    if "regime" in sample.columns:
        rc = SH.regime_conditional_importance(ds, sample["regime"])
        save_table(rc, PATHS.tables / "flagship_shap_regime.parquet", csv_mirror=True)
    try:
        SH.save_beeswarm(ds, PATHS.shap_dir / "flagship_beeswarm.png")
    except Exception as exc:
        log.debug("flagship beeswarm skipped: %s", exc)

    for feat in ["swe", "smlt", "t2m_mean"]:
        if feat in sample.columns:
            try:
                al = FX.ale(flag, sample, feat, bins=20)
                FX.save_ale_plot(al, model_name="regimeprobnet")
            except Exception as exc:
                log.debug("ALE %s skipped: %s", feat, exc)
    print("EXTRAS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
