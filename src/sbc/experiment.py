"""Top-level, config-driven experiment runner for the sbc framework.

Pipeline: load data (synthetic or assembled real) -> engineer features and
classify hydrological regimes -> evaluate the baseline ladder, the boosting trio,
the EA-LSTM and the flagship RegimeProbNet (and optionally the stacked ensemble)
across the temporal / LOBO / PUR validation matrix -> aggregate skill tables ->
run SHAP attribution. All artefacts are written under ``results/``.

Run::
    PYTHONPATH=src python -m sbc.experiment --scale decadal --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .config import PATHS
from .features.engineering import build_features
from .features.regimes import classify_regimes
from .utils import get_logger, save_json, save_table, seed_everything
from .validation import cv

log = get_logger(__name__)

DEFAULT_CONFIG = PATHS.configs / "default.yaml"


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_data(cfg: dict, scale: str, source: str) -> pd.DataFrame:
    if source == "synthetic":
        from .synthetic import generate

        s = cfg["data"]["synthetic"]
        return generate(n_basins=s["n_basins"], gauges_per_basin=tuple(s["gauges_per_basin"]),
                        years=s["years"], scale=scale, seed=s["seed"])
    if source == "real":
        from .data.assemble import assemble

        return assemble(scale)
    raise ValueError(f"unknown data source {source!r}")


def prepare(df: pd.DataFrame, scale: str, cfg: dict) -> pd.DataFrame:
    from .schemas import validate

    df = validate(df)               # guarantees the log_residual target column
    df = build_features(df, scale=scale)
    if cfg.get("regimes", {}).get("enabled", True):
        df = classify_regimes(df)
    return df


# --------------------------------------------------------------------------- #
#  Model factories
# --------------------------------------------------------------------------- #
def build_factories(cfg: dict, scale: str, quick: bool,
                    which: list[str] | None = None) -> dict:
    from .models.boosting import CatBoostCorrector, LightGBMCorrector, XGBoostCorrector
    from .models.ea_lstm import EALSTMCorrector
    from .models.ensemble import StackedEnsemble
    from .models.quantile_mapping import LinearScalingCorrector, QuantileMappingCorrector
    from .models.regime_prob_net import RegimeProbNet

    dcfg = cfg["models"]["deep"]
    seq = dcfg["ealstm"]["seq_len"][scale]
    seqp = dcfg["regimeprobnet"]["seq_len"][scale]
    trials = 0 if quick else cfg["models"]["boosting"]["n_optuna_trials"]
    ep_l = 8 if quick else dcfg["ealstm"]["epochs"]
    ep_p = 8 if quick else dcfg["regimeprobnet"]["epochs"]
    hid = 32 if quick else dcfg["ealstm"]["hidden"]
    pn = dcfg["regimeprobnet"]

    def _ealstm():
        return EALSTMCorrector(seq_length=seq, hidden_size=hid, max_epochs=ep_l)

    def _probnet():
        return RegimeProbNet(seq_len=seqp, hidden=hid, K=pn["n_experts"], epochs=ep_p,
                             lambda_gate=pn["lambda_gate"], lambda_phys=pn["lambda_phys"],
                             loss="crps" if pn.get("crps_loss", True) else "nll")

    facs = {
        "scaling": LinearScalingCorrector,
        "qmap": QuantileMappingCorrector,
        "xgb": lambda: XGBoostCorrector(n_optuna_trials=trials),
        "lgbm": lambda: LightGBMCorrector(n_optuna_trials=trials),
        "catboost": lambda: CatBoostCorrector(n_optuna_trials=trials),
        "ealstm": _ealstm,
        "regimeprobnet": _probnet,
        "stacked": lambda: StackedEnsemble(
            [LightGBMCorrector(n_optuna_trials=0), CatBoostCorrector(n_optuna_trials=0),
             _ealstm(), _probnet()], meta=cfg["models"]["ensemble"]["meta"]),
    }
    if which is not None:
        facs = {k: v for k, v in facs.items() if k in which}
    return facs


# --------------------------------------------------------------------------- #
#  Explainability
# --------------------------------------------------------------------------- #
def run_shap(df: pd.DataFrame, cfg: dict) -> dict:
    from .explain import shap_analysis as X
    from .models.boosting import LightGBMCorrector
    from .validation.splits import temporal_split

    tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
    df = df.reset_index(drop=True)
    model = LightGBMCorrector(n_optuna_trials=0).fit(df[tr])
    res = X.tree_shap(model, df[te], max_samples=cfg["explain"]["max_samples"])
    glob = X.global_importance(res)
    save_table(glob, PATHS.shap_dir / "global_importance.parquet", csv_mirror=True)
    out = {"top_features": glob.head(15).to_dict("records")}
    if cfg["explain"].get("regime_conditional") and "regime" in df.columns:
        reg = X.regime_conditional_importance(res, df[te].reset_index(drop=True)["regime"])
        save_table(reg, PATHS.shap_dir / "regime_importance.parquet", csv_mirror=True)
    try:
        X.save_beeswarm(res, PATHS.shap_dir / "beeswarm.png")
    except Exception as exc:  # pragma: no cover
        log.debug("beeswarm plot skipped: %s", exc)
    return out


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def run_experiment(scale: str = "decadal", source: str | None = None,
                   config: str | Path | None = None, quick: bool = False,
                   models: list[str] | None = None, do_shap: bool = True) -> dict:
    cfg = load_config(config)
    seed_everything(cfg.get("seed", 1234))
    source = source or cfg["data"]["source"]
    PATHS.ensure()

    log.info("loading data (source=%s, scale=%s)", source, scale)
    df = get_data(cfg, scale, source)
    df = prepare(df, scale, cfg)
    log.info("prepared table: %d rows, %d gauges, %d basins",
             len(df), df["code"].nunique(), df["basin"].nunique())

    facs = build_factories(cfg, scale, quick, which=models)
    log.info("evaluating models: %s", list(facs))
    vcfg = cfg["validation"]
    results = cv.compare(facs, df, test_frac=vcfg["temporal_test_frac"])
    summary = cv.summarise(results)

    tag = f"{source}_{scale}" + ("_quick" if quick else "")
    save_table(results, PATHS.tables / f"per_gauge_{tag}.parquet", csv_mirror=True)
    save_table(summary, PATHS.tables / f"summary_{tag}.parquet", csv_mirror=True)
    log.info("\n%s", summary.to_string(index=False))

    shap_out = {}
    if do_shap:
        try:
            shap_out = run_shap(df, cfg)
        except Exception as exc:  # pragma: no cover
            log.warning("SHAP step failed: %s", exc)

    save_json({"scale": scale, "source": source, "quick": quick,
               "summary": summary.to_dict("records"), "shap": shap_out},
              PATHS.tables / f"report_{tag}.json")
    return {"results": results, "summary": summary, "shap": shap_out}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the sbc bias-correction experiment")
    ap.add_argument("--scale", default="decadal", choices=["decadal", "daily"])
    ap.add_argument("--source", default=None, choices=[None, "synthetic", "real"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--models", default=None, help="comma-separated subset of models")
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args()
    models = a.models.split(",") if a.models else None
    run_experiment(scale=a.scale, source=a.source, config=a.config, quick=a.quick,
                   models=models, do_shap=not a.no_shap)


if __name__ == "__main__":
    main()
