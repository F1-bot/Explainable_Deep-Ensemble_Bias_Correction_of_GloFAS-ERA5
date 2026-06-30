"""One-off Optuna (TPE) hyper-parameter tuning of the boosting correctors.

Per-fold tuning across the whole validation matrix is prohibitively expensive, so
the matrix experiment uses strong defaults and the Bayesian optimisation is
reported once here on the temporal-holdout split.  The tuned parameters and the
KGE' gain over the default configuration are written to ``results/tables``.

Usage::
    PYTHONPATH=src python scripts/tune_boosting.py --backend lgbm --trials 40 --scale decadal
"""
from __future__ import annotations

import argparse

import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import load_config, prepare
from sbc.models.boosting import BoostingCorrector
from sbc.utils import get_logger, save_json
from sbc.validation import metrics as M
from sbc.validation.splits import temporal_split

log = get_logger("tune")


def _median_gauge_kge(df: pd.DataFrame, pred) -> float:
    from sbc.schemas import SIM_COL, back_transform

    d = df.assign(q_pred=back_transform(df[SIM_COL].to_numpy(float), pred))
    k = [M.kge_prime(g["q_obs"].values, g["q_pred"].values)["kge"]
         for _, g in d.groupby("code") if len(g) >= 5]
    return float(pd.Series(k).median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lgbm", choices=["lgbm", "xgb", "catboost"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--scale", default="decadal")
    a = ap.parse_args()

    cfg = load_config()
    df = prepare(assemble(a.scale), a.scale, cfg).reset_index(drop=True)
    tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
    train, test = df[tr], df[te]

    default = BoostingCorrector(backend=a.backend, n_optuna_trials=0).fit(train)
    kge_def = _median_gauge_kge(test, default.predict_residual(test))

    tuned = BoostingCorrector(backend=a.backend, n_optuna_trials=a.trials).fit(train)
    kge_tuned = _median_gauge_kge(test, tuned.predict_residual(test))

    result = {"backend": a.backend, "scale": a.scale, "trials": a.trials,
              "kge_default": round(kge_def, 4), "kge_tuned": round(kge_tuned, 4),
              "delta": round(kge_tuned - kge_def, 4),
              "best_params": getattr(tuned, "best_params_", getattr(tuned, "params", {}))}
    log.info("tuned %s: KGE' %.3f -> %.3f (d=%+.3f)",
             a.backend, kge_def, kge_tuned, kge_tuned - kge_def)
    save_json(result, PATHS.tables / f"optuna_{a.backend}_{a.scale}.json")
    print(result)


if __name__ == "__main__":
    main()
