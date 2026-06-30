"""Honest full-scale evaluation of the CMAL head vs the Gaussian flagship vs QRF
on the complete UQ suite (CRPS, coverage, Winkler, Alpha-reliability, twCRPS).
Does the Klotz-2022 SOTA UQ head actually improve our calibrated-probabilistic axis?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import load_config, prepare
from sbc.schemas import OBS_COL
from sbc.utils import get_logger, save_table, seed_everything
from sbc.validation import calibration as C, uq_scores as U
from sbc.validation.metrics import crps_ensemble, evaluate
from sbc.validation.splits import temporal_split

log = get_logger("cmal")
LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def _uq_row(name, split, obs, q):
    lo, hi = q[:, 0], q[:, -1]                # 5% / 95% -> 90% PI
    cov = C.coverage(obs, lo, hi) if hasattr(C, "coverage") else float(((obs >= lo) & (obs <= hi)).mean())
    row = {"model": name, "split": split, "n": len(obs),
           "kge": evaluate(obs, q[:, 3])["kge"],          # median col
           "crps": crps_ensemble(obs, q),
           "cov90": float(np.mean((obs >= lo) & (obs <= hi))),
           "width90": float(np.mean(hi - lo)),            # mean 90% PI width (sharpness/MPIW)
           "winkler90": U.winkler_interval_score(obs, lo, hi, alpha=0.1),
           "alpha_reliability": U.alpha_reliability(obs, q, LEVELS),
           "twcrps_q90": U.twcrps(obs, q, q_level=0.9) if "q_level" in U.twcrps.__code__.co_varnames
           else U.twcrps(obs, q, 0.9)}
    return row


def main():
    cfg = load_config()
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    from sbc.models import load_all
    load_all()
    df = prepare(assemble("decadal"), "decadal", cfg).reset_index(drop=True)
    tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
    train, test = df[tr], df[te].reset_index(drop=True)
    pur = df[df["domain"] == "transfer"].reset_index(drop=True)

    from sbc.models.base import get_model
    factories = {
        "cmal": lambda: get_model("cmal")(seq_len=12, hidden=64, epochs=60),
        "regimeprobnet": lambda: get_model("regimeprobnet")(seq_len=12, hidden=64, epochs=60),
        "qrf": lambda: get_model("qrf")(),
    }
    rows = []
    for name, fac in factories.items():
        try:
            m = fac().fit(train)
            for split, part in (("temporal", test), ("pur", pur)):
                if len(part) < 20:
                    continue
                q = m.predict_discharge_quantiles(part, tuple(LEVELS))
                rows.append(_uq_row(name, split, part[OBS_COL].to_numpy(float), np.asarray(q)))
                log.info("%-14s %-9s done", name, split)
        except Exception as exc:
            log.warning("model %s failed: %s", name, exc)
    out = pd.DataFrame(rows)
    save_table(out, PATHS.tables / "cmal_uq_comparison_real_decadal.parquet", csv_mirror=True)
    log.info("CMAL vs Gaussian-flagship vs QRF — full UQ suite:\n%s", out.to_string(index=False))
    print("CMAL EVAL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
