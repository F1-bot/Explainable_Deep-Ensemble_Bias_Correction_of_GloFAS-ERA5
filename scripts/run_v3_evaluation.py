"""v3 evaluation: do the new cutting-edge techniques actually help on real data?

Evaluates the differentiable-snow corrector and the generative head vs the
flagship/best models, measures the REAL monotonicity-violation rate (diffsnow vs
soft vs hard), and the group-level SHAP stability. Feeds the verification round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import load_config, prepare
from sbc.utils import get_logger, save_table, seed_everything
from sbc.validation import cv
from sbc.validation.splits import temporal_split

log = get_logger("v3")


def _fac(name, **kw):
    from sbc.models.base import get_model
    cls = get_model(name)

    def f():
        try:
            return cls(**kw)
        except TypeError:
            try:
                return cls(epochs=kw.get("epochs", 25))
            except TypeError:
                return cls()
    return f


def main():
    cfg = load_config()
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    from sbc.models import load_all
    load_all()
    df = prepare(assemble("decadal"), "decadal", cfg).reset_index(drop=True)

    # ---- (1) do the new techniques help? -----------------------------------
    try:
        kw = dict(seq_len=12, hidden=64, epochs=30)
        facs = {
            "diffsnow": _fac("diffsnow", **kw),
            "gen_resid": _fac("gen_resid", **kw),
            "regimeprobnet": _fac("regimeprobnet", **kw),
            "stacked": _fac("stacked"),
        }
        res = cv.compare(facs, df, temporal=True, lobo=False, pur=True)
        summ = cv.summarise(res)
        save_table(summ, PATHS.tables / "summary_v3_real_decadal.parquet", csv_mirror=True)
        log.info("v3 new-technique evaluation:\n%s", summ.to_string(index=False))
    except Exception as exc:
        log.warning("v3 eval failed: %s", exc)

    # ---- (2) REAL monotonicity-violation rate (the physics earned?) ---------
    try:
        from sbc.validation.reporting import monotonicity_violation_rate
        from sbc.models.base import get_model
        tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
        rows = []
        for name in ("diffsnow", "regimeprobnet", "probnet_hardmono"):
            try:
                m = _fac(name, seq_len=12, hidden=64, epochs=30)().fit(df[tr])
                vr = monotonicity_violation_rate(m, df[te].reset_index(drop=True),
                                                 features=("swe", "smlt"))
                rows.append({"model": name, "violation_rate": vr})
                log.info("%-16s real melt-monotonicity violation rate = %s", name, vr)
            except Exception as exc:
                log.warning("violation %s: %s", name, exc)
        if rows:
            save_table(pd.DataFrame(rows),
                       PATHS.tables / "monotonicity_violation_real_decadal.parquet", csv_mirror=True)
    except Exception as exc:
        log.warning("monotonicity section failed: %s", exc)

    # ---- (3) group-level SHAP stability (the XAI earned?) -------------------
    try:
        from sbc.explain.group_causal_shap import group_stability
        from sbc.models.boosting import LightGBMCorrector
        gs = group_stability(lambda: LightGBMCorrector(n_optuna_trials=0), df, n_seeds=5)
        log.info("group SHAP stability (lgbm): %s", gs if not isinstance(gs, tuple) else gs[1])
        # persist whatever tidy frame it returns
        obj = gs[0] if isinstance(gs, tuple) else gs
        if isinstance(obj, pd.DataFrame):
            save_table(obj, PATHS.tables / "group_shap_stability_real_decadal.parquet", csv_mirror=True)
    except Exception as exc:
        log.warning("group-stability section failed: %s", exc)

    print("V3 EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
