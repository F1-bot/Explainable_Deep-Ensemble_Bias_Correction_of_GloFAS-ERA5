"""Round-2 real-data analyses: hard-vs-soft constraint ablation + decision skill."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.data.assemble import assemble
from sbc.experiment import load_config, prepare
from sbc.schemas import OBS_COL
from sbc.utils import get_logger, save_table, seed_everything
from sbc.validation import cv
from sbc.validation.splits import temporal_split

log = get_logger("r2")


def _factory(name, **kw):
    from sbc.models.base import get_model
    cls = get_model(name)

    def f():
        try:
            return cls(**kw)
        except TypeError:
            return cls()
    return f


def main():
    cfg = load_config()
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    from sbc.models import load_all
    load_all()  # register every model (regimeprobnet, probnet_hardmono, qrf, ...)
    df = prepare(assemble("decadal"), "decadal", cfg).reset_index(drop=True)

    # ---- (1) soft vs hard-monotonic vs asymmetric-Laplace flagship ---------
    try:
        kw = dict(seq_len=12, hidden=64, epochs=30)
        facs = {"probnet_soft": _factory("regimeprobnet", **kw),
                "probnet_hardmono": _factory("probnet_hardmono", **kw),
                "probnet_alaplace": _factory("probnet_alaplace", **kw)}
        res = cv.compare(facs, df, temporal=True, lobo=False, pur=True)
        summ = cv.summarise(res)
        save_table(summ, PATHS.tables / "constraint_ablation_real_decadal.parquet",
                   csv_mirror=True)
        log.info("constraint ablation (soft vs hard vs asym-Laplace):\n%s",
                 summ.to_string(index=False))
    except Exception as exc:
        log.warning("constraint ablation failed: %s", exc)

    # optional: monotonicity-violation rate probe (soft vs hard)
    try:
        import sbc.models.constraint_variants as CVm
        probe = next((getattr(CVm, n) for n in dir(CVm)
                      if "violation" in n.lower() and callable(getattr(CVm, n))), None)
        if probe is not None:
            log.info("violation-rate probe available: %s", probe.__name__)
    except Exception as exc:
        log.debug("violation probe: %s", exc)

    # ---- (2) decision-relevant + extreme-flow skill (QRF probabilistic) ----
    try:
        from sbc.models.probabilistic_baselines import QRFCorrector
        from sbc.validation import decision_skill as DS

        tr, te = temporal_split(df, cfg["validation"]["temporal_test_frac"])
        model = QRFCorrector().fit(df[tr])
        test = df[te].reset_index(drop=True)
        levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        q = model.predict_discharge_quantiles(test, tuple(levels))
        tbl = DS.exceedance_skill_table(test[OBS_COL].values, q, levels)
        save_table(tbl, PATHS.tables / "decision_skill_real_decadal.parquet", csv_mirror=True)
        log.info("decision/extreme-flow skill (QRF):\n%s", tbl.to_string(index=False))
        pk = DS.peak_flow_metrics(test[OBS_COL].values, model.predict(test),
                                  test["date"].values, codes=test["code"].values)
        log.info("peak-flow metrics: %s", pk)
    except Exception as exc:
        log.warning("decision skill failed: %s", exc)

    print("ROUND2 ANALYSES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
