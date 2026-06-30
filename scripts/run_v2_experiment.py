"""Integration driver for the *v2* real-data bias-correction experiment.

This script is the single entry point the maintainer runs on the **real decadal**
data once the sibling agents have landed their modules.  It evaluates the new
state-of-the-art baselines and the fixed ensembles against the flagship on the
two leakage-safe protocols that matter for an operational corrector -- the
per-gauge **temporal** holdout and **prediction-in-ungauged-regions** (PUR) -- and
writes the headline skill table, the per-gauge table and a summary figure under
``results/``.  The expensive leave-one-basin-out (LOBO) matrix is deliberately
*not* run here.

Model set
---------
* ``glofas_lstm`` -- :class:`sbc.models.sota_baselines.GloFASLSTMCorrector`
* ``donor``       -- :class:`sbc.models.sota_baselines.DonorRegionalizationCorrector`
* ``qrf``         -- :class:`sbc.models.probabilistic_baselines.QRFCorrector`
* ``stacked``     -- :class:`sbc.models.ensemble.StackedEnsemble` (lgbm+catboost+probnet, nnls)
* ``deepens``     -- :class:`sbc.models.robust.DeepEnsembleCorrector` (probnet members, full ``epochs=50``)
* ``regimeprobnet`` -- :class:`sbc.models.regime_prob_net.RegimeProbNet` (reference flagship)

The ``--teleconnections`` flag inserts
:func:`sbc.data.teleconnections.add_teleconnections` *before* feature engineering;
running the script with and without the flag quantifies the teleconnection skill
gain (outputs are tagged ``..._teleconn`` so both runs coexist).

Run
---
::

    # full publication run (after the sibling agents finish)
    PYTHONPATH=src python scripts/run_v2_experiment.py
    PYTHONPATH=src python scripts/run_v2_experiment.py --teleconnections
    PYTHONPATH=src python scripts/run_v2_experiment.py --quick
    PYTHONPATH=src python scripts/run_v2_experiment.py --models stacked,deepens --epochs 80

    # ~1 minute synthetic wiring check (no real data, no sibling modules needed)
    PYTHONPATH=src python scripts/run_v2_experiment.py --dry-run

Artefacts (``results/tables`` and ``results/figures``)::

    summary_v2_real_decadal.{parquet,csv}      # median skill per (model, split)
    per_gauge_v2_real_decadal.{parquet,csv}    # raw per-gauge metrics
    summary_v2_real_decadal.png                # KGE' bar chart
"""
from __future__ import annotations

import argparse
import inspect
import time
from typing import Callable

import pandas as pd

from sbc.config import PATHS
from sbc.experiment import load_config, prepare
from sbc.utils import get_logger, save_json, save_table, seed_everything
from sbc.validation import cv

log = get_logger("run_v2")
T0 = time.time()

#: full v2 model roster (order defines table / figure ordering)
V2_MODELS: tuple[str, ...] = (
    "glofas_lstm", "donor", "qrf", "stacked", "deepens", "regimeprobnet",
)


def _stamp(msg: str) -> None:
    log.info("[%6.0fs] %s", time.time() - T0, msg)


# --------------------------------------------------------------------------- #
#  Model factories                                                            #
# --------------------------------------------------------------------------- #
def _supported(cls: type, **kwargs) -> dict:
    """Filter ``kwargs`` to those accepted by ``cls.__init__``.

    The sibling baseline modules are authored in parallel, so their exact
    constructor signatures are not pinned here.  Forwarding only the accepted
    keywords (or everything, when ``__init__`` declares ``**kwargs``) lets this
    driver pass tuning knobs without breaking if a class does not expose one.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins w/o signature
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def build_v2_factories(scale: str, *, epochs: int, quick: bool,
                       which: list[str] | None = None) -> dict[str, Callable[[], object]]:
    """Zero-argument factories for the v2 model roster.

    Heavy model imports live *inside* each factory (house style) so the driver
    imports cheaply and a missing sibling module only disables its own model
    (``cv.compare`` catches the import error and logs a warning).

    Parameters
    ----------
    scale : str
        Temporal scale (``"decadal"``).
    epochs : int
        Training epochs for the deep correctors / ensemble members (the deep
        ensemble runs at this full budget; default 50).
    quick : bool
        Shrink the deep models (fewer epochs / members / hidden units) for a
        fast real-data smoke test.
    which : list of str, optional
        Subset of :data:`V2_MODELS` to keep (preserving roster order).

    Returns
    -------
    dict
        Mapping ``name -> factory`` ready for :func:`sbc.validation.cv.compare`.
    """
    seq = 4 if quick else (12 if scale == "decadal" else 90)
    hid = 32 if quick else 64
    ep = 8 if quick else int(epochs)
    n_members = 2 if quick else 5

    def _glofas_lstm():
        from sbc.models.sota_baselines import GloFASLSTMCorrector

        return GloFASLSTMCorrector(**_supported(
            GloFASLSTMCorrector, seq_len=seq, seq_length=seq, hidden=hid,
            hidden_size=hid, epochs=ep, max_epochs=ep, seed=0))

    def _donor():
        from sbc.models.sota_baselines import DonorRegionalizationCorrector

        return DonorRegionalizationCorrector(**_supported(
            DonorRegionalizationCorrector, seed=0))

    def _qrf():
        from sbc.models.probabilistic_baselines import QRFCorrector

        return QRFCorrector(**_supported(QRFCorrector, seed=0))

    def _stacked():
        from sbc.models.boosting import CatBoostCorrector, LightGBMCorrector
        from sbc.models.ensemble import StackedEnsemble
        from sbc.models.regime_prob_net import RegimeProbNet

        return StackedEnsemble(
            [LightGBMCorrector(n_optuna_trials=0),
             CatBoostCorrector(n_optuna_trials=0),
             RegimeProbNet(seq_len=seq, hidden=hid, epochs=ep)],
            meta="nnls")

    def _deepens():
        from sbc.models.robust import DeepEnsembleCorrector

        return DeepEnsembleCorrector(
            base="probnet", n_members=n_members,
            member_kwargs=dict(epochs=ep, hidden=hid, seq_len=seq))

    def _regimeprobnet():
        from sbc.models.regime_prob_net import RegimeProbNet

        return RegimeProbNet(seq_len=seq, hidden=hid, epochs=ep)

    facs: dict[str, Callable[[], object]] = {
        "glofas_lstm": _glofas_lstm,
        "donor": _donor,
        "qrf": _qrf,
        "stacked": _stacked,
        "deepens": _deepens,
        "regimeprobnet": _regimeprobnet,
    }
    if which is not None:
        unknown = [m for m in which if m not in facs]
        if unknown:
            log.warning("ignoring unknown model name(s): %s (known: %s)",
                        unknown, list(facs))
        facs = {k: facs[k] for k in V2_MODELS if k in which}
    return facs


def _dryrun_factories() -> dict[str, Callable[[], object]]:
    """Two fast, always-available correctors for the synthetic wiring check.

    Uses only modules that exist independently of the sibling agents so the
    dry-run proves the generate -> prepare -> compare -> save pipeline in ~1 min.
    """
    def _scaling():
        from sbc.models.quantile_mapping import LinearScalingCorrector

        return LinearScalingCorrector()

    def _lgbm():
        from sbc.models.boosting import LightGBMCorrector

        return LightGBMCorrector(n_optuna_trials=0)

    return {"scaling": _scaling, "lgbm": _lgbm}


# --------------------------------------------------------------------------- #
#  Data                                                                        #
# --------------------------------------------------------------------------- #
def _maybe_add_teleconnections(df: pd.DataFrame, enabled: bool,
                               allow_missing: bool) -> pd.DataFrame:
    """Insert climate-teleconnection predictors before feature engineering.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw modelling table (pre feature-engineering).
    enabled : bool
        Whether the ``--teleconnections`` flag was given.
    allow_missing : bool
        When ``True`` (dry-run) a missing sibling module is warned and skipped;
        when ``False`` (real run) the import error propagates so the maintainer
        notices the contribution was silently dropped.
    """
    if not enabled:
        return df
    try:
        from sbc.data.teleconnections import add_teleconnections
    except Exception as exc:
        if allow_missing:
            log.warning("teleconnections module unavailable (%s); continuing "
                        "without it (dry-run)", exc)
            return df
        raise
    try:
        out = add_teleconnections(df)
    except TypeError:  # signature may also require the temporal scale
        out = add_teleconnections(df, str(df["scale"].iloc[0]))
    log.info("teleconnections added: +%d column(s)", out.shape[1] - df.shape[1])
    return out


def load_real(scale: str, cfg: dict, teleconnections: bool) -> pd.DataFrame:
    """Assemble the real table, add teleconnections (optional), engineer features."""
    from sbc.data.assemble import assemble

    raw = assemble(scale)
    raw = _maybe_add_teleconnections(raw, teleconnections, allow_missing=False)
    return prepare(raw, scale, cfg).reset_index(drop=True)


def load_synthetic(scale: str, cfg: dict, teleconnections: bool) -> pd.DataFrame:
    """Small synthetic table standing in for the real one in the dry-run."""
    from sbc.synthetic import generate

    raw = generate(scale=scale, years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    raw = _maybe_add_teleconnections(raw, teleconnections, allow_missing=True)
    return prepare(raw, scale, cfg).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Evaluation + artefacts                                                      #
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame, factories: dict, tag: str, test_frac: float
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run temporal+PUR for every factory and persist the tables and figure."""
    _stamp(f"evaluating {list(factories)} on temporal+PUR (tag={tag})")
    results = cv.compare(factories, df, temporal=True, lobo=False, pur=True,
                         test_frac=test_frac)
    summary = cv.summarise(results)

    save_table(results, PATHS.tables / f"per_gauge_{tag}.parquet", csv_mirror=True)
    save_table(summary, PATHS.tables / f"summary_{tag}.parquet", csv_mirror=True)
    save_json({"tag": tag, "models": list(factories),
               "summary": summary.to_dict("records")},
              PATHS.tables / f"report_{tag}.json")
    try:
        _plot_summary(summary, tag)
    except Exception as exc:  # pragma: no cover - a plotting hiccup must not fail the run
        log.debug("summary figure skipped: %s", exc)

    if not summary.empty:
        log.info("\n%s", summary.to_string(index=False))
    return results, summary


def _plot_summary(summary: pd.DataFrame, tag: str) -> None:
    """Grouped horizontal bar chart of median KGE' per model and split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if summary.empty:
        return
    piv = summary.pivot(index="model", columns="split", values="kge")
    raw = summary.pivot(index="model", columns="split", values="kge_raw")
    splits = list(piv.columns)
    models = list(piv.index)
    y = np.arange(len(models))
    h = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 2))
    for i, sp in enumerate(splits):
        ax.barh(y + i * h, piv[sp].to_numpy(), height=h, label=f"corrected [{sp}]")
        rraw = float(np.nanmedian(raw[sp].to_numpy()))
        ax.axvline(rraw, ls="--", lw=0.8, color=f"C{i}",
                   label=f"raw GloFAS [{sp}] = {rraw:.2f}")
    ax.set_yticks(y + h * (len(splits) - 1) / 2)
    ax.set_yticklabels(models)
    ax.set_xlabel("median KGE'  (higher is better)")
    ax.set_title(f"v2 bias-correction skill ({tag})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", lw=0.3, alpha=0.6)
    fig.tight_layout()
    fig.savefig(PATHS.figures / f"summary_{tag}.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run(*, scale: str = "decadal", dry_run: bool = False, quick: bool = False,
        models: list[str] | None = None, teleconnections: bool = False,
        epochs: int = 50, config: str | None = None) -> dict:
    """Load the data, evaluate the model set and write all artefacts."""
    cfg = load_config(config)
    seed_everything(cfg.get("seed", 1234))
    PATHS.ensure()
    test_frac = cfg["validation"]["temporal_test_frac"]

    if dry_run:
        _stamp("DRY-RUN: synthetic generate -> prepare -> compare (2 models)")
        df = load_synthetic(scale, cfg, teleconnections)
        _stamp(f"prepared synthetic table {df.shape}, gauges={df['code'].nunique()}, "
               f"domains={sorted(df['domain'].unique())}")
        tag = f"v2_dryrun_{scale}"
        factories = _dryrun_factories()
    else:
        _stamp("loading + preparing REAL decadal data")
        df = load_real(scale, cfg, teleconnections)
        _stamp(f"prepared real table {df.shape}, gauges={df['code'].nunique()}, "
               f"basins={df['basin'].nunique()}")
        tag = f"v2_real_{scale}" + ("_teleconn" if teleconnections else "")
        factories = build_v2_factories(scale, epochs=epochs, quick=quick, which=models)
        if quick:
            tag += "_quick"

    if not factories:
        raise SystemExit("no models selected to evaluate")

    results, summary = evaluate(df, factories, tag, test_frac)
    _stamp("DONE")
    return {"tag": tag, "results": results, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v2 real-decadal integration experiment "
                    "(new baselines + fixed ensembles; temporal+PUR).")
    ap.add_argument("--scale", default="decadal", choices=["decadal", "daily"])
    ap.add_argument("--dry-run", action="store_true",
                    help="~1 min synthetic wiring check (no real data / siblings)")
    ap.add_argument("--quick", action="store_true",
                    help="fast real-data smoke (small/short deep models)")
    ap.add_argument("--models", default=None,
                    help=f"comma-separated subset of {list(V2_MODELS)}")
    ap.add_argument("--teleconnections", action="store_true",
                    help="add climate teleconnection predictors before features")
    ap.add_argument("--epochs", type=int, default=50,
                    help="deep-model / deep-ensemble training epochs (default 50)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    run(scale=a.scale, dry_run=a.dry_run, quick=a.quick, models=models,
        teleconnections=a.teleconnections, epochs=a.epochs, config=a.config)


if __name__ == "__main__":
    main()
