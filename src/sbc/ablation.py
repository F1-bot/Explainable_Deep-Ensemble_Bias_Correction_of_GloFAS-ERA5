"""Controlled ablation harness: isolate the contribution of each design choice.

A bias-correction paper claims that *specific* ingredients carry the gains.  This
module turns those claims into falsifiable evidence by re-running the leakage-safe
validation matrix (:mod:`sbc.validation.cv`) with one ingredient toggled at a time
and reporting the **change in median KGE'** it produces.  Five studies are wired,
mirroring the contributions advertised in the abstract:

``snow_features``
    snow forcings and their engineered descendants ON vs OFF (columns whose name
    carries a snow token -- ``swe``/``smlt``/``scf``/``snow``/``t2m``/``sf`` and
    the snow-derived ``f_`` family such as ``f_pdd_cum`` / ``f_melt_season`` /
    ``f_freeze_thaw``).
``static_attrs``
    static catchment attributes (the per-gauge-constant columns discovered by
    :func:`sbc.schemas.static_feature_columns`) ON vs OFF.
``residual_target``
    the framework's log-residual target ``log(q_obs+EPS) - log(q_glofas+EPS)`` vs
    predicting the discharge *directly* (target ``log(q_obs+EPS)``, reconstructed
    without leaning on GloFAS).  GloFAS-derived features are retained in *both*
    arms, so only the target parameterisation changes -- a clean controlled test.
``regime_gating`` (flagship)
    the regime-gated mixture-of-experts head ON (``K=5`` gated, regime-supervised)
    vs OFF (collapsed to a single Gaussian, ``K=1, lambda_gate=0``).
``physics_penalty`` (flagship)
    the soft SWE/temperature monotonicity penalty ON (``lambda_phys>0``) vs OFF
    (``lambda_phys=0, physics=False``).

The harness is *model-agnostic*: studies (a)-(c) run through any zero-argument
``feature_factory`` (default LightGBM), and the flagship studies through any
``flagship_factory`` accepting ``RegimeProbNet`` keyword overrides.  Every run is
funnelled through :func:`sbc.validation.cv.run_matrix`, so the same splits,
metrics and back-transform used in the headline results are used here.

Sign convention
---------------
``delta_kge = KGE'(full) - KGE'(ablated)`` and ``delta_nse`` analogously; for the
strictly-proper CRPS (lower is better) ``delta_crps = CRPS(ablated) - CRPS(full)``.
In every case a **positive delta means the toggled-on component adds skill**.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .schemas import (
    OBS_COL,
    SIM_COL,
    TARGET_COL,
    feature_columns,
    make_target,
    static_feature_columns,
    validate,
)
from .utils import get_logger, seed_everything
from .validation import cv

log = get_logger(__name__)

ModelFactory = Callable[[], object]

#: substring tokens (case-insensitive) marking a feature column as snow-related.
SNOW_TOKENS: tuple[str, ...] = (
    "swe", "smlt", "scf", "snow", "t2m", "sf",
    "pdd", "melt", "freeze", "thaw", "frost", "rain_on_snow",
)

#: flagship (RegimeProbNet) base keyword defaults; overridable per call.
_FLAGSHIP_DEFAULTS: dict = dict(
    K=5, hidden=32, expert_hidden=24, gate_hidden=24, epochs=30,
    batch_size=256, patience=8, lambda_gate=0.5, lambda_phys=0.1, loss="crps",
)
#: flagship look-back window per temporal scale.
_FLAGSHIP_SEQ: dict[str, int] = {"decadal": 6, "daily": 30}

#: metric columns lifted (as per-gauge medians) from a :func:`cv.run_matrix` table.
_METRIC_COLS: tuple[str, ...] = (
    "kge", "kge_raw", "nse", "nse_raw", "pbias", "pbias_raw",
    "crps", "peak_timing_err",
)

#: ordered output columns of :func:`run_ablations`.
_OUT_COLS: tuple[str, ...] = (
    "study", "contribution", "split", "model", "full", "ablated", "n_gauges",
    "kge_raw", "kge_full", "kge_ablated", "delta_kge",
    "nse_full", "nse_ablated", "delta_nse",
    "pbias_full", "pbias_ablated",
    "crps_full", "crps_ablated", "delta_crps",
)

__all__ = [
    "SNOW_TOKENS",
    "snow_feature_columns",
    "direct_target_frame",
    "run_ablations",
    "delta_table",
]


# --------------------------------------------------------------------------- #
#  Feature-group selection and frame surgery                                  #
# --------------------------------------------------------------------------- #
def snow_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model-input columns judged snow-related by :data:`SNOW_TOKENS`.

    Only numeric *feature* columns are considered (id / target / obs / sim are
    never returned), so the result is always safe to drop.

    Parameters
    ----------
    df : pandas.DataFrame
        A modelling table (see :mod:`sbc.schemas`).

    Returns
    -------
    list of str
        Snow-related feature column names, in table order.
    """
    out = []
    for c in feature_columns(df):
        lc = c.lower()
        if any(tok in lc for tok in SNOW_TOKENS):
            out.append(c)
    return out


def _drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` without ``cols`` (missing names ignored)."""
    present = [c for c in cols if c in df.columns]
    return df.drop(columns=present).copy() if present else df.copy()


def direct_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Recast a table for the *direct-discharge* target ablation.

    The raw GloFAS baseline is zeroed (``q_glofas -> 0``) and the target is
    rebuilt as ``log(q_obs+EPS) - log(EPS)``.  Because
    :func:`sbc.schemas.back_transform` reconstructs
    ``exp(log(q_glofas+EPS) + residual) - EPS``, a model fitted on this frame
    must predict the *whole* log-discharge and is reconstructed without GloFAS --
    exactly the "predict discharge directly" baseline.  All engineered features
    (including the GloFAS-memory family) are left untouched, so the comparison
    against the residual target isolates the target parameterisation alone.

    Parameters
    ----------
    df : pandas.DataFrame
        A modelling table carrying ``q_obs`` and ``q_glofas``.

    Returns
    -------
    pandas.DataFrame
        A copy with ``q_glofas`` set to zero and ``log_residual`` rebuilt.
    """
    out = df.copy()
    obs = out[OBS_COL].to_numpy(float)
    out[SIM_COL] = 0.0
    out[TARGET_COL] = make_target(obs, np.zeros_like(obs))
    return out


# --------------------------------------------------------------------------- #
#  Run / aggregate helpers                                                     #
# --------------------------------------------------------------------------- #
def _split_flags(splits: tuple[str, ...]) -> dict[str, bool]:
    """Map a requested-split tuple to :func:`cv.run_matrix` boolean kwargs."""
    s = {str(x).lower() for x in splits}
    return {"temporal": "temporal" in s, "lobo": "lobo" in s, "pur": "pur" in s}


def _safe_run(make_model: ModelFactory, df: pd.DataFrame, label: str,
              splits: tuple[str, ...], test_frac: float) -> pd.DataFrame:
    """Run one model through the requested splits; never raise (log and skip)."""
    flags = _split_flags(splits)
    try:
        return cv.run_matrix(make_model, df, label, test_frac=test_frac, **flags)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("ablation run %r failed: %s", label, exc)
        return pd.DataFrame()


def _median(pg: pd.DataFrame, split: str) -> dict[str, float]:
    """Per-gauge median of the metric bundle for one split of a run-matrix table."""
    if pg is None or pg.empty or "split" not in pg.columns:
        return {}
    sub = pg[pg["split"] == split]
    if sub.empty:
        return {}
    out: dict[str, float] = {
        k: float(sub[k].median(skipna=True)) for k in _METRIC_COLS if k in sub.columns
    }
    out["n_gauges"] = int(sub["code"].nunique())
    return out


def _comparison_rows(study: str, contribution: str, full_pg: pd.DataFrame,
                     abl_pg: pd.DataFrame, full_label: str, abl_label: str,
                     model: str, splits: tuple[str, ...]) -> list[dict]:
    """Assemble one tidy comparison row per split for a single ablation study."""
    rows: list[dict] = []
    nan = float("nan")
    for split in splits:
        f = _median(full_pg, split)
        a = _median(abl_pg, split)
        if not f and not a:
            continue
        kge_f, kge_a = f.get("kge", nan), a.get("kge", nan)
        nse_f, nse_a = f.get("nse", nan), a.get("nse", nan)
        crps_f, crps_a = f.get("crps", nan), a.get("crps", nan)
        rows.append({
            "study": study,
            "contribution": contribution,
            "split": split,
            "model": model,
            "full": full_label,
            "ablated": abl_label,
            "n_gauges": f.get("n_gauges", a.get("n_gauges", 0)),
            "kge_raw": f.get("kge_raw", a.get("kge_raw", nan)),
            "kge_full": kge_f,
            "kge_ablated": kge_a,
            "delta_kge": kge_f - kge_a,
            "nse_full": nse_f,
            "nse_ablated": nse_a,
            "delta_nse": nse_f - nse_a,
            "pbias_full": f.get("pbias", nan),
            "pbias_ablated": a.get("pbias", nan),
            "crps_full": crps_f,
            "crps_ablated": crps_a,
            "delta_crps": crps_a - crps_f,
        })
    return rows


# --------------------------------------------------------------------------- #
#  Public entry point                                                          #
# --------------------------------------------------------------------------- #
def run_ablations(df: pd.DataFrame, scale: str = "decadal",
                  splits: tuple[str, ...] = ("temporal",), seed: int = 0, *,
                  feature_factory: ModelFactory | None = None,
                  flagship_factory: Callable[..., object] | None = None,
                  flagship_kwargs: dict | None = None,
                  include_flagship: bool = True,
                  test_frac: float = 0.3) -> pd.DataFrame:
    """Run the controlled ablation suite and return a tidy delta-skill table.

    Parameters
    ----------
    df : pandas.DataFrame
        A *prepared* modelling table -- engineered features
        (:func:`sbc.features.engineering.build_features`) and, for the flagship
        studies, a ``regime`` label (:func:`sbc.features.regimes.classify_regimes`)
        already attached.  The log-residual target is ensured internally.
    scale : str, default "decadal"
        Temporal scale; selects the flagship look-back window.
    splits : tuple of str, default ``("temporal",)``
        Which leakage-safe protocols to score; any subset of
        ``"temporal"`` / ``"lobo"`` / ``"pur"``.
    seed : int, default 0
        Global reproducibility seed (wired into the default factories).
    feature_factory : callable, optional
        Zero-argument factory for the feature/target studies (a)-(c).  Defaults
        to a deterministic, untuned LightGBM corrector.
    flagship_factory : callable, optional
        Factory accepting ``RegimeProbNet`` keyword overrides for the flagship
        studies (d).  Defaults to :class:`~sbc.models.regime_prob_net.RegimeProbNet`
        built from :data:`_FLAGSHIP_DEFAULTS` updated by ``flagship_kwargs``.
    flagship_kwargs : dict, optional
        Overrides merged onto :data:`_FLAGSHIP_DEFAULTS` (e.g. ``epochs``,
        ``hidden``) -- use small values for smoke runs.
    include_flagship : bool, default True
        Skip the (heavy) flagship studies when ``False`` or when ``torch`` is
        unavailable.
    test_frac : float, default 0.3
        Forwarded to the temporal split.

    Returns
    -------
    pandas.DataFrame
        One row per (study, split) with the full vs ablated medians and the
        signed deltas (see the module docstring for the sign convention).  Empty
        if every run failed.
    """
    seed_everything(int(seed))
    df = validate(df).reset_index(drop=True)
    splits = tuple(splits)
    rng_seed = int(seed)

    if feature_factory is None:
        def feature_factory():  # noqa: D401 - tiny default factory
            from .models.boosting import LightGBMCorrector

            return LightGBMCorrector(n_optuna_trials=0, seed=rng_seed)

    model_name = str(getattr(feature_factory(), "name", "model"))

    # -- ablated frames (a)-(c) -------------------------------------------- #
    snow_cols = snow_feature_columns(df)
    static_cols = static_feature_columns(df)
    df_snow_off = _drop(df, snow_cols)
    df_static_off = _drop(df, static_cols)
    df_direct = direct_target_frame(df)
    log.info("ablation feature groups: %d snow, %d static (of %d features)",
             len(snow_cols), len(static_cols), len(feature_columns(df)))

    # one shared "full" residual run feeds studies (a), (b) and (c) ---------
    pg_full = _safe_run(feature_factory, df, "full", splits, test_frac)
    pg_snow = _safe_run(feature_factory, df_snow_off, "snow_off", splits, test_frac)
    pg_static = _safe_run(feature_factory, df_static_off, "static_off", splits, test_frac)
    pg_direct = _safe_run(feature_factory, df_direct, "direct", splits, test_frac)

    rows: list[dict] = []
    rows += _comparison_rows("snow_features", "snow forcings & derived features",
                             pg_full, pg_snow, "snow_on", "snow_off", model_name, splits)
    rows += _comparison_rows("static_attrs", "static catchment attributes",
                             pg_full, pg_static, "static_on", "static_off", model_name, splits)
    rows += _comparison_rows("residual_target", "log-residual vs direct discharge",
                             pg_full, pg_direct, "residual", "direct", model_name, splits)

    # -- flagship studies (d): regime gating and physics penalty ------------ #
    if include_flagship:
        try:
            base = {**_FLAGSHIP_DEFAULTS, "seq_len": _FLAGSHIP_SEQ.get(scale, 6)}
            if flagship_kwargs:
                base.update(flagship_kwargs)
            base["seed"] = rng_seed
            base["verbose"] = False

            if flagship_factory is None:
                from .models.regime_prob_net import RegimeProbNet

                def flagship_factory(**ov):  # noqa: D401 - tiny default factory
                    return RegimeProbNet(**{**base, **ov})

            pg_flag_full = _safe_run(lambda: flagship_factory(),
                                     df, "flagship_full", splits, test_frac)
            pg_flag_nogate = _safe_run(lambda: flagship_factory(K=1, lambda_gate=0.0),
                                       df, "flagship_nogate", splits, test_frac)
            pg_flag_nophys = _safe_run(lambda: flagship_factory(lambda_phys=0.0, physics=False),
                                       df, "flagship_nophys", splits, test_frac)

            rows += _comparison_rows("regime_gating", "regime-gated mixture-of-experts",
                                     pg_flag_full, pg_flag_nogate,
                                     "gating_on", "gating_off", "regimeprobnet", splits)
            rows += _comparison_rows("physics_penalty", "SWE/temperature monotonicity",
                                     pg_flag_full, pg_flag_nophys,
                                     "physics_on", "physics_off", "regimeprobnet", splits)
        except Exception as exc:  # pragma: no cover - torch/flagship optional
            log.warning("flagship ablations skipped: %s", exc)

    if not rows:
        return pd.DataFrame(columns=list(_OUT_COLS))
    out = pd.DataFrame(rows)
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].round(4)
    return out[list(_OUT_COLS)]


def delta_table(result: pd.DataFrame, metric: str = "delta_kge") -> pd.DataFrame:
    """Pivot a :func:`run_ablations` table to studies x splits of one delta metric.

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`run_ablations`.
    metric : str, default ``"delta_kge"``
        Column to display (e.g. ``"delta_kge"``, ``"delta_nse"``, ``"delta_crps"``).

    Returns
    -------
    pandas.DataFrame
        Rows indexed by ``study``, columns the requested splits.
    """
    if result.empty or metric not in result.columns:
        return pd.DataFrame()
    return (result.pivot_table(index="study", columns="split", values=metric, sort=False)
            .reindex(result["study"].drop_duplicates()))


# --------------------------------------------------------------------------- #
#  Scripts-style smoke entry point                                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .features.engineering import build_features
    from .features.regimes import classify_regimes
    from .synthetic import generate

    # small synthetic table -> features -> regimes (contract smoke recipe)
    table = generate(scale="decadal", years=8, n_basins=3,
                     gauges_per_basin=(2, 3), seed=0)
    table = build_features(table, scale="decadal")
    table = classify_regimes(table)
    print(f"[ablation] synthetic table: {len(table)} rows, "
          f"{table['code'].nunique()} gauges, {table['basin'].nunique()} basins, "
          f"{len(feature_columns(table))} features")

    res = run_ablations(
        table, scale="decadal", splits=("temporal",), seed=0,
        flagship_kwargs=dict(epochs=3, hidden=16, seq_len=4, expert_hidden=12,
                             gate_hidden=12, batch_size=256, patience=3),
    )

    view = ["study", "split", "model", "full", "ablated",
            "kge_full", "kge_ablated", "delta_kge", "crps_full", "crps_ablated"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Ablation delta-KGE' table (synthetic, temporal holdout) ===")
    print(res[view].to_string(index=False))
    print("\ndelta_kge = KGE'(full) - KGE'(ablated); "
          "positive => the toggled-on component adds skill.")
    print("\nDelta-KGE' pivot (study x split):")
    print(delta_table(res).round(4).to_string())
