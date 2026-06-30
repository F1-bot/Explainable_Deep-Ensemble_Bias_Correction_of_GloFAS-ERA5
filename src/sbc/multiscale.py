"""Multi-scale experiments: cross-resolution transfer and aggregation consistency.

The paper advertises a *multi-scale* bias-correction framework that operates on
both the Central-Asian **decadal** (10-day) discharge bulletins and the native
**daily** GloFAS-ERA5 stream.  A label alone is not a contribution; this module
turns "multi-scale" into two falsifiable, quantified experiments that probe
whether a single correction methodology genuinely spans the two temporal
resolutions:

``cross_scale_transfer``
    Train a corrector on *one* resolution and deploy it at the *other*, then ask
    how much of the **native-scale** skill survives the resolution change:

    * **daily -> decadal** — fit on daily rows, correct daily discharge, then
      *aggregate* the corrected series to the Central-Asian decades and score it
      against the decadal observations.  This is the operationally relevant
      direction when a daily model must feed a decadal water-accounting bulletin.
    * **decadal -> daily** — fit on the decadal bulletins, then *broadcast* the
      decadal log-residual correction onto every day of its decade and apply it
      to the daily GloFAS.  This is the "downscaling" direction: can a model
      learned on coarse data still sharpen the daily hydrograph?

    For each direction the cross-scale model is compared against a model trained
    *natively* at the evaluation resolution, and a **retention** ratio
    (transferred skill / native skill, on the mean-flow-benchmark skill-score
    scale) quantifies the price of the resolution change in KGE' and PBIAS.

``consistency_check``
    A multi-scale framework should be *internally coherent*: the corrected daily
    series, when aggregated to decades, ought to reproduce the decadal truth.
    This routine aggregates a daily-corrected series to the decadal scale and
    reports the **mean absolute decadal discrepancy** against the decadal
    observations (and, for context, the same discrepancy for raw GloFAS), i.e. a
    volume-conservation / scale-coherence diagnostic.

Both experiments are deliberately leakage-safe — every corrector is trained on
the *earliest* part of its record (via :func:`sbc.validation.splits.temporal_split`)
and scored on a strictly posterior window — and entirely model-agnostic: any
zero-argument :class:`~sbc.models.base.BaseCorrector` factory can be supplied,
defaulting to a deterministic, untuned LightGBM corrector for smoke runs.

Sign / scale conventions
------------------------
KGE' and NSE: higher is better (1.0 perfect).  PBIAS: closer to 0 is better.
``retention = kge_ss_transfer / kge_ss_native`` on the skill-score scale where
the mean-flow benchmark maps to 0 and a perfect forecast to 1; ``retention``
near 1.0 means the cross-scale model keeps essentially all of the native skill,
``< 1`` means degradation, ``> 1`` means the transfer *improves* on native
training, and it is ``NaN`` when the native model has no positive skill to
retain.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .schemas import OBS_COL, PRED_COL, SIM_COL, back_transform, validate
from .utils import get_logger, seed_everything
from .validation import metrics as M
from .validation.splits import temporal_split

log = get_logger(__name__)

ModelFactory = Callable[[], object]

GAUGE_COL = "code"
DATE_COL = "date"

#: ordered columns of the :func:`cross_scale_transfer` report.
_TRANSFER_COLS: tuple[str, ...] = (
    "direction", "eval_scale", "n_gauges", "n_periods",
    "kge_raw", "kge_native", "kge_transfer", "delta_kge", "retention",
    "nse_native", "nse_transfer",
    "pbias_raw", "pbias_native", "pbias_transfer",
)

__all__ = ["cross_scale_transfer", "consistency_check", "ModelFactory"]


# --------------------------------------------------------------------------- #
#  Small shared helpers                                                        #
# --------------------------------------------------------------------------- #
def _default_factory(seed: int) -> ModelFactory:
    """Return a zero-argument factory for a fast, deterministic corrector.

    Mirrors :mod:`sbc.ablation`: an untuned LightGBM corrector (no Optuna) so the
    multi-scale experiments have a sensible default without a heavy net.
    """
    def factory():  # noqa: D401 - tiny default factory
        from .models.boosting import LightGBMCorrector

        return LightGBMCorrector(n_optuna_trials=0, seed=int(seed))

    return factory


def _decade_date(dates: pd.Series) -> pd.Series:
    """Map daily timestamps to their Central-Asian decade representative date.

    Each calendar month is split into three decades — days 1-10, 11-20 and
    21-end — represented by day 5, 15 and 25 respectively, matching
    :func:`sbc.synthetic.decadal_aggregate`.

    Parameters
    ----------
    dates : pandas.Series
        Datetime-like series of daily timestamps.

    Returns
    -------
    pandas.Series
        The decade representative date for every input timestamp, index-aligned.
    """
    dt = pd.to_datetime(dates)
    d = dt.dt.day
    dec = np.where(d <= 10, 5, np.where(d <= 20, 15, 25))
    rep = pd.to_datetime(dict(year=dt.dt.year, month=dt.dt.month, day=dec))
    return pd.Series(np.asarray(rep), index=dt.index)


def _aggregate_to_decadal(df: pd.DataFrame, value_cols: list[str],
                          keep: tuple[str, ...] = ("basin", "domain")) -> pd.DataFrame:
    """Mean-aggregate selected discharge columns to the decadal scale.

    Unlike :func:`sbc.synthetic.decadal_aggregate` this collapses *only* the
    requested numeric discharge columns (plus a few id columns carried with
    ``first``), so it is safe on a feature-rich, regime-labelled modelling table
    where a blanket ``mean`` over engineered / categorical columns would fail.

    Parameters
    ----------
    df : pandas.DataFrame
        A daily modelling table with ``code`` and ``date`` columns.
    value_cols : list of str
        Discharge columns to mean-aggregate within each (gauge, decade).
    keep : tuple of str, default ``("basin", "domain")``
        Per-gauge-constant id columns carried through with ``first``.

    Returns
    -------
    pandas.DataFrame
        One row per (``code``, decade ``date``) with the aggregated columns.
    """
    work = df.copy()
    work[DATE_COL] = _decade_date(work[DATE_COL])
    agg: dict[str, str] = {c: "mean" for c in value_cols if c in work.columns}
    for c in keep:
        if c in work.columns:
            agg[c] = "first"
    return work.groupby([GAUGE_COL, DATE_COL], as_index=False).agg(agg)


def _median_skill(frame: pd.DataFrame, obs_col: str, pred_col: str) -> dict[str, float]:
    """Median per-gauge skill bundle for one (obs, pred) column pair.

    Uses :func:`sbc.validation.metrics.evaluate_by_group` (one row per gauge),
    then reduces to the across-gauge median — the house convention used by
    :mod:`sbc.validation.cv` and :mod:`sbc.ablation`.
    """
    pg = M.evaluate_by_group(frame, obs_col, pred_col, group=GAUGE_COL, date_col=DATE_COL)
    keys = ("kge", "kge_ss", "nse", "pbias", "rmse")
    out = {k: float(pg[k].median(skipna=True)) if k in pg else float("nan") for k in keys}
    return out


def _retention(ss_transfer: float, ss_native: float) -> float:
    """Skill-score retention ratio, guarded against non-positive native skill."""
    if not np.isfinite(ss_native) or ss_native <= 0.0:
        return float("nan")
    return float(ss_transfer / ss_native)


def _row(direction: str, eval_scale: str, frame: pd.DataFrame,
         transfer: dict[str, float], native: dict[str, float],
         raw: dict[str, float]) -> dict:
    """Assemble one tidy report row for a transfer direction."""
    return {
        "direction": direction,
        "eval_scale": eval_scale,
        "n_gauges": int(frame[GAUGE_COL].nunique()),
        "n_periods": int(len(frame)),
        "kge_raw": raw["kge"],
        "kge_native": native["kge"],
        "kge_transfer": transfer["kge"],
        "delta_kge": transfer["kge"] - native["kge"],
        "retention": _retention(transfer["kge_ss"], native["kge_ss"]),
        "nse_native": native["nse"],
        "nse_transfer": transfer["nse"],
        "pbias_raw": raw["pbias"],
        "pbias_native": native["pbias"],
        "pbias_transfer": transfer["pbias"],
    }


def _nan_row(direction: str, eval_scale: str) -> dict:
    """Placeholder row when a direction has no overlapping evaluation periods."""
    nan = float("nan")
    row = {c: nan for c in _TRANSFER_COLS}
    row["direction"] = direction
    row["eval_scale"] = eval_scale
    row["n_gauges"] = 0
    row["n_periods"] = 0
    return row


def _split(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-gauge temporal holdout returning (train, test) frames."""
    tr, te = temporal_split(df, test_frac=test_frac)
    return df[tr].copy(), df[te].copy()


# --------------------------------------------------------------------------- #
#  Direction 1: daily-trained -> aggregated to decadal                        #
# --------------------------------------------------------------------------- #
def _daily_to_decadal(daily: pd.DataFrame, decadal: pd.DataFrame,
                      factory: ModelFactory, test_frac: float) -> dict:
    """Train daily, aggregate corrected daily to decades; compare to native decadal."""
    daily_train, daily_test = _split(daily, test_frac)
    dec_train, dec_test = _split(decadal, test_frac)

    # transfer: daily model -> corrected daily discharge -> decade means
    m_daily = factory()
    m_daily.fit(daily_train)
    daily_test = daily_test.assign(**{PRED_COL: m_daily.predict(daily_test)})
    dec_eval = _aggregate_to_decadal(daily_test, [OBS_COL, SIM_COL, PRED_COL])
    dec_eval = dec_eval.rename(columns={PRED_COL: "q_pred_transfer",
                                        SIM_COL: "q_glofas_dec"})

    # native: decadal model scored on the same evaluation decades
    m_dec = factory()
    m_dec.fit(dec_train)
    nat = dec_test[[GAUGE_COL, DATE_COL]].copy()
    nat["q_pred_native"] = m_dec.predict(dec_test)

    merged = dec_eval.merge(nat, on=[GAUGE_COL, DATE_COL], how="inner")
    if merged.empty:
        log.warning("daily->decadal transfer: no overlapping decades to score")
        return _nan_row("daily->decadal", "decadal")

    transfer = _median_skill(merged, OBS_COL, "q_pred_transfer")
    native = _median_skill(merged, OBS_COL, "q_pred_native")
    raw = _median_skill(merged, OBS_COL, "q_glofas_dec")
    return _row("daily->decadal", "decadal", merged, transfer, native, raw)


# --------------------------------------------------------------------------- #
#  Direction 2: decadal-trained -> broadcast to daily                         #
# --------------------------------------------------------------------------- #
def _decadal_to_daily(daily: pd.DataFrame, decadal: pd.DataFrame,
                      factory: ModelFactory, test_frac: float) -> dict:
    """Train decadal, broadcast its log-residual onto daily; compare to native daily."""
    daily_train, daily_test = _split(daily, test_frac)
    dec_train, dec_test = _split(decadal, test_frac)

    # native: daily model -> corrected daily discharge
    m_daily = factory()
    m_daily.fit(daily_train)
    daily_test = daily_test.assign(q_pred_native=m_daily.predict(daily_test))

    # transfer: decadal log-residual broadcast onto every day of its decade
    m_dec = factory()
    m_dec.fit(dec_train)
    dec_res = dec_test[[GAUGE_COL, DATE_COL]].copy()
    dec_res["log_residual_dec"] = np.asarray(m_dec.predict_residual(dec_test), float)
    dec_res = dec_res.rename(columns={DATE_COL: "dec_date"})

    work = daily_test.copy()
    work["dec_date"] = _decade_date(work[DATE_COL])
    work = work.merge(dec_res, on=[GAUGE_COL, "dec_date"], how="inner")
    if work.empty:
        log.warning("decadal->daily transfer: no overlapping days to score")
        return _nan_row("decadal->daily", "daily")

    work["q_pred_transfer"] = back_transform(
        work[SIM_COL].to_numpy(float), work["log_residual_dec"].to_numpy(float))

    transfer = _median_skill(work, OBS_COL, "q_pred_transfer")
    native = _median_skill(work, OBS_COL, "q_pred_native")
    raw = _median_skill(work, OBS_COL, SIM_COL)
    return _row("decadal->daily", "daily", work, transfer, native, raw)


# --------------------------------------------------------------------------- #
#  Public entry point: cross-scale transfer                                   #
# --------------------------------------------------------------------------- #
def cross_scale_transfer(daily_df: pd.DataFrame, decadal_df: pd.DataFrame,
                         model_factory: ModelFactory | None = None, *,
                         test_frac: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Quantify how much skill a corrector keeps when transferred across scales.

    Trains the supplied corrector at each temporal resolution and deploys it at
    the other — *daily -> decadal* by aggregating the corrected daily discharge to
    the Central-Asian decades, and *decadal -> daily* by broadcasting the decadal
    log-residual onto each day — then reports the cross-scale skill alongside a
    model trained *natively* at the evaluation resolution.  Every corrector is
    fitted on the earliest ``1 - test_frac`` of its record and scored on the
    strictly posterior remainder (leakage-safe; see
    :func:`sbc.validation.splits.temporal_split`).  Both directions are scored on
    the intersection of evaluation periods common to the transfer and native
    constructions.

    Parameters
    ----------
    daily_df : pandas.DataFrame
        Daily modelling table — engineered features
        (:func:`sbc.features.engineering.build_features` with ``scale="daily"``)
        and, ideally, a ``regime`` label already attached.  Must share its
        gauges/basins with ``decadal_df``.
    decadal_df : pandas.DataFrame
        Decadal (10-day) modelling table for the *same* gauges, with decadal-scale
        engineered features.
    model_factory : callable, optional
        Zero-argument factory returning a fresh, untrained
        :class:`~sbc.models.base.BaseCorrector`.  Defaults to a deterministic,
        untuned LightGBM corrector.
    test_frac : float, default 0.3
        Posterior fraction of each gauge's record held out for evaluation.
    seed : int, default 0
        Global reproducibility seed (also wired into the default factory).

    Returns
    -------
    pandas.DataFrame
        One row per transfer direction with columns
        ``direction, eval_scale, n_gauges, n_periods, kge_raw, kge_native,
        kge_transfer, delta_kge, retention, nse_native, nse_transfer, pbias_raw,
        pbias_native, pbias_transfer``.  KGE'/NSE are median per-gauge; ``retention``
        is the skill-score ratio of transferred to native skill (see module
        docstring).
    """
    seed_everything(int(seed))
    daily = validate(daily_df).reset_index(drop=True)
    decadal = validate(decadal_df).reset_index(drop=True)
    if model_factory is None:
        model_factory = _default_factory(int(seed))

    rows = [
        _daily_to_decadal(daily, decadal, model_factory, test_frac),
        _decadal_to_daily(daily, decadal, model_factory, test_frac),
    ]
    out = pd.DataFrame(rows)[list(_TRANSFER_COLS)]
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].round(4)
    if not out.empty:
        log.info("cross_scale_transfer retention: %s",
                 dict(zip(out["direction"], out["retention"])))
    return out


# --------------------------------------------------------------------------- #
#  Public entry point: aggregation-consistency diagnostic                     #
# --------------------------------------------------------------------------- #
def consistency_check(daily_pred: pd.DataFrame, decadal_obs: pd.DataFrame, *,
                      pred_col: str = PRED_COL, obs_col: str = OBS_COL) -> dict:
    """Does the daily-corrected series aggregate back to the decadal truth?

    Aggregates a daily-corrected discharge series to the Central-Asian decades
    (mean within each gauge-decade) and compares it to the decadal observations,
    reporting the **mean absolute decadal discrepancy** — a scale-coherence /
    volume-conservation diagnostic for the multi-scale framework.  When the daily
    table also carries raw GloFAS (``q_glofas``), the same aggregated discrepancy
    for raw GloFAS is returned for context, so one can read off whether the daily
    correction makes the decadal aggregate *closer* to the bulletin truth.

    Parameters
    ----------
    daily_pred : pandas.DataFrame
        Daily table with ``code``, ``date`` and a corrected-discharge column
        ``pred_col`` (default :data:`sbc.schemas.PRED_COL`).
    decadal_obs : pandas.DataFrame
        Decadal table with ``code``, decade ``date`` and the observed discharge
        ``obs_col`` (default :data:`sbc.schemas.OBS_COL`).  Dates must be the
        decade representative dates (day 5/15/25).
    pred_col : str, default ``"q_pred"``
        Corrected-discharge column in ``daily_pred``.
    obs_col : str, default ``"q_obs"``
        Observed-discharge column in ``decadal_obs``.

    Returns
    -------
    dict
        ``mean_abs_discrepancy`` / ``median_abs_discrepancy`` [m3 s-1],
        ``rel_discrepancy_pct`` (mean abs discrepancy as a percentage of mean
        decadal observed flow), the raw-GloFAS counterparts
        (``raw_glofas_mean_abs_discrepancy`` / ``raw_glofas_rel_discrepancy_pct``,
        ``NaN`` when GloFAS is absent), ``n_decades``, ``n_gauges`` and a
        per-gauge breakdown DataFrame under ``per_gauge``.
    """
    if pred_col not in daily_pred.columns:
        raise ValueError(f"daily_pred missing prediction column {pred_col!r}")
    if obs_col not in decadal_obs.columns:
        raise ValueError(f"decadal_obs missing observation column {obs_col!r}")

    has_glofas = SIM_COL in daily_pred.columns
    value_cols = [pred_col] + ([SIM_COL] if has_glofas else [])
    agg = _aggregate_to_decadal(daily_pred, value_cols)
    agg = agg.rename(columns={pred_col: "q_pred_agg", SIM_COL: "q_glofas_agg"})

    ref = decadal_obs[[GAUGE_COL, DATE_COL, obs_col]].copy()
    merged = agg.merge(ref, on=[GAUGE_COL, DATE_COL], how="inner")
    if merged.empty:
        raise ValueError("consistency_check: no overlapping (code, decade) keys "
                         "between aggregated daily predictions and decadal obs")

    merged["abs_disc"] = (merged["q_pred_agg"] - merged[obs_col]).abs()
    if has_glofas:
        merged["abs_disc_raw"] = (merged["q_glofas_agg"] - merged[obs_col]).abs()

    mean_obs = float(merged[obs_col].mean())
    mean_abs = float(merged["abs_disc"].mean())
    raw_mean_abs = float(merged["abs_disc_raw"].mean()) if has_glofas else float("nan")

    per = (merged.groupby(GAUGE_COL)
           .agg(mae=("abs_disc", "mean"),
                mean_obs=(obs_col, "mean"),
                n_decades=("abs_disc", "size"))
           .reset_index())
    per["rel_pct"] = 100.0 * per["mae"] / per["mean_obs"].where(per["mean_obs"] != 0)

    return {
        "mean_abs_discrepancy": mean_abs,
        "median_abs_discrepancy": float(merged["abs_disc"].median()),
        "rel_discrepancy_pct": 100.0 * mean_abs / mean_obs if mean_obs != 0 else float("nan"),
        "raw_glofas_mean_abs_discrepancy": raw_mean_abs,
        "raw_glofas_rel_discrepancy_pct": (
            100.0 * raw_mean_abs / mean_obs if (has_glofas and mean_obs != 0) else float("nan")),
        "n_decades": int(len(merged)),
        "n_gauges": int(merged[GAUGE_COL].nunique()),
        "per_gauge": per.round(4),
    }


# --------------------------------------------------------------------------- #
#  Self-test (synthetic, small, < 3 min)                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .features.engineering import build_features
    from .features.regimes import classify_regimes
    from .synthetic import decadal_aggregate, generate

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    # Same seed -> daily and decadal tables describe the *same* gauges/truth.
    raw_daily = generate(scale="daily", years=8, n_basins=3,
                         gauges_per_basin=(2, 3), seed=0)
    decadal_raw = decadal_aggregate(raw_daily)          # demo: decadal_aggregate

    daily = classify_regimes(build_features(raw_daily, scale="daily"))
    decadal = classify_regimes(build_features(decadal_raw, scale="decadal"))
    print(f"[multiscale] daily   : {len(daily):6d} rows, "
          f"{daily['code'].nunique()} gauges, {daily['basin'].nunique()} basins")
    print(f"[multiscale] decadal : {len(decadal):6d} rows, "
          f"{decadal['code'].nunique()} gauges, {decadal['basin'].nunique()} basins")

    # --- experiment 1: cross-scale transfer with retention ----------------- #
    report = cross_scale_transfer(daily, decadal, seed=0, test_frac=0.3)
    print("\n=== Cross-scale transfer (synthetic, temporal holdout) ===")
    print(report.to_string(index=False))
    print("\nretention = KGE'-skill-score(transfer) / KGE'-skill-score(native); "
          "1.0 = full native skill retained across the resolution change.")
    assert list(report["direction"]) == ["daily->decadal", "decadal->daily"]
    assert (report["n_gauges"] > 0).all(), "no gauges scored"
    assert report["kge_transfer"].notna().all(), "transfer KGE' missing"
    # both cross-scale models should beat raw GloFAS at their evaluation scale
    assert (report["kge_transfer"] > report["kge_raw"]).all(), \
        "cross-scale transfer failed to beat raw GloFAS"

    # --- experiment 2: aggregation-consistency diagnostic ------------------ #
    m = _default_factory(0)()
    train, _ = _split(validate(daily), 0.3)
    m.fit(train)
    daily_pred = daily.assign(**{PRED_COL: m.predict(daily)})
    cons = consistency_check(daily_pred, decadal_raw)
    print("\n=== Decadal aggregation consistency (daily-corrected -> decades) ===")
    for k in ("mean_abs_discrepancy", "median_abs_discrepancy", "rel_discrepancy_pct",
              "raw_glofas_mean_abs_discrepancy", "raw_glofas_rel_discrepancy_pct",
              "n_decades", "n_gauges"):
        print(f"  {k:34s} {cons[k]:.4f}" if isinstance(cons[k], float)
              else f"  {k:34s} {cons[k]}")
    print("\nper-gauge discrepancy (head):")
    print(cons["per_gauge"].head().to_string(index=False))
    assert cons["n_decades"] > 0 and cons["n_gauges"] > 0
    assert np.isfinite(cons["mean_abs_discrepancy"])
    # the corrected daily aggregate should sit closer to the decadal truth than raw GloFAS
    assert cons["mean_abs_discrepancy"] < cons["raw_glofas_mean_abs_discrepancy"], \
        "daily correction did not improve decadal-scale agreement over raw GloFAS"

    print("\nOK: multi-scale transfer + consistency experiments ran and passed.")
