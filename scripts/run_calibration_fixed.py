"""Earn the *calibrated probabilistic* claim: fixed CRPSS + restored coverage.

The adversarial red-team flagged the paper's flagship as **not** demonstrably
"calibrated probabilistic": the reported PIT was 0.605, the nominal-90 % interval
covered only ~0.72 of observations, and the CRPS skill score was ``-10`` /
``-30``.  That last number is not a model failure but a **unit / space bug** in
the prior calibration code (:func:`scripts.run_enhancements._calibration_and_gate`):
it compared a **discharge-space** model CRPS (obs and predictive quantiles in
``m3 s-1``, so CRPS is *O*(10-100)) against a **residual-space** climatological
reference CRPS (the log-residual is *O*(1), so its CRPS is *O*(0.3)).  Dividing
the two yields ``1 - 100/0.3`` ~ ``-300``-scale nonsense.  CRPS has units, so the
model and the reference must live in the *same* space.

This script closes that exposure on the real decadal data (the maintainer runs it
without ``--dry-run``; ``--dry-run`` exercises the identical wiring on a tiny
synthetic table so the logic can be proven green in < 3 min).  For the temporal
and PUR (core -> transfer) splits it:

1. **Fits the flagship** :class:`~sbc.models.regime_prob_net.RegimeProbNet` on the
   training block only (disjoint from the conformal calibration and test sets).

2. **Computes calibration correctly.**  PIT, KS, the 50/80/90 % central-interval
   coverage and sharpness are read off the flagship's predictive **discharge**
   quantiles (PIT / coverage are invariant under the monotone back-transform, so
   this matches the residual-space diagnostics while reporting sharpness in
   physical ``m3 s-1``).  The CRPS skill score is rebuilt the *honest* way: the
   climatological reference is an **ensemble in discharge space** -- the training
   ``q_obs`` resampled both *pooled* and *per-season* (calendar month) -- and both
   the model and the reference are scored with the **same**
   :func:`~sbc.validation.metrics.crps_ensemble` in ``m3 s-1``, so
   ``CRPSS = 1 - crps_model / crps_clim`` is now correctly signed.  The old
   mismatched-space ratio is also recomputed and reported side-by-side
   (``crpss_residualspace_bug``) so the fix is auditable.

3. **Restores coverage with conformal prediction** on the *same* splits:
   split-conformal (absolute + CQR) and EnbPI from
   :mod:`sbc.validation.conformal`, plus the regime-conditional (hard- and
   soft-Mondrian) estimators from :mod:`sbc.validation.regime_conformal`.  Each
   method's **empirical coverage vs the nominal 0.90**, overall and *per
   hydrological regime*, is tabulated -- the evidence that the under-coverage is
   fixed where it matters (the melt / glacier / rain-on-snow flood states).

Outputs (``results/tables/``)::

    calibration_fixed_<tag>.csv           # per-split PIT/KS/coverage/sharpness/CRPSS
    calibration_fixed_coverage_<tag>.csv  # per-split/method/regime empirical coverage

Run::

    PYTHONPATH=src python -m scripts.run_calibration_fixed --dry-run          # synthetic smoke
    PYTHONPATH=src python scripts/run_calibration_fixed.py                    # REAL decadal
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.schemas import OBS_COL, SIM_COL, TARGET_COL, make_target, validate
from sbc.utils import ensure_dir, get_logger, seed_everything

log = get_logger("calib_fixed")
T0 = time.time()

#: nominal central-interval levels reported in the calibration table
CENTRAL_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9)
#: quantile grid for PIT / reliability / coverage (multiples of 0.05 span the tails)
LEVELS_CAL: np.ndarray = np.round(np.linspace(0.05, 0.95, 19), 4)
#: finer, uniform-probability grid used as the *model ensemble* for discharge CRPS
LEVELS_CRPS: np.ndarray = np.round(np.linspace(0.01, 0.99, 99), 4)


def _stamp(msg: str) -> None:
    log.info("[%5.0fs] %s", time.time() - T0, msg)


# --------------------------------------------------------------------------- #
#  Data preparation                                                           #
# --------------------------------------------------------------------------- #
def _prepare(df: pd.DataFrame, scale: str) -> pd.DataFrame:
    """Validate -> engineer features -> classify regimes (mirrors experiment.prepare)."""
    from sbc.features.engineering import build_features
    from sbc.features.regimes import classify_regimes

    df = validate(df)
    df = build_features(df, scale=scale)
    df = classify_regimes(df)
    return df.reset_index(drop=True)


def load_data(dry_run: bool, scale: str = "decadal") -> pd.DataFrame:
    """Assemble (real) or synthesise (dry-run) the prepared decadal modelling table."""
    if dry_run:
        from sbc.synthetic import generate

        raw = generate(scale=scale, years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
        _stamp(f"synthetic {scale}: {raw.shape}")
    else:
        from sbc.data.assemble import assemble

        raw = assemble(scale)
        _stamp(f"assembled real {scale}: {raw.shape}")
    df = _prepare(raw, scale)
    _stamp(f"prepared {df.shape}: gauges={df['code'].nunique()} basins={df['basin'].nunique()}")
    return df


# --------------------------------------------------------------------------- #
#  Flagship                                                                   #
# --------------------------------------------------------------------------- #
def fit_flagship(train: pd.DataFrame, valid: pd.DataFrame | None, *, epochs: int,
                 hidden: int, seq_len: int, seed: int):
    """Fit the flagship RegimeProbNet on ``train`` (K=5 experts aligned to regimes)."""
    from sbc.models import load_all
    from sbc.models.regime_prob_net import RegimeProbNet

    load_all()  # honour the registry contract even though we use the class directly
    model = RegimeProbNet(
        K=5, hidden=hidden, seq_len=seq_len, expert_hidden=max(16, hidden // 2),
        gate_hidden=max(16, hidden // 2), epochs=int(epochs), batch_size=256,
        patience=max(3, epochs // 4), lambda_gate=0.5, lambda_phys=0.0,
        loss="crps", seed=int(seed), verbose=False,
    )
    model.fit(train, valid=valid)
    return model


# --------------------------------------------------------------------------- #
#  Discharge-space CRPS helpers (the bug fix)                                 #
# --------------------------------------------------------------------------- #
def _batched_crps_ensemble(obs: np.ndarray, ens: np.ndarray, batch: int = 1000) -> float:
    """Mean ensemble CRPS over finite rows, computed in memory-bounded row chunks.

    Thin wrapper around :func:`sbc.validation.metrics.crps_ensemble` (which forms
    the *O(m^2)* pairwise term densely) that pre-filters non-finite rows and
    averages chunk means weighted by chunk size -- exact because every retained
    row is finite -- so peak memory stays at ``batch * m * m`` floats regardless
    of the (large, real-decadal) row count.

    Parameters
    ----------
    obs : ndarray, shape (n,)
        Observations in discharge space.
    ens : ndarray, shape (n, m)
        Ensemble members in discharge space (model quantile grid or climatology).
    batch : int, default 1000
        Row-chunk size bounding peak memory.

    Returns
    -------
    float
        Mean CRPS (``m3 s-1``); ``nan`` when no finite row exists.
    """
    from sbc.validation.metrics import crps_ensemble

    obs = np.asarray(obs, float)
    ens = np.asarray(ens, float)
    if ens.ndim == 1:
        ens = ens[:, None]
    keep = np.isfinite(obs) & np.isfinite(ens).all(axis=1)
    obs, ens = obs[keep], ens[keep]
    if obs.size == 0:
        return float("nan")
    total, count = 0.0, 0
    for s in range(0, obs.size, int(batch)):
        chunk = obs[s: s + batch]
        cm = crps_ensemble(chunk, ens[s: s + batch])
        if np.isfinite(cm):
            total += cm * chunk.size
            count += chunk.size
    return total / count if count else float("nan")


def _climatological_ensemble(train: pd.DataFrame, test: pd.DataFrame, n_members: int,
                             by: str, rng: np.random.Generator,
                             min_season: int = 20) -> np.ndarray:
    """Climatological **discharge** ensemble for each test row by resampling train ``q_obs``.

    Parameters
    ----------
    train, test : DataFrame
        Training (resampled) and test modelling tables.
    n_members : int
        Ensemble size ``m`` per test row.
    by : {"pooled", "season"}
        ``"pooled"`` resamples the whole training ``q_obs``; ``"season"`` resamples
        within the test row's calendar month (falling back to the pooled record
        for months with fewer than ``min_season`` training values).
    rng : numpy.random.Generator
        Source of the resampling draws.
    min_season : int, default 20
        Minimum per-month training count before falling back to the pooled record.

    Returns
    -------
    ndarray, shape (len(test), n_members)
        The climatological ensemble in ``m3 s-1``.
    """
    q = train[OBS_COL].to_numpy(float)
    pool = q[np.isfinite(q)]
    n, m = len(test), int(n_members)
    if pool.size == 0:
        return np.full((n, m), np.nan)
    if by == "pooled":
        return np.tile(rng.choice(pool, size=m, replace=True), (n, 1))

    mon_tr = pd.to_datetime(train["date"]).dt.month.to_numpy()[np.isfinite(q)]
    mon_te = pd.to_datetime(test["date"]).dt.month.to_numpy()
    out = np.empty((n, m), float)
    for mo in np.unique(mon_te):
        sel = mon_te == mo
        sub = pool[mon_tr == mo]
        src = sub if sub.size >= int(min_season) else pool
        out[sel] = rng.choice(src, size=(int(sel.sum()), m), replace=True)
    return out


# --------------------------------------------------------------------------- #
#  Calibration row (per split)                                                #
# --------------------------------------------------------------------------- #
def calibration_row(split: str, model, train: pd.DataFrame, test: pd.DataFrame,
                    n_members: int, seed: int) -> dict:
    """Compute the fixed calibration bundle for one split.

    Returns PIT mean / KS, mean absolute calibration error, 50/80/90 % coverage and
    sharpness (from the flagship's predictive discharge quantiles), and the
    correctly-signed discharge-space CRPS skill score against pooled and per-season
    climatological ensembles -- plus the old mismatched-space ratio for contrast.
    """
    from sbc.validation.calibration import calibration_summary
    from sbc.validation.metrics import crps_gaussian

    rng = np.random.default_rng(seed)
    obs = test[OBS_COL].to_numpy(float)
    sim = test[SIM_COL].to_numpy(float)

    # -- PIT / coverage / sharpness from predictive DISCHARGE quantiles --------
    q_disc_cal = np.asarray(model.predict_discharge_quantiles(test, tuple(LEVELS_CAL)), float)
    summ = calibration_summary(obs, q_disc_cal, LEVELS_CAL, central_levels=CENTRAL_LEVELS)

    # -- model CRPS: ensemble in DISCHARGE space (uniform-probability quantiles) -
    q_disc_crps = np.asarray(model.predict_discharge_quantiles(test, tuple(LEVELS_CRPS)), float)
    crps_model = _batched_crps_ensemble(obs, q_disc_crps)

    # -- climatological references, ALSO in discharge space (the fix) ----------
    clim_pool = _climatological_ensemble(train, test, n_members, "pooled", rng)
    clim_seas = _climatological_ensemble(train, test, n_members, "season", rng)
    crps_clim_pool = _batched_crps_ensemble(obs, clim_pool)
    crps_clim_seas = _batched_crps_ensemble(obs, clim_seas)

    def _ss(ref: float) -> float:
        return float(1.0 - crps_model / ref) if (np.isfinite(ref) and ref > 0) else float("nan")

    # -- reproduce the OLD bug for transparency: residual-space clim reference -
    y_res = make_target(obs, sim)
    mu0 = float(make_target(train[OBS_COL], train[SIM_COL]).mean())
    sd0 = float(make_target(train[OBS_COL], train[SIM_COL]).std() + 1e-9)
    crps_ref_resid = crps_gaussian(y_res, np.full(y_res.size, mu0), np.full(y_res.size, sd0))
    crpss_bug = (float(1.0 - crps_model / crps_ref_resid)
                 if (np.isfinite(crps_ref_resid) and crps_ref_resid > 0) else float("nan"))

    row = {
        "split": split,
        "n": int(np.isfinite(obs).sum()),
        "pit_mean": summ["pit_mean"],
        "pit_ks": summ["pit_ks"],
        "calibration_error": summ["calibration_error"],
        "cov_0.5": summ["cov_0.5"], "width_0.5": summ["width_0.5"],
        "cov_0.8": summ["cov_0.8"], "width_0.8": summ["width_0.8"],
        "cov_0.9": summ["cov_0.9"], "width_0.9": summ["width_0.9"],
        "crps_model_discharge": crps_model,
        "crps_clim_pooled": crps_clim_pool,
        "crps_clim_season": crps_clim_seas,
        "crpss_pooled": _ss(crps_clim_pool),
        "crpss_season": _ss(crps_clim_seas),
        "crps_ref_residualspace": float(crps_ref_resid),
        "crpss_residualspace_bug": crpss_bug,
    }
    log.info("[%s] PIT=%.3f KS=%.3f cov90=%.3f | CRPS model=%.4g clim(pool)=%.4g "
             "clim(seas)=%.4g | CRPSS pooled=%+.3f season=%+.3f (buggy residual-ref=%+.2f)",
             split, row["pit_mean"], row["pit_ks"], row["cov_0.9"], crps_model,
             crps_clim_pool, crps_clim_seas, row["crpss_pooled"], row["crpss_season"],
             crpss_bug)

    # best-effort diagnostic figures (Agg-safe) -------------------------------
    try:
        from sbc.validation.calibration import (pit_values, save_pit_histogram,
                                                save_reliability_diagram)
        save_reliability_diagram(obs, q_disc_cal, LEVELS_CAL,
                                 path=PATHS.figures / f"calibration_fixed_reliability_{split}.png",
                                 title=f"RegimeProbNet reliability ({split})")
        save_pit_histogram(pit_values(obs, q_disc_cal, levels=LEVELS_CAL),
                           path=PATHS.figures / f"calibration_fixed_pit_{split}.png",
                           title=f"RegimeProbNet PIT ({split})")
    except Exception as exc:  # pragma: no cover - plotting is optional
        log.debug("[%s] calibration figures skipped: %s", split, exc)
    return row


# --------------------------------------------------------------------------- #
#  Conformal coverage rows (per split / method / regime)                      #
# --------------------------------------------------------------------------- #
def _coverage_rows(split: str, method: str, lower, upper, y, regimes, alpha: float,
                   overall_cov: float, overall_width: float) -> list[dict]:
    """Long-format rows: one ``__overall__`` row plus one per populated regime."""
    from sbc.validation.regime_conformal import per_regime_coverage

    nominal = 1.0 - float(alpha)
    rows = [{
        "split": split, "method": method, "regime": "__overall__",
        "n": int(np.isfinite(np.asarray(y, float)).sum()),
        "coverage": float(overall_cov), "nominal": nominal,
        "gap": float(overall_cov) - nominal, "mean_width": float(overall_width),
    }]
    pr = per_regime_coverage(y, lower, upper, regimes, alpha)
    for r in pr.itertuples(index=False):
        rows.append({
            "split": split, "method": method, "regime": r.regime, "n": int(r.n),
            "coverage": float(r.coverage), "nominal": float(r.nominal),
            "gap": float(r.gap), "mean_width": float(r.mean_width),
        })
    return rows


def conformal_coverage(split: str, model, train: pd.DataFrame, calib: pd.DataFrame,
                       test: pd.DataFrame, *, alpha: float, enbpi_members: int,
                       seed: int) -> list[dict]:
    """Apply every conformal estimator on one split and tabulate coverage vs nominal.

    The flagship is already fitted on ``train`` (disjoint from ``calib`` / ``test``),
    so split- and regime-conditional conformal calibrate leakage-free on ``calib``
    and are evaluated on ``test``; EnbPI bootstraps a light point corrector on
    ``train + calib`` and forms online time-series intervals on ``test``.
    """
    from sbc.models.quantile_mapping import LinearScalingCorrector
    from sbc.validation.conformal import enbpi_intervals, split_conformal
    from sbc.validation.regime_conformal import regime_conditional_conformal

    y_test = (test[TARGET_COL].to_numpy(float) if TARGET_COL in test.columns
              else make_target(test[OBS_COL], test[SIM_COL]))
    test_reg = test["regime"].astype(str).to_numpy()
    rows: list[dict] = []

    # -- split conformal: absolute + CQR around the fitted flagship ------------
    for sc_method, label in (("absolute", "split-absolute"), ("cqr", "split-cqr")):
        try:
            res = split_conformal(model, calib, test, alpha=alpha, method=sc_method)
            rows += _coverage_rows(split, label, res.lower, res.upper, y_test,
                                   test_reg, alpha, res.coverage, res.sharpness)
        except Exception as exc:  # pragma: no cover
            log.warning("[%s] %s failed: %s", split, label, exc)

    # -- regime-conditional conformal: hard (regime) + soft (gate) Mondrian ----
    for by, label in (("regime", "regime-hard"), ("gate", "regime-soft-gate")):
        try:
            res = regime_conditional_conformal(model, calib, test, alpha=alpha,
                                               by=by, method="absolute", min_calib=10)
            rows += _coverage_rows(split, label, res.lower, res.upper, y_test,
                                   test_reg, alpha, res.coverage, res.sharpness)
        except Exception as exc:  # pragma: no cover
            log.warning("[%s] regime-conformal[%s] failed: %s", split, by, exc)

    # -- EnbPI: online time-series intervals around a light point corrector ----
    try:
        train_calib = pd.concat([train, calib], ignore_index=True)
        res = enbpi_intervals(LinearScalingCorrector, train_calib, test, alpha=alpha,
                              n_bootstrap=int(enbpi_members), seed=int(seed))
        # enbpi sorts test by date internally -> align y / regimes to that order
        ts = test.sort_values("date", kind="stable")
        y_ts = (ts[TARGET_COL].to_numpy(float) if TARGET_COL in ts.columns
                else make_target(ts[OBS_COL], ts[SIM_COL]))
        rows += _coverage_rows(split, "enbpi", res.lower, res.upper, y_ts,
                               ts["regime"].astype(str).to_numpy(), alpha,
                               res.coverage, res.sharpness)
    except Exception as exc:  # pragma: no cover
        log.warning("[%s] enbpi failed: %s", split, exc)

    for r in rows:
        if r["regime"] == "__overall__":
            log.info("[%s] %-16s overall coverage=%.3f (nominal %.2f) width=%.4g",
                     split, r["method"], r["coverage"], r["nominal"], r["mean_width"])
    return rows


# --------------------------------------------------------------------------- #
#  Split construction                                                         #
# --------------------------------------------------------------------------- #
def _halve(future: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly halve an (exchangeable) held-out block into calib / test."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(future))
    half = len(future) // 2
    calib = future.iloc[perm[:half]].reset_index(drop=True)
    test = future.iloc[perm[half:]].reset_index(drop=True)
    return calib, test


def build_splits(df: pd.DataFrame, test_frac: float, seed: int
                 ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Build ``{split: (train, calib, test)}`` for the temporal and PUR protocols.

    For *temporal* the per-gauge future block is split into exchangeable calib /
    test halves.  For *PUR* the train side is the core domain and the held-out
    transfer domain is halved into calib / test, so the conformal calibration set
    is exchangeable with the (ungauged) transfer test set.
    """
    from sbc.validation.splits import pur_split, temporal_split

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    tr, fut = temporal_split(df, test_frac=test_frac)
    train = df[tr].reset_index(drop=True)
    calib, test = _halve(df[fut].reset_index(drop=True), seed)
    out["temporal"] = (train, calib, test)

    if "domain" in df.columns and (df["domain"] == "transfer").any():
        ptr, pte = pur_split(df)
        core = df[ptr].reset_index(drop=True)
        calib_p, test_p = _halve(df[pte].reset_index(drop=True), seed + 1)
        out["pur"] = (core, calib_p, test_p)
    else:
        log.warning("transfer domain absent -> skipping the PUR split")
    return out


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def run(dry_run: bool, *, epochs: int, hidden: int, seq_len: int, alpha: float,
        test_frac: float, n_members: int, enbpi_members: int, seed: int) -> dict:
    """Execute the full fixed-calibration + conformal-coverage analysis."""
    seed_everything(seed)
    PATHS.ensure()
    scale = "decadal"
    df = load_data(dry_run, scale=scale)

    splits = build_splits(df, test_frac=test_frac, seed=seed)
    cal_rows: list[dict] = []
    cov_rows: list[dict] = []

    for split, (train, calib, test) in splits.items():
        if len(train) < 30 or len(test) < 15 or len(calib) < 15:
            log.warning("[%s] too few rows (train=%d calib=%d test=%d) -> skip",
                        split, len(train), len(calib), len(test))
            continue
        _stamp(f"[{split}] fitting flagship on train={len(train)} "
               f"(calib={len(calib)} test={len(test)})")
        model = fit_flagship(train, valid=calib, epochs=epochs, hidden=hidden,
                             seq_len=seq_len, seed=seed)
        cal_rows.append(calibration_row(split, model, train, test, n_members, seed))
        cov_rows += conformal_coverage(split, model, train, calib, test, alpha=alpha,
                                       enbpi_members=enbpi_members, seed=seed)

    tag = "synthetic_decadal_dryrun" if dry_run else "real_decadal"
    cal_df = pd.DataFrame(cal_rows)
    cov_df = pd.DataFrame(cov_rows)

    ensure_dir(PATHS.tables)
    cal_path = PATHS.tables / f"calibration_fixed_{tag}.csv"
    cov_path = PATHS.tables / f"calibration_fixed_coverage_{tag}.csv"
    cal_df.to_csv(cal_path, index=False)
    cov_df.to_csv(cov_path, index=False)
    _stamp(f"wrote {cal_path}")
    _stamp(f"wrote {cov_path}")

    _print_report(cal_df, cov_df, alpha)
    return {"calibration": cal_df, "coverage": cov_df,
            "cal_path": cal_path, "cov_path": cov_path}


def _print_report(cal_df: pd.DataFrame, cov_df: pd.DataFrame, alpha: float) -> None:
    """Print a compact console summary of the headline evidence."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("\n===== FIXED CALIBRATION (per split) =====")
    if not cal_df.empty:
        show = cal_df[["split", "n", "pit_mean", "pit_ks", "cov_0.9", "width_0.9",
                       "crpss_pooled", "crpss_season", "crpss_residualspace_bug"]]
        print(show.round(3).to_string(index=False))
        print("note: crpss_pooled / crpss_season are the FIXED (discharge-space) skill "
              "scores; crpss_residualspace_bug reproduces the prior mismatched-space ratio.")
    print(f"\n===== CONFORMAL COVERAGE vs nominal {1 - alpha:.2f} (overall) =====")
    if not cov_df.empty:
        ov = cov_df[cov_df["regime"] == "__overall__"]
        print(ov[["split", "method", "n", "coverage", "nominal", "gap",
                  "mean_width"]].round(3).to_string(index=False))
        print(f"\n===== per-regime coverage (melt / glacier / rain-on-snow) =====")
        melt = cov_df[cov_df["regime"].isin(
            ["melt_freshet", "glacier_melt", "rain_on_snow"])]
        if not melt.empty:
            print(melt[["split", "method", "regime", "n", "coverage", "gap",
                        "mean_width"]].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the wiring on a tiny synthetic decadal table (< 3 min)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="flagship training epochs (default 3 for --dry-run, 40 for real)")
    ap.add_argument("--hidden", type=int, default=None,
                    help="flagship hidden width (default 16 for --dry-run, 64 for real)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="flagship sequence length (default 4 for --dry-run, 12 for real)")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="conformal miscoverage; nominal coverage 1 - alpha (default 0.1)")
    ap.add_argument("--test-frac", type=float, default=0.4,
                    help="temporal held-out fraction (halved into calib/test)")
    ap.add_argument("--samples", type=int, default=150,
                    help="climatological ensemble size for the discharge CRPS reference")
    ap.add_argument("--enbpi-members", type=int, default=None,
                    help="EnbPI bootstrap members (default 5 for --dry-run, 20 for real)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    epochs = a.epochs if a.epochs is not None else (3 if a.dry_run else 40)
    hidden = a.hidden if a.hidden is not None else (16 if a.dry_run else 64)
    seq_len = a.seq_len if a.seq_len is not None else (4 if a.dry_run else 12)
    enbpi_members = (a.enbpi_members if a.enbpi_members is not None
                     else (5 if a.dry_run else 20))

    run(a.dry_run, epochs=epochs, hidden=hidden, seq_len=seq_len, alpha=a.alpha,
        test_frac=a.test_frac, n_members=a.samples, enbpi_members=enbpi_members,
        seed=a.seed)


if __name__ == "__main__":
    main()
