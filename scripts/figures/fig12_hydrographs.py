"""Figure 12 - Example corrected hydrographs.

Illustrative decadal time series on four held-out test gauges spanning the snow
regimes and basins of the study (a Naryn glacier/snow headwater, a strongly
damped Chu gauge, a Syr-Darya tributary and an Amu-Darya transfer gauge). For
each gauge we plot observed discharge, raw GloFAS-ERA5, the LightGBM point
correction and an illustrative QRF 5-95% band, and annotate the raw -> corrected
per-gauge KGE'. Data are built inline from the assembled decadal table.

CAPTION / HONESTY DISCLOSURE (audit M20):
  * The shaded interval is an ILLUSTRATIVE, subsampled QRF 5-95% band - it only
    sketches a predictive interval and is NOT calibrated; on the damped Chu gauge
    its low-flow quantile collapses toward zero, so the band spans orders of
    magnitude (shown on a log y-axis so the full upper bound is rendered rather
    than silently truncated).
  * The corrected LINE is the LightGBM point model, NOT the flagship
    RegimeProbNet. Hue follows the shared series grammar (LightGBM = ochre).
  * The paper's HEADLINE uncertainty quantification is the conformal /
    regime-conditional wrapper evaluated elsewhere, not this QRF sketch.

The (expensive) model fit + prediction is cached to the scratchpad so that
visual/style iterations re-render instantly; pass --refit to rebuild the cache.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sbc.schemas import OBS_COL, SIM_COL
from sbc.validation.metrics import kge_prime
from sbc.viz import style

style.apply()
# Enable matplotlib per-glyph font fallback: Arial lacks the superscript-minus
# (U+207B) used by style.UNIT_Q's "m^3 s^-1" label. Declaring font.family as an
# explicit list (rather than the "sans-serif" alias, which resolves to a single
# face) lets only the missing glyph fall back to DejaVu Sans; Arial stays primary
# for every other character, so the look is unchanged.
import matplotlib as mpl  # noqa: E402

mpl.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

# --- colours: use the shared series grammar so a hue means ONE thing set-wide.
# Critically, the corrected line is LightGBM (NOT the flagship), so it takes the
# canonical "lgbm" hue (ochre) - never the flagship orange - to avoid implying
# the hero hydrograph is the flagship (audit M20 / C3).
C_OBS = style.SERIES_COLORS["observed"]   # observed (reference), black
C_RAW = style.SERIES_COLORS["raw"]        # raw GloFAS-ERA5 (biased baseline), grey
C_COR = style.SERIES_COLORS["lgbm"]       # LightGBM point correction (ochre)
C_BAND = style.SERIES_COLORS["qrf"]       # illustrative QRF 5-95% band (green)

# --- four illustrative TEST gauges (code -> title, regime/basin tag) ----------
GAUGES = [
    ("16936", "Naryn - inflow to Toktogul reservoir", "Naryn basin (core) - snow/glacier-fed"),
    ("15102", "Chu - Kochkorka",                        "Chu basin (core) - strongly damped GloFAS"),
    ("16139", "Kugart - Mikhaylovskoe",                 "Syr Darya basin (core)"),
    ("17089", "Vakhsh - Tutkaul",                       "Amu Darya transfer domain"),
]
CODES = [c for c, _, _ in GAUGES]

CACHE = Path(r"C:/Users/Marano/AppData/Local/Temp/claude/G--MDPI-Q1-2026/"
             r"781d7b47-341d-4e32-8e79-c67713e4fcd6/scratchpad/fig12_panel_data.parquet")


def build_panel_data() -> pd.DataFrame:
    """Fit LightGBM + a (subsampled, illustrative) QRF and return panel arrays
    for the four displayed gauges only."""
    from sbc.data.assemble import assemble
    from sbc.experiment import load_config, prepare
    from sbc.models.boosting import LightGBMCorrector
    from sbc.models.probabilistic_baselines import QRFCorrector
    from sbc.validation.splits import temporal_split

    cfg = load_config()
    df = prepare(assemble("decadal"), "decadal", cfg).reset_index(drop=True)
    df["code"] = df["code"].astype(str)

    tr_mask, te_mask = temporal_split(df, test_frac=cfg["validation"]["temporal_test_frac"])
    train = df[tr_mask].reset_index(drop=True)
    test = df[te_mask].reset_index(drop=True)
    sub = test[test["code"].isin(CODES)].copy()

    # Deterministic LightGBM corrector (full train; fast).
    lgbm = LightGBMCorrector(n_optuna_trials=0, seed=0).fit(train)
    sub["q_cor"] = lgbm.predict(sub)

    # Probabilistic QRF for the uncertainty band - subsample the training rows so
    # the illustrative band fits quickly (the deep-ensemble UQ is evaluated in
    # full elsewhere; here the QRF only sketches a predictive interval).
    n_qrf = min(10000, len(train))
    qtrain = train.sample(n=n_qrf, random_state=0)
    qrf = QRFCorrector(method="gbr", n_estimators=120, learning_rate=0.05,
                       max_depth=3, min_samples_leaf=40, seed=0).fit(qtrain)
    band = qrf.predict_discharge_quantiles(sub, (0.05, 0.95))
    sub["q_lo"], sub["q_hi"] = band[:, 0], band[:, 1]

    out = sub[["code", "date", OBS_COL, SIM_COL, "q_cor", "q_lo", "q_hi"]].copy()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


def break_gaps(x, series, max_gap_days: int = 31):
    """Insert NaN breaks where the decadal record has a gap, so plotted lines
    are not drawn as long straight interpolations across missing periods."""
    x = np.asarray(x, dtype="datetime64[ns]")
    if x.size < 2:
        return x, series
    dt = np.diff(x).astype("timedelta64[D]").astype(int)
    pos = np.where(dt > max_gap_days)[0] + 1
    if pos.size == 0:
        return x, series
    xins = x[pos - 1] + (x[pos] - x[pos - 1]) / 2
    x2 = np.insert(x, pos, xins)
    out = [np.insert(np.asarray(s, float), pos, np.nan) for s in series]
    return x2, out


def main() -> None:
    refit = "--refit" in sys.argv
    if CACHE.exists() and not refit:
        panel = pd.read_parquet(CACHE)
        print("loaded cached panel data", CACHE)
    else:
        panel = build_panel_data()
        print("built and cached panel data", CACHE)
    panel["code"] = panel["code"].astype(str)

    # --- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_2COL, 5.4))
    axes = axes.ravel()
    tags = ["a", "b", "c", "d"]

    for ax, tag, (code, title, sub) in zip(axes, tags, GAUGES):
        g = panel[panel["code"] == code].sort_values("date")
        x = g["date"].to_numpy()
        obs = g[OBS_COL].to_numpy(float)
        raw = g[SIM_COL].to_numpy(float)
        cor = g["q_cor"].to_numpy(float)
        lo = g["q_lo"].to_numpy(float)
        hi = g["q_hi"].to_numpy(float)

        kge_raw = kge_prime(obs, raw)["kge"]
        kge_cor = kge_prime(obs, cor)["kge"]

        # --- LOG y-axis so the FULL upper band is rendered (audit M20) --------
        # The previous signal-scaled linear axis silently truncated the QRF
        # upper bound off-panel (up to 42.5% of steps on Chu, ~8x off-screen),
        # making the UQ look far tighter than it is. A log axis spans the whole
        # band; the (uncalibrated, illustrative) low-flow tail that collapses
        # toward zero is floored at the axis and flagged with down-markers.
        hi_max = float(np.nanmax(hi))
        lo_pos = lo[np.isfinite(lo) & (lo > 0)]
        bottom = max(float(np.nanmin(lo_pos)) * 0.9, hi_max * 1e-3)
        top = hi_max * 1.18

        # break lines/band across record gaps (no straight interpolation)
        xb, (obs_b, raw_b, cor_b, lo_b, hi_b) = break_gaps(x, [obs, raw, cor, lo, hi])
        # Floor the band's lower edge at the axis bottom for the fill (keep NaN
        # gaps as NaN so gaps stay broken); the true min is disclosed in-panel.
        lo_fill = np.where(np.isnan(lo_b), np.nan, np.maximum(lo_b, bottom))

        ax.fill_between(xb, lo_fill, hi_b, color=C_BAND, alpha=0.22, linewidth=0, zorder=1)
        ax.plot(xb, raw_b, color=C_RAW, lw=1.0, zorder=2)
        ax.plot(xb, cor_b, color=C_COR, lw=1.4, zorder=4)
        ax.plot(xb, obs_b, color=C_OBS, lw=1.0, zorder=5)

        ax.set_yscale("log")
        ax.set_ylim(bottom, top)
        from matplotlib.ticker import LogLocator, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(style.thousands_formatter())

        ax.set_title(title, fontsize=8.5, pad=11)
        ax.text(0.5, 1.01, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color="#555555")
        style.panel(ax, tag)

        # raw -> corrected KGE' annotation (typographic minus, real arrow, KGE')
        # zorder above every plotted series so the orange line / markers never
        # overdraw the digits (matplotlib text defaults below line zorder).
        txt = f"{style.KGE_PRIME}: {style.num(kge_raw, 2)} {style.ARROW} {style.num(kge_cor, 2)}"
        ax.text(0.035, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold", color=C_COR, zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          lw=0.6, alpha=1.0))

        # --- per-panel notes: band extent, lower-bound floor, data gaps -------
        # band-range lower bound: adaptive precision so a sub-unit value (the Chu
        # low-flow quantile collapses to ~0.006) is NOT rounded to a misleading
        # "0" on the log axis; show enough decimals to read as a real number.
        lo_min = float(np.nanmin(lo_pos))
        lo_dec = 0 if lo_min >= 10 else (1 if lo_min >= 1 else 3)
        note = [f"QRF band: {style.num(lo_min, lo_dec)}–"
                f"{style.num(hi_max, 0)} {style.UNIT_Q}"]
        # data-gap note (decadal record gaps were broken, not interpolated)
        xd = np.asarray(x, dtype="datetime64[ns]")
        dgap = np.diff(xd).astype("timedelta64[D]").astype(int) if xd.size > 1 else np.array([])
        if dgap.size and dgap.max() > 31:
            note.append(f"record gaps broken (max {int(dgap.max())} d)")
        # zorder=12 + opaque box: keep the disclosure legible over the orange
        # LightGBM line and the floor down-markers (audit M20 legibility).
        ax.text(0.965, 0.045, "\n".join(note), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.2, color="#444444",
                linespacing=1.35, zorder=12,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd",
                          lw=0.5, alpha=1.0))

        ax.margins(x=0.01)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in (axes[0], axes[2]):
        ax.set_ylabel(style.discharge_label("Discharge"))

    # --- shared legend (does not overlap data) -------------------------------
    handles = [
        Line2D([0], [0], color=C_OBS, lw=1.4, label="Observed"),
        Line2D([0], [0], color=C_RAW, lw=1.4, label="Raw GloFAS-ERA5"),
        Line2D([0], [0], color=C_COR, lw=1.6, label="LightGBM point correction"),
        Patch(facecolor=C_BAND, alpha=0.22, label="Illustrative QRF 5–95% band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.6, handlelength=1.8)


    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=2.0, h_pad=2.4)
    paths = style.savefig(fig, "fig12_hydrographs")
    print("wrote", paths[0])


if __name__ == "__main__":
    main()
