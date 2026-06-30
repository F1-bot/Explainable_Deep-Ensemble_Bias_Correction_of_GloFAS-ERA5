"""Figure 13 - Daily Naryn case study (high-resolution snow freshet).

Panels:
  (a) Corrected KGE' per model for the daily Naryn CASE STUDY (n=2 gauges),
      with per-gauge points + min-max whiskers and a single raw reference line.
  (b) Corrected percent bias (PBIAS) per model, same per-gauge overlay.
  (c) Daily hydrograph for one Naryn gauge (16055) over two snowmelt-freshet
      cycles: GloFAS-ERA5 catastrophically underestimates and damps the
      freshet; an illustrative LightGBM correction recovers the peak toward
      observations (LightGBM shown, not the RegimeProbNet flagship).

Honest framing: this is a CASE STUDY on only two daily Naryn gauges. No single
model is a general daily winner; the learned correctors (XGBoost, LightGBM,
CatBoost, EA-LSTM, RegimeProbNet) tie - their corrected KGE' differences
(0.78-0.83) are far smaller than the gauge-to-gauge spread (~0.13-0.23), and
EA-LSTM edges the flagship. No model is colour- or weight-crowned.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"G:/MDPI Q1-2026/src")
from sbc.viz import style  # noqa: E402

style.apply()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TABLES = r"G:/MDPI Q1-2026/results/tables"
GAUGE = "16055"            # representative Naryn gauge for the hydrograph
WIN = ("2009-10-01", "2011-09-30")  # two complete snowmelt-freshet cycles
FLAGSHIP = "regimeprobnet"
LEARNED = {"xgb", "lgbm", "catboost", "ealstm", "regimeprobnet"}

# One colour per role/model, shared with the rest of the set (audit C3 / M3).
RAW_C = style.SERIES_COLORS["raw"]          # raw GloFAS-ERA5 reference
CORR_C = style.SERIES_COLORS["corrected"]   # generic corrected (bars)
LGBM_C = style.SERIES_COLORS["lgbm"]        # the LightGBM corrector line in (c)
OBS_C = style.SERIES_COLORS["observed"]     # observed gauge series
PT_C = "#222222"                            # per-gauge marker / whisker

# --------------------------------------------------------------------------- #
#  (a)/(b) metrics from the saved daily summary + per-gauge spread             #
# --------------------------------------------------------------------------- #
summ = pd.read_csv(f"{TABLES}/summary_real_daily.csv")
summ = summ[summ["split"] == "temporal"].copy()
# order models top-to-bottom by corrected skill (best at the top of the axis)
summ = summ.sort_values("kge", ascending=True).reset_index(drop=True)
models = summ["model"].tolist()
# short y-labels so panel (a) does not overhang far left (which made the full-
# width panel (c) look shifted right relative to it); the flagship is named in
# the caption, not crowned on the axis (audit M4)
SHORT = {"regimeprobnet": "RegimeProbNet", "ealstm": "EA-LSTM",
         "catboost": "CatBoost", "lgbm": "LightGBM", "xgb": "XGBoost",
         "qmap": "Quantile mapping", "scaling": "Linear scaling"}
labels = [SHORT.get(m, style.label(m)) for m in models]
y = np.arange(len(models))
hbar = 0.55

RAW_KGE = float(summ["kge_raw"].iloc[0])      # identical raw baseline for all
RAW_PBIAS = float(summ["pbias_raw"].iloc[0])

# per-gauge values (n=2) so the panels carry their own uncertainty
pg = pd.read_csv(f"{TABLES}/per_gauge_real_daily.csv")
pg = pg[pg["split"] == "temporal"]
kge_pg = {m: pg[pg["model"] == m].sort_values("code")["kge"].tolist() for m in models}
pbias_pg = {m: pg[pg["model"] == m].sort_values("code")["pbias"].tolist() for m in models}
N_GAUGES = int(summ["n_gauges"].iloc[0])

# --------------------------------------------------------------------------- #
#  (c) inline daily hydrograph: assemble -> prepare -> fit LightGBM           #
# --------------------------------------------------------------------------- #
from sbc.experiment import load_config, prepare  # noqa: E402
from sbc.models.boosting import LightGBMCorrector  # noqa: E402
from sbc.validation.splits import temporal_split  # noqa: E402

cfg = load_config(None)
tbl = pd.read_parquet(r"G:/MDPI Q1-2026/datasets/processed/model_table_daily.parquet")
tbl = prepare(tbl, "daily", cfg).reset_index(drop=True)
tr, te = temporal_split(tbl, cfg["validation"]["temporal_test_frac"])
lgbm = LightGBMCorrector(n_optuna_trials=0).fit(tbl[tr])
tbl["q_corr"] = lgbm.predict(tbl)
tbl["test"] = te

g = tbl[tbl["code"] == GAUGE].sort_values("date").reset_index(drop=True)
w = g[(g["date"] >= WIN[0]) & (g["date"] <= WIN[1])].copy()
assert bool(w["test"].all()), "hydrograph window must be inside the held-out test period"


def _kge(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    r = np.corrcoef(obs, sim)[0, 1]
    beta = sim.mean() / obs.mean()
    gamma = (sim.std() / sim.mean()) / (obs.std() / obs.mean())
    return 1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


kge_raw_g = _kge(w["q_obs"], w["q_glofas"])
kge_corr_g = _kge(w["q_obs"], w["q_corr"])

# --------------------------------------------------------------------------- #
#  Figure layout                                                              #
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(style.WIDTH_2COL, 5.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                      hspace=0.95, wspace=0.30,
                      left=0.175, right=0.985, top=0.93, bottom=0.085)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])


def _overlay(ax, vals_by_model):
    """Per-gauge points + min-max whisker (the n=2 uncertainty, drawn in-panel)."""
    for i, m in enumerate(models):
        vals = vals_by_model[m]
        if len(vals) > 1:
            ax.hlines(y[i], min(vals), max(vals), color=PT_C, lw=1.0, zorder=5)
        ax.plot(vals, [y[i]] * len(vals), marker="o", ls="none", mfc="white",
                mec=PT_C, mew=0.9, ms=4.2, zorder=6)


# --- (a) KGE' --------------------------------------------------------------- #
ax_a.barh(y, summ["kge"], height=hbar, color=CORR_C,
          edgecolor="white", linewidth=0.4, zorder=3)
_overlay(ax_a, kge_pg)
ax_a.axvline(RAW_KGE, color=RAW_C, linewidth=1.4, ls=(0, (4, 2)), zorder=4)
ax_a.text(RAW_KGE + 0.012, len(models) - 0.4,
          f"raw {style.num(RAW_KGE, 2)}", color=RAW_C, fontsize=7,
          ha="left", va="center")
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(-0.6, len(models) - 0.4)
ax_a.set_yticks(y)
ax_a.set_yticklabels(labels)
ax_a.set_xlabel(style.KGE_PRIME)
ax_a.set_title(f"Corrected skill ({style.KGE_PRIME})", fontsize=9, pad=4)
ax_a.grid(axis="y", visible=False)
# the learned correctors are tied within the n=2 spread - group, do not crown
lo = min(i for i, m in enumerate(models) if m in LEARNED)
hi = max(i for i, m in enumerate(models) if m in LEARNED)
xb = 0.945
ax_a.plot([xb, xb], [lo - 0.42, hi + 0.42], color="#888888", lw=0.9, zorder=4)
for yy in (lo - 0.42, hi + 0.42):
    ax_a.plot([xb - 0.02, xb], [yy, yy], color="#888888", lw=0.9, zorder=4)
ax_a.text(xb + 0.02, (lo + hi) / 2, "tied (n = 2)", fontsize=6.6,
          color="#666666", ha="center", va="center", rotation=90)
style.panel(ax_a, "a", x=-0.42)

# --- (b) PBIAS -------------------------------------------------------------- #
ax_b.barh(y, summ["pbias"], height=hbar, color=CORR_C,
          edgecolor="white", linewidth=0.4, zorder=3)
_overlay(ax_b, pbias_pg)
ax_b.axvline(RAW_PBIAS, color=RAW_C, linewidth=1.4, ls=(0, (4, 2)), zorder=4)
ax_b.text(RAW_PBIAS + 1.5, len(models) - 0.4,
          f"raw {style.num(RAW_PBIAS, 0, pct=True)}", color=RAW_C, fontsize=7,
          ha="left", va="center")
for yi, v in zip(y, summ["pbias"]):
    ax_b.text(1.5, yi, style.num(v, 0), va="center", ha="left", fontsize=7,
              color="#444444")
ax_b.axvline(0, color="#333333", linewidth=0.8, zorder=2)
ax_b.set_xlim(-92, 8)
ax_b.set_ylim(-0.6, len(models) - 0.4)
ax_b.set_yticks(y)
ax_b.set_yticklabels([])
ax_b.set_xlabel("Percent bias (%)")
ax_b.set_title("Corrected bias (PBIAS)", fontsize=9, pad=4)
ax_b.grid(axis="y", visible=False)
style.panel(ax_b, "b", x=-0.10)

# --- (c) daily hydrograph --------------------------------------------------- #
ax_c.plot(w["date"], w["q_glofas"], color=RAW_C, lw=1.1, ls="--",
          label="Raw GloFAS-ERA5", zorder=3)
ax_c.plot(w["date"], w["q_corr"], color=LGBM_C, lw=1.4,
          label="Corrected (LightGBM)", zorder=4)
ax_c.plot(w["date"], w["q_obs"], color=OBS_C, lw=1.1,
          label="Observed (gauge)", zorder=5)
ax_c.set_ylim(0, max(w["q_obs"].max(), w["q_corr"].max()) * 1.18)
ax_c.set_ylabel(style.discharge_label("Discharge"))
ax_c.yaxis.set_major_formatter(style.thousands_formatter())
ax_c.set_title(f"Daily hydrograph, Naryn gauge {GAUGE} "
               f"(held-out test; LightGBM corrector shown)", fontsize=9, pad=4)
ax_c.annotate(f"{style.KGE_PRIME}  {style.num(kge_raw_g, 2)} "
              f"{style.ARROW} {style.num(kge_corr_g, 2)}  (this window)",
              xy=(0.015, 0.93), xycoords="axes fraction", fontsize=7.5,
              ha="left", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
import matplotlib.dates as mdates  # noqa: E402
ax_c.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
style.panel(ax_c, "c", x=-0.075, y=1.05)

# --- single consolidated legend (audit M4): one colour per role, n labelled - #
leg_handles = [
    Patch(facecolor=CORR_C, edgecolor="white", label="Corrected (per-gauge mean)"),
    Line2D([0], [0], marker="o", ls="none", mfc="white", mec=PT_C, mew=0.9,
           ms=4.2, label=f"Per-gauge value, n = {N_GAUGES} (whisker = range)"),
    Line2D([0], [0], color=RAW_C, lw=1.4, ls="--", label="Raw GloFAS-ERA5"),
    Line2D([0], [0], color=LGBM_C, lw=1.4, label="Corrected (LightGBM), panel (c)"),
    Line2D([0], [0], color=OBS_C, lw=1.1, label="Observed (gauge), panel (c)"),
]
fig.legend(handles=leg_handles, loc="center", bbox_to_anchor=(0.5, 0.475),
           ncol=3, fontsize=7.3, handlelength=1.6, columnspacing=1.4,
           frameon=False)

paths = style.savefig(fig, "fig13_naryn_daily")
print("saved:", paths[0])
print(f"gauge {GAUGE} window {WIN}: KGE' {kge_raw_g:.3f} -> {kge_corr_g:.3f}; "
      f"obs peak {w['q_obs'].max():.0f}, raw peak {w['q_glofas'].max():.0f}, "
      f"corr peak {w['q_corr'].max():.0f} m3/s")
