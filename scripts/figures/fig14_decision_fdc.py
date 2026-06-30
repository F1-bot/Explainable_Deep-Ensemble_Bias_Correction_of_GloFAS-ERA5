"""Figure 14 - Decision skill and flow-regime performance.

(a) Brier skill score + ROC-AUC for high-flow (Q90 freshet) and low-flow (Q10)
    decadal (10-day) exceedance events - decision skill (n = 23,420 steps).
(b) Flow-duration-curve segment bias, raw vs corrected, across
    very-high / high / mid / low flow bands, with per-segment delta|bias|.
(c) Monthly KGE' (KGE-prime), raw vs corrected, with the snowmelt-freshet season shaded.

Message (honest): correction reduces peak/mid FDC bias but INCREASES low-flow
overestimation (+6.94% -> +16.53%); decision skill is reported at the decadal
(10-day) exceedance scale.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sbc.viz import style

style.apply()

# M22: deterministic build - no sampling is used (all panels are read straight
# from the on-disk decadal CSVs), but seed defensively so the rendered PNG is
# byte-stable across re-runs.
np.random.seed(0)

TAB = r"G:/MDPI Q1-2026/results/tables"

RAW_C = style.SERIES_COLORS["raw"]
COR_C = style.SERIES_COLORS["corrected"]
BSS_C = style.SERIES_COLORS["corrected"]
AUC_C = "#D55E00"

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
dec = pd.read_csv(f"{TAB}/decision_skill_real_decadal.csv")
fdc = pd.read_csv(f"{TAB}/diag_fdc_segments_real_decadal.csv")
mon = pd.read_csv(f"{TAB}/diag_seasonal_month_real_decadal.csv")

# event order: high-flow (Q90 upper) then low-flow (Q10 lower)
dec = dec.set_index("event")
events = ["Q90_upper", "Q10_lower"]
ev_lab = ["High flow\n(Q90 freshet)", "Low flow\n(Q10)"]
bss = [dec.loc[e, "bss"] for e in events]
auc = [dec.loc[e, "roc_auc"] for e in events]

# FDC segments very-high -> low
seg_order = ["very_high", "high", "mid", "low"]
seg_lab = ["Very high\n(0-2%)", "High\n(2-20%)", "Mid\n(20-70%)", "Low\n(70-100%)"]
fdc = fdc.set_index("segment").loc[seg_order]

# monthly
mon = mon.sort_values("period")
months = mon["period"].to_numpy()
mlab = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# --------------------------------------------------------------------------- #
# Figure layout
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(style.WIDTH_2COL, 5.4))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.92],
              hspace=0.62, wspace=0.30,
              left=0.085, right=0.985, top=0.93, bottom=0.095)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# --------------------------------------------------------------------------- #
# (a) Decision skill: BSS + ROC-AUC
# --------------------------------------------------------------------------- #
x = np.arange(len(events))
w = 0.36
b1 = ax_a.bar(x - w / 2, bss, w, color=BSS_C, label="Brier skill score",
              edgecolor="white", linewidth=0.5, zorder=3)
b2 = ax_a.bar(x + w / 2, auc, w, color=AUC_C, label="ROC-AUC",
              edgecolor="white", linewidth=0.5, zorder=3)
# no-skill references: the 0.5 line is the ROC-AUC no-skill level ONLY; the
# Brier-skill-score no-skill level is BSS = 0 (the bar baseline). Tie each
# reference to its own metric so the 0.5 line is not misread against the BSS bars.
ax_a.axhline(0.5, color=AUC_C, ls=":", lw=0.9, zorder=2)
ax_a.text(1.62, 0.5, "AUC no-skill\n(0.5)", color=AUC_C, fontsize=6.0,
          va="center", ha="left", zorder=8,
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
# BSS = 0 no-skill reference (climatology); dashed in the BSS hue at the baseline
ax_a.axhline(0.0, color=BSS_C, ls=(0, (4, 2)), lw=0.9, zorder=2)
ax_a.text(1.62, 0.045, "BSS no-skill\n(0)", color=BSS_C, fontsize=6.0,
          va="center", ha="left", zorder=8,
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
# n annotation (in-panel): decadal exceedance sample size, placed in the clear

for rects in (b1, b2):
    for r in rects:
        h = r.get_height()
        ax_a.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2, h),
                      xytext=(0, 2), textcoords="offset points",
                      ha="center", va="bottom", fontsize=6.6, fontweight="bold")

ax_a.set_xticks(x)
ax_a.set_xticklabels(ev_lab)
ax_a.set_xlim(-0.6, 2.05)
ax_a.set_ylim(0, 1.08)
ax_a.set_yticks(np.arange(0, 1.01, 0.2))
ax_a.set_ylabel("Skill score (-)")
ax_a.set_title("Decadal (10-day) exceedance decision skill", pad=4)
ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=6.8,
            handlelength=1.1, handletextpad=0.5, columnspacing=1.2, ncol=2)
ax_a.grid(axis="x", visible=False)
style.panel(ax_a, "a")

# --------------------------------------------------------------------------- #
# (b) FDC-segment percent bias, raw vs corrected
# --------------------------------------------------------------------------- #
xs = np.arange(len(seg_order))
b3 = ax_b.bar(xs - w / 2, fdc["pbias_raw"], w, color=RAW_C,
              label="Raw GloFAS-ERA5", edgecolor="white", linewidth=0.5, zorder=3)
b4 = ax_b.bar(xs + w / 2, fdc["pbias"], w, color=COR_C,
              label="Corrected", edgecolor="white", linewidth=0.5, zorder=3)
ax_b.axhline(0.0, color="#333333", lw=0.8, zorder=2)

for rects in (b3, b4):
    for r in rects:
        h = r.get_height()
        off = 2 if h >= 0 else -2
        va = "bottom" if h >= 0 else "top"
        ax_b.annotate(style.num(h, 0), (r.get_x() + r.get_width() / 2, h),
                      xytext=(0, off), textcoords="offset points",
                      ha="center", va=va, fontsize=6.4)

# per-segment delta|bias| (raw -> corrected): positive = |bias| reduced
# (improved), negative = |bias| grew (worsened). Annotate the direction so the
# growing blue low-flow bar is not misread as an improvement.
for xi, d in zip(xs, fdc["d_pbias_abs"].to_numpy()):
    improved = d > 0
    arr = "↓" if improved else "↑"          # down = reduced |bias|
    col = "#444444"   # neutral grey: direction is conveyed by the arrow glyph,
    #                   not a colour that could clash with the Corrected series
    ax_b.text(xi, 36, f"Δ|bias|\n{arr}{style.num(abs(d), 1)}",
              ha="center", va="top", fontsize=5.8, color=col, fontweight="bold",
              linespacing=0.95)


ax_b.set_xticks(xs)
ax_b.set_xticklabels(seg_lab)
ax_b.set_ylabel("Bias (%)")
ax_b.set_ylim(-48, 45)
ax_b.set_title("FDC segment bias (n = 74 gauges)", pad=4)
ax_b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.4,
            handlelength=1.1, handletextpad=0.5, labelspacing=0.25, ncol=1)
ax_b.grid(axis="x", visible=False)
style.panel(ax_b, "b")

# --------------------------------------------------------------------------- #
# (c) Monthly KGE-prime, raw vs corrected, snowmelt season shaded
# --------------------------------------------------------------------------- #
# snowmelt-freshet season (Apr-Jul); months are 1-indexed -> x = month - 1
ax_c.axvspan(3 - 0.5, 7 - 0.5, color="#56B4E9", alpha=0.16, zorder=0)
ax_c.text(4.5 - 0.5, 0.455, "Snowmelt freshet", color="#0072B2",
          fontsize=7.2, ha="center", va="top", fontweight="bold")

xm = months - 1
ax_c.plot(xm, mon["kge_raw"], "-o", color=RAW_C, ms=3.2, lw=1.3,
          label="Raw GloFAS-ERA5", zorder=3)
ax_c.plot(xm, mon["kge"], "-o", color=COR_C, ms=3.6, lw=1.6,
          label="Corrected", zorder=4)
ax_c.axhline(0.0, color="#333333", lw=0.8, zorder=1)

ax_c.set_xticks(xm)
ax_c.set_xticklabels(mlab)
ax_c.set_xlim(-0.5, 11.5)
ax_c.set_ylim(-0.42, 0.5)
ax_c.set_xlabel("Month")
ax_c.set_ylabel(f"{style.KGE_PRIME} (-)")
ax_c.set_title(f"Seasonal {style.KGE_PRIME} by month (n = 74 gauges)", pad=4)
ax_c.legend(loc="lower right", fontsize=7.4, handlelength=1.5,
            handletextpad=0.5, ncol=2, columnspacing=1.2)
style.panel(ax_c, "c", x=-0.072, y=1.04)

out = style.savefig(fig, "fig14_decision_fdc")
print("saved:", out)
