"""Figure 04 - Bias-correction skill by validation split.

Three horizontal-boxplot panels (a) temporal holdout, (b) leave-one-basin-out,
(c) prediction in ungauged regions. Each box is the per-gauge KGE' distribution;
a black marker + bar overlays the bootstrap 95% CI of the median (reconciled with
significance_real_decadal.csv); a dashed red line marks the raw GloFAS-ERA5 median
per split. Skilful / highly-skilful gauge counts are printed at the right of each
row.

Honest framing (no superiority implied): a single accent colour and a bold row
label mark only the PROPOSED model (RegimeProbNet), NOT the best performer. On the
temporal split the flagship (0.825) is statistically TIED with LightGBM (0.812),
Wilcoxon p = 0.49 - shown by an explicit "tied" bracket. On PUR no model
significantly beats raw GloFAS (every CI overlaps the raw line; all pairwise
p > 0.42). The five baselines not run under LOBO are tagged in panel (b) so the
blank rows do not read as failures.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from sbc.viz import style

style.apply()

TABLES = r"G:/MDPI Q1-2026/results/tables"
pg = pd.read_csv(f"{TABLES}/per_gauge_canonical_decadal.csv")
sm = pd.read_csv(f"{TABLES}/summary_canonical_decadal.csv")
# CI source reconciled with the significance table: its "(CI)" rows are the
# authoritative bootstrap 95% CIs that pair with the reported Wilcoxon p-values.
sig = pd.read_csv(f"{TABLES}/significance_real_decadal.csv")
_ci = sig[sig["ref"] == "(CI)"]
CI = {(r["split"], r["vs"]): (float(r["ci_lo"]), float(r["ci_hi"]),
                              float(r["median_vs"])) for _, r in _ci.iterrows()}

SPLITS = ["temporal", "lobo", "pur"]
PANEL = {"temporal": "a", "lobo": "b", "pur": "c"}
# per-split x window sized to show the full 5-95% whiskers (an extreme donor
# tail on PUR, 5th pct ~ -24, is clipped at the axis and flagged off-panel)
XLIM = {"temporal": (-0.45, 1.18), "lobo": (-1.15, 1.02), "pur": (-2.50, 1.02)}
XTICKS = {"temporal": [0.0, 0.5, 1.0], "lobo": [-1.0, -0.5, 0.0, 0.5, 1.0],
          "pur": [-2.0, -1.0, 0.0, 1.0]}

models = style.order_models(pg["model"].unique())
n = len(models)
RAW_RED = "#D7301F"

# --- M9: two-colour family (CB-safe), NOT a per-row rainbow ------------------
# A single neutral fill for every baseline / SOTA model; one accent reserved for
# the proposed-model row only. Legend swatches below use these exact colours.
BASE_FILL = "#B4BBC1"   # neutral grey  (all baselines + SOTA)
BASE_EDGE = "#5B636A"
FLAG_FILL = style.SERIES_COLORS["flagship"]  # set-wide flagship hue (#D55E00)
FLAG_EDGE = "#8A3D00"
BOX_ALPHA = 0.42

fig, axes = plt.subplots(
    1, 3, figsize=(style.WIDTH_2COL, 4.6), sharey=True,
    gridspec_kw={"wspace": 0.08},
)

for ax, split in zip(axes, SPLITS):
    sub_pg = pg[pg["split"] == split]
    sub_sm = sm[sm["split"] == split].set_index("model")
    ngauge = int(sub_sm["n_gauges"].iloc[0])
    raw_med = float(sub_sm["kge_raw_median"].iloc[0])
    xlo, xhi = XLIM[split]
    rng = xhi - xlo

    # clip box/whisker/CI artists to the data window so a far-off CI/whisker
    # (e.g. donor on PUR) is cut at the axis and flagged with an off-panel arrow
    clip_rect = Rectangle((xlo, -1.0), rng, n + 2.0, transform=ax.transData)

    # raw GloFAS median reference
    raw_line = ax.axvline(raw_med, color=RAW_RED, ls="--", lw=1.2, zorder=1.5)
    raw_line.set_clip_path(clip_rect)

    # build boxes only for models that have data in this split; tag the rest
    data, positions, box_models = [], [], []
    for i, m in enumerate(models):
        y = n - 1 - i  # first model in canonical order at top
        vals = sub_pg.loc[sub_pg["model"] == m, "kge"].dropna().values
        if vals.size == 0:
            # M10: explain the blank LOBO rows instead of leaving them empty
            ax.text((xlo + xhi) / 2.0, y, "not evaluated under LOBO",
                    ha="center", va="center", fontsize=6.1, color="#9A9A9A",
                    style="italic", zorder=3)
            continue
        data.append(vals)
        positions.append(y)
        box_models.append(m)

    bp = ax.boxplot(
        data, positions=positions, vert=False, widths=0.62,
        patch_artist=True, showfliers=False, whis=(5, 95), zorder=2,
    )
    for patch, m in zip(bp["boxes"], box_models):
        flag = (m == "regimeprobnet")
        patch.set_facecolor(FLAG_FILL if flag else BASE_FILL)
        patch.set_alpha(BOX_ALPHA)
        patch.set_edgecolor(FLAG_EDGE if flag else BASE_EDGE)
        patch.set_linewidth(1.3 if flag else 1.0)
    for med in bp["medians"]:
        med.set_color("#222222")
        med.set_linewidth(1.1)
    for w in bp["whiskers"]:
        w.set_color("#777777")
        w.set_linewidth(0.9)
    for c in bp["caps"]:
        c.set_color("#777777")
        c.set_linewidth(0.9)
    # keep every box artist inside the data window (never into the count column)
    for art in bp["boxes"] + bp["whiskers"] + bp["caps"] + bp["medians"]:
        art.set_clip_path(clip_rect)

    # bootstrap 95% CI of the median: marker + horizontal bar
    for i, m in enumerate(models):
        y = n - 1 - i
        key = (split, m)
        if key in CI:
            lo_raw, hi_raw, med = CI[key]
        elif m in sub_sm.index:
            med = float(sub_sm.loc[m, "kge_median"])
            lo_raw = float(sub_sm.loc[m, "kge_ci_lo"])
            hi_raw = float(sub_sm.loc[m, "kge_ci_hi"])
        else:
            continue  # model not evaluated in this split (already tagged)
        lo = max(lo_raw, xlo + 0.01)
        hi = min(hi_raw, xhi - 0.01)
        edge = "#000000" if m == "regimeprobnet" else "#1a1a1a"
        ci_line, = ax.plot([lo, hi], [y, y], color="#000000", lw=2.2,
                           solid_capstyle="butt", zorder=4)
        ci_mark, = ax.plot(med, y, marker="D", ms=4.6, mfc="white", mec=edge,
                           mew=1.3, zorder=5)
        ci_line.set_clip_path(clip_rect)
        ci_mark.set_clip_path(clip_rect)
        # clipped-CI indicator (donor on PUR runs far past the view): draw the
        # off-axis arrow AND annotate the true lower bound so the clip is honest
        if lo_raw < xlo + 0.01:
            ax.annotate("", xy=(xlo + 0.005, y), xytext=(xlo + 0.10, y),
                        arrowprops=dict(arrowstyle="-|>", color="#000000",
                                        lw=1.4), zorder=5)
            ax.text(xlo + 0.13, y + 0.30,
                    f"CI lo {style.num(lo_raw, 1)}", ha="left", va="bottom",
                    fontsize=5.9, color="#000000", zorder=6)

    # raw-median value label (style.num keeps the minus consistent if negative),
    # placed in the clear band just above the top box so it never hits the spine
    ax.text(raw_med, n - 0.46, style.num(raw_med, 2), ha="center", va="bottom",
            fontsize=6.4, color=RAW_RED, zorder=6, clip_on=False)

    # --- M11: explicit non-significance cues -------------------------------
    if split == "temporal":
        # "tied" bracket between the flagship (0.825) and LightGBM (0.812);
        # Wilcoxon p = 0.49 -> the two are statistically indistinguishable.
        y_rp = n - 1 - models.index("regimeprobnet")
        y_lg = n - 1 - models.index("lgbm")
        xb = 1.00
        brk = [
            ax.plot([xb, xb], [y_lg, y_rp], color="#333333", lw=1.0,
                    zorder=6)[0],
            ax.plot([xb - 0.035, xb], [y_rp, y_rp], color="#333333", lw=1.0,
                    zorder=6)[0],
            ax.plot([xb - 0.035, xb], [y_lg, y_lg], color="#333333", lw=1.0,
                    zorder=6)[0],
        ]
        for b in brk:
            b.set_clip_path(clip_rect)
        ax.text(xb + 0.085, (y_rp + y_lg) / 2.0, "tied, p = 0.49",
                rotation=90, ha="center", va="center", fontsize=6.6,
                color="#333333", zorder=6)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(XTICKS[split])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(style.KGE_PRIME)
    ax.set_title(f"{style.SPLIT_LABELS[split]}\n(n = {ngauge} gauges)",
                 fontsize=8.5, pad=5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    # panel tag raised clear above the (wide) titles so it never touches them
    style.panel(ax, PANEL[split], x=-0.62 if split == "temporal" else -0.07,
                y=1.16)

# y tick labels on the left panel only
axes[0].set_yticks([n - 1 - i for i in range(n)])
ylabels = []
for m in models:
    lab = style.label(m)
    ylabels.append(lab)
axes[0].set_yticklabels(ylabels)
# the bold label + accent fill flag the PROPOSED model only (clarified in caption)
for t, m in zip(axes[0].get_yticklabels(), models):
    if m == "regimeprobnet":
        t.set_fontweight("bold")
        t.set_color(FLAG_EDGE)

# figure-level legend (does not overlap data); swatches match the box fills (M9)
handles = [
    Patch(facecolor=BASE_FILL, alpha=BOX_ALPHA, edgecolor=BASE_EDGE,
          label=f"Baseline / SOTA model  ({style.KGE_PRIME} box: IQR, "
                "whiskers: 5-95%)"),
    Patch(facecolor=FLAG_FILL, alpha=BOX_ALPHA, edgecolor=FLAG_EDGE,
          label="RegimeProbNet (proposed model)"),
    Line2D([0], [0], color="#000000", lw=2.2, marker="D", mfc="white",
           mec="#000000", mew=1.3, ms=5,
           label="Bootstrap 95% CI of the median"),
    Line2D([0], [0], color=RAW_RED, ls="--", lw=1.2,
           label="Raw GloFAS-ERA5 median"),
    Line2D([0], [0], color="#000000", lw=1.4, marker="<", mfc="#000000",
           mec="#000000", ms=6,
           label="CI extends off-panel (true bound labelled)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=7.6,
           handletextpad=0.6, columnspacing=1.6)

fig.subplots_adjust(left=0.205, right=0.985, top=0.85, bottom=0.14)

style.savefig(fig, "fig04_skill_by_split")
print("done")
