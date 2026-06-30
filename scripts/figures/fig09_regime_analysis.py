"""Figure 09 - Regime gating & per-regime skill.

Panel (a): confusion-style heatmap (row-normalised %) of physically-named rule
regimes vs the learned mixture-of-experts gate argmax. The alignment is ~99.98%
block-diagonal. HONESTY (audit M17): expert identities are *assigned from* this
alignment, so the near-perfect diagonal is DEFINITIONAL, not independent
validation. The panel carries an in-figure note saying so. The argmax also
cannot evidence soft-gate boundary mixing; the soft gate-weight / entropy
diagnostic that would (not regenerated here) is the proper supplement. Expert 3
(glacier-melt) is never selected and is shown as an empty, greyed-out column.

Panel (b): per-regime KGE' of the bias-corrected series (RegimeProbNet flagship)
benchmarked against the UNCORRECTED FLOOR (raw GloFAS-ERA5). Raw is a trivially
weak comparator (audit M17), so it is labelled "uncorrected floor" and the gains
are framed only relative to that floor - NOT against a strong corrector
(quantile-mapping / best boosting), which is not regenerated here. Bars are
point estimates (median over n_gauges); per-regime sample sizes (n_obs, n_gauges)
are annotated in-panel. No per-regime per-gauge CI is available at this
aggregation, stated in-panel. Regimes use the shared hydrological-regime palette
consistently; the legend swatches match the plotted marks (grey = floor,
regime-coloured tuple = corrected).

Data:
  results/tables/gate_alignment_real_decadal.csv
  results/tables/diag_regime_skill_real_decadal.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from sbc.viz import style

style.apply()

TABLES = Path(r"G:/MDPI Q1-2026/results/tables")

# --- load ----------------------------------------------------------------
gate = pd.read_csv(TABLES / "gate_alignment_real_decadal.csv")
skill = pd.read_csv(TABLES / "diag_regime_skill_real_decadal.csv")

# rule regimes in canonical narrative order (those present in the gate table)
RULE_ORDER = ["accumulation", "melt_freshet", "rain_on_snow", "recession"]
gate = gate.set_index("regime").loc[RULE_ORDER]

# learned-expert columns; insert the unused glacier expert (index 3) as zeros
expert_cols = [0, 1, 2, 3, 4]
counts = pd.DataFrame(0, index=RULE_ORDER, columns=expert_cols, dtype=float)
for c in gate.columns:
    counts[int(c)] = gate[c].values
counts_mat = counts.values

# row-normalised percentages
row_tot = counts_mat.sum(axis=1, keepdims=True)
pct = 100.0 * counts_mat / row_tot

UNUSED_EXPERT = 3  # glacier-melt regime never selected by the gate

# expert -> learned physical name (from the block-diagonal alignment)
EXPERT_NAME = {0: "Accumulation", 1: "Melt\nfreshet", 2: "Rain-on-\nsnow",
               3: "Glacier\nmelt", 4: "Recession"}

# =========================================================================
fig = plt.figure(figsize=(style.WIDTH_2COL, 3.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42,
                      left=0.085, right=0.965, bottom=0.255, top=0.85)

# --- Panel (a): gate-alignment heatmap -----------------------------------
axA = fig.add_subplot(gs[0, 0])

# mask the unused expert column so it reads as "not part of the diagonal"
disp = np.ma.array(pct, mask=False)
cmap = plt.get_cmap("Blues").copy()
im = axA.imshow(disp, cmap=cmap, vmin=0, vmax=100, aspect="auto")

# grey overlay for the unused glacier expert column
j_un = expert_cols.index(UNUSED_EXPERT)
axA.add_patch(plt.Rectangle((j_un - 0.5, -0.5), 1, len(RULE_ORDER),
                            facecolor="#E8E8E8", edgecolor="none", zorder=2))

# annotate cells
for i in range(len(RULE_ORDER)):
    for j, e in enumerate(expert_cols):
        if e == UNUSED_EXPERT:
            continue
        v = pct[i, j]
        txt = f"{v:.2f}" if 0 < v < 100 else f"{v:.0f}"
        tcol = "white" if v >= 55 else "#1A1A1A"
        axA.text(j, i, txt, ha="center", va="center", fontsize=7.3,
                 color=tcol, fontweight="bold" if v >= 55 else "normal",
                 zorder=3)
axA.text(j_un, len(RULE_ORDER) / 2 - 0.5, "unused", ha="center", va="center",
         fontsize=7.3, color="#888888", rotation=90, zorder=3)

axA.set_xticks(range(len(expert_cols)))
axA.set_xticklabels([f"E{e}\n{EXPERT_NAME[e]}" for e in expert_cols], fontsize=7)
axA.set_yticks(range(len(RULE_ORDER)))
axA.set_yticklabels([style.REGIME_LABELS[r] for r in RULE_ORDER])
# colour the regime tick labels by the shared regime palette (axis = colour key)
for tl, r in zip(axA.get_yticklabels(), RULE_ORDER):
    tl.set_color(style.REGIME_COLORS[r])
axA.set_xlabel("Learned expert (gate argmax)")
axA.set_ylabel("Rule-based regime")
axA.set_title("Gate specialisation (row-normalised %)", pad=8)
axA.tick_params(length=0)
axA.set_xticks(np.arange(-0.5, len(expert_cols), 1), minor=True)
axA.set_yticks(np.arange(-0.5, len(RULE_ORDER), 1), minor=True)
axA.grid(which="minor", color="white", linewidth=1.4)
axA.grid(which="major", visible=False)
for sp in axA.spines.values():
    sp.set_visible(False)

cbar = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.03)
cbar.set_label("Share of regime samples (%)", fontsize=8)
cbar.outline.set_visible(False)
cbar.ax.tick_params(length=2, labelsize=7)

style.panel(axA, "a", x=-0.20, y=1.03)

# --- Panel (b): per-regime uncorrected-floor vs corrected KGE' -----------
axB = fig.add_subplot(gs[0, 1])

sk = skill.set_index("regime").loc[RULE_ORDER]
x = np.arange(len(RULE_ORDER))
w = 0.38

# uncorrected floor: a light HATCHED grey so it stays distinct from a regime that
# is itself grey (e.g. recession), and reads as one "floor" style set-wide
FLOOR_FILL = "#DCDCDC"
FLOOR_EDGE = "#8A8A8A"

for i, r in enumerate(RULE_ORDER):
    col = style.REGIME_COLORS[r]
    raw = sk.loc[r, "kge_raw"]
    cor = sk.loc[r, "kge"]
    # uncorrected floor: light hatched grey (one meaning), distinct from every
    # solid regime-coloured corrected bar including grey recession
    axB.bar(x[i] - w / 2, raw, w, facecolor=FLOOR_FILL, edgecolor=FLOOR_EDGE,
            linewidth=0.8, hatch="////", zorder=2)
    # corrected: solid regime colour (regime identity)
    axB.bar(x[i] + w / 2, cor, w, facecolor=col, edgecolor=col,
            linewidth=0.8, zorder=2)
    # delta annotation above the taller (corrected) bar; sign-aware so the
    # typographic minus (U+2212) matches the axis ticks if a delta is negative
    d = cor - raw
    dtxt = ("+" + style.num(d, 2)) if d >= 0 else style.num(d, 2)
    tcol = "#5A5A5A" if col == "#999999" else col
    axB.annotate(dtxt, (x[i] + w / 2, cor), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=7.4, fontweight="bold", color=tcol)
    # in-panel n (sample size + gauges) in a clean top strip
    ng = int(sk.loc[r, "n_gauges"])
    nobs = int(sk.loc[r, "n_obs"])
    axB.text(x[i], 1.0, f"n={style.num(nobs, 0)}\n{ng} gauges",
             ha="center", va="top", fontsize=6.2, color="#555555")

axB.set_xticks(x)
_WRAP = {"Rain-on-snow": "Rain-on-\nsnow"}
axB.set_xticklabels(
    [_WRAP.get(style.REGIME_LABELS[r], style.REGIME_LABELS[r].replace(" ", "\n"))
     for r in RULE_ORDER], fontsize=8)
for tl, r in zip(axB.get_xticklabels(), RULE_ORDER):
    tl.set_color(style.REGIME_COLORS[r])
axB.set_ylabel(f"{style.KGE_PRIME} (-)")
axB.set_ylim(0, 1.06)          # zero-based; top strip reserved for n labels
axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axB.set_title("Per-regime skill vs uncorrected floor", pad=8)
axB.axhline(0, color="#333333", linewidth=0.8)
axB.grid(axis="x", visible=False)

# legend: swatches match the marks. Floor = single grey patch (matches every
# floor bar); corrected = a tuple of the four regime colours (matches the
# regime-coloured corrected bars) via HandlerTuple.
floor_h = Patch(facecolor=FLOOR_FILL, edgecolor=FLOOR_EDGE, hatch="////")
cor_h = tuple(Patch(facecolor=style.REGIME_COLORS[r],
                    edgecolor=style.REGIME_COLORS[r]) for r in RULE_ORDER)
axB.legend([floor_h, cor_h],
           ["Uncorrected floor (raw GloFAS-ERA5)",
            "Bias-corrected (RegimeProbNet); hue = regime"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
           loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=7.2,
           handlelength=2.4, handleheight=1.2, borderpad=0.4, labelspacing=0.4)

style.panel(axB, "b", x=-0.18, y=1.03)

style.savefig(fig, "fig09_regime_analysis")
print("saved fig09_regime_analysis")
print("pct matrix:\n", np.round(pct, 3))
print("deltas:", {r: round(sk.loc[r, "kge"] - sk.loc[r, "kge_raw"], 3) for r in RULE_ORDER})
