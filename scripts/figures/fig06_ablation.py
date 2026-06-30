"""Figure 06 - Component ablation.

Grouped horizontal bar charts showing each component's marginal contribution.

Panel (a): delta KGE' = KGE(full) - KGE(ablated) for every ablation study,
grouped by evaluation split (temporal holdout, n=74, vs prediction in ungauged
regions, PUR, n=14). A positive value means skill is lost when the component is
removed (the component helps). Each study is tagged with its BASE model
(LightGBM-base or RegimeProbNet-base) because the "full" reference differs
between the two families - they are NOT a single model.

Honesty: the PUR deltas are single-run, n=14 point estimates. Their 95% bootstrap
confidence intervals (over the 14 ungauged gauges) are very wide and all cross 0;
the PUR component differences are not statistically separable (pairwise p>0.42).
The temporal deltas are precise but small.

Panel (b): the three physics-constraint variants, plotted as the median per-gauge
KGE' improvement over raw GloFAS, AND annotated with the aggregate KGE' so the
median does not hide the asymmetric-Laplace PUR collapse (aggregate KGE' 0.06 vs
raw 0.40). For every variant the RegimeProbNet PUR aggregate stays BELOW raw
GloFAS (0.40) - physics is a regulariser with a real PUR skill cost.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from sbc.config import PATHS
from sbc.viz import style

style.apply()

DELTA = "Δ"
KGEP = style.KGE_PRIME

# --- load -------------------------------------------------------------------
TAB = PATHS.tables
abl = pd.read_csv(TAB / "ablation_real_decadal.csv")
con = pd.read_csv(TAB / "constraint_ablation_real_decadal.csv")
pg = pd.read_csv(TAB / "per_gauge_real_decadal.csv")

# --- bootstrap CIs over gauges (single ablation run -> sampling uncertainty) -
# We have only aggregate, single-run ablation deltas. To quantify the
# (irreducible) gauge-sampling uncertainty we bootstrap the aggregate KGE' of
# the relevant BASE model over its n gauges and use the 95% CI half-width as a
# conservative band on the delta. The PUR band (n=14) is enormous; the temporal
# band (n=74) is tight.
def boot_halfwidth(model: str, split: str, n_boot: int = 5000, seed: int = 0) -> float:
    k = pg[(pg.model == model) & (pg.split == split)]["kge"].dropna().values
    if len(k) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(k, len(k), replace=True).mean() for _ in range(n_boot)])
    return float((np.percentile(means, 97.5) - np.percentile(means, 2.5)) / 2.0)

HW = {(m, s): boot_halfwidth(m, s) for m in ("lgbm", "regimeprobnet")
      for s in ("temporal", "pur")}

# --- split colours / labels -------------------------------------------------
SPLIT_COLOR = {"temporal": style.SPLIT_COLORS["temporal"],
               "pur": style.SPLIT_COLORS["pur"]}
SPLIT_NAME = {"temporal": "Temporal holdout (n = 74)",
              "pur": "Ungauged regions, PUR (n = 14)"}

# --- panel (a) data: grouped by BASE MODEL (M13), then by PUR contribution ---
STUDY_LABEL = {
    "residual_target": "Log-residual target\n(LightGBM base)",
    "snow_features": "Snow features\n(LightGBM base)",
    "static_attrs": "Static attributes\n(LightGBM base)",
    "regime_gating": "Regime gating\n(RegimeProbNet base)",
    "physics_penalty": "Physics penalty\n(RegimeProbNet base)",
}
# base model per study (drives which "full" reference + which bootstrap band)
STUDY_BASE = {
    "residual_target": "lgbm", "snow_features": "lgbm", "static_attrs": "lgbm",
    "regime_gating": "regimeprobnet", "physics_penalty": "regimeprobnet",
}
study_order = ["residual_target", "snow_features", "static_attrs",
               "regime_gating", "physics_penalty"]

a = abl.pivot_table(index="study", columns="split", values="delta_kge")
a = a.reindex(study_order)

# --- panel (b) data: physics-constraint variants ----------------------------
VAR_LABEL = {
    "probnet_alaplace": "Asymmetric-Laplace",
    "probnet_hardmono": "Hard-monotonic",
    "probnet_soft": "Soft penalty",
}
var_order = ["probnet_soft", "probnet_hardmono", "probnet_alaplace"]
b = con.pivot_table(index="model", columns="split", values="d_kge").reindex(var_order)
b_kge = con.pivot_table(index="model", columns="split", values="kge").reindex(var_order)
RAW_PUR = float(con["kge_raw"].iloc[0])  # raw GloFAS PUR aggregate KGE' (0.395)

# --- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(style.WIDTH_2COL, 4.05))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.50,
                      left=0.205, right=0.965, top=0.86, bottom=0.215)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

BARH = 0.38
SPLITS = ["temporal", "pur"]
ECAP = "#1a1a1a"


def add_bar(ax, v, y, split, hw):
    ax.barh(y, v, height=BARH, color=SPLIT_COLOR[split],
            edgecolor="white", linewidth=0.5, zorder=3)
    if hw is not None and np.isfinite(hw):
        ax.errorbar(v, y, xerr=hw, fmt="none", ecolor=ECAP, elinewidth=0.9,
                    capsize=2.2, capthick=0.9, zorder=4)
    return hw if (hw and np.isfinite(hw)) else 0.0


def value_label(ax, v, y, hw, fs=6.8):
    # place the numeric label just outside the error-bar cap, on the sign side
    if v >= 0:
        lblx, ha = v + hw + 0.030, "left"
        txt = "+" + style.num(v, 2)
    else:
        lblx, ha = v - hw - 0.030, "right"
        txt = style.num(v, 2)
    ax.text(lblx, y, txt, va="center", ha=ha, fontsize=fs, color="#1a1a1a",
            zorder=5)


# panel (a) ------------------------------------------------------------------
ypos = list(range(len(study_order)))
for i, key in enumerate(study_order):
    base = STUDY_BASE[key]
    for split in SPLITS:
        v = a.loc[key, split]
        y = ypos[i] + (BARH / 2 if split == "temporal" else -BARH / 2)
        hw = HW[(base, split)]
        add_bar(axA, v, y, split, hw)
        value_label(axA, v, y, hw)
axA.set_yticks(ypos)
axA.set_yticklabels([STUDY_LABEL[k] for k in study_order], fontsize=7)
axA.invert_yaxis()
axA.axvline(0, color="#333333", linewidth=0.9, zorder=2)
# separator between the two base-model groups (after the 3 LightGBM studies)
axA.axhline(2.5, color="#bbbbbb", linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
axA.grid(axis="y", visible=False)
axA.set_axisbelow(True)
axA.set_xlim(-0.55, 0.95)
axA.set_xlabel(f"{DELTA}{KGEP}  =  {KGEP}(full) {style.MINUS} {KGEP}(ablated)")
axA.set_title("Component ablation: skill change when a component is removed",
              pad=6, fontsize=8.5)
style.panel(axA, "a", x=-0.50, y=1.16)

axA.text(0.985, 0.03, f"positive {style.ARROW} component helps",
         transform=axA.transAxes, ha="right", va="bottom",
         fontsize=6.8, style="italic", color="#555555")

# panel (b) ------------------------------------------------------------------
ypos2 = list(range(len(var_order)))
for i, key in enumerate(var_order):
    for split in SPLITS:
        v = b.loc[key, split]
        y = ypos2[i] + (BARH / 2 if split == "temporal" else -BARH / 2)
        hw = HW[("regimeprobnet", split)]
        add_bar(axB, v, y, split, hw)
        # annotate with the AGGREGATE KGE' (median d_kge hides the ala collapse).
        # Always place to the RIGHT of the (wide PUR) error bar so the small /
        # negative PUR-bar labels never crowd the y-axis.
        kagg = b_kge.loc[key, split]
        axB.text(v + abs(hw) + 0.030, y, f"{KGEP}={style.num(kagg, 2)}",
                 va="center", ha="left", fontsize=6.4, color="#1a1a1a", zorder=5)
axB.set_yticks(ypos2)
axB.set_yticklabels([VAR_LABEL[k] for k in var_order], fontsize=7.5)
axB.invert_yaxis()
axB.axvline(0, color="#333333", linewidth=0.9, zorder=2)
axB.grid(axis="y", visible=False)
axB.set_axisbelow(True)
axB.set_xlim(-0.65, 1.30)
axB.set_xlabel(f"median {DELTA}{KGEP} vs raw")
axB.set_title("Physics-constraint variants\n(RegimeProbNet base)",
              pad=4, fontsize=8.5)
style.panel(axB, "b", x=-0.58, y=1.16)
# consistent x-tick format on both panels: 0.5 spacing, 1 decimal, U+2212 minus
from matplotlib.ticker import MultipleLocator, FuncFormatter
_xfmt = FuncFormatter(lambda x, _pos: style.num(x, 1) if abs(x) > 1e-9 else "0.0")
for _ax in (axA, axB):
    _ax.xaxis.set_major_locator(MultipleLocator(0.5))
    _ax.xaxis.set_major_formatter(_xfmt)

# --- shared split legend (bottom) -------------------------------------------
handles = [Patch(facecolor=SPLIT_COLOR[s], edgecolor="white", label=SPLIT_NAME[s])
           for s in SPLITS]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.01), columnspacing=1.6, handlelength=1.3,
           fontsize=8)

style.savefig(fig, "fig06_ablation")
print("HW:", {k: round(v, 3) for k, v in HW.items()})
print("done")
