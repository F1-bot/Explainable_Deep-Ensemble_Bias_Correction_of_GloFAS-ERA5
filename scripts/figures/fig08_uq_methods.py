"""Figure 08 - UQ head comparison (CMAL vs Gaussian mixture vs QRF).

Grouped-bar panels, one per probabilistic-skill metric, bars grouped by UQ head
and split (solid = temporal holdout, hatched = prediction in ungauged regions).

Honest reading (audit M15/M2/M16): the CMAL head is the *sharpest* and
out-calibrates the Gaussian-mixture flagship, but the QRF baseline achieves
better RAW coverage by issuing WIDER intervals (see the interval-width panel).
The pooled-over-timestep CRPS in panel (a) is dominated by the many in-sample
temporal points; under PUR it does NOT mean the flagship out-sharpens QRF --
per gauge (n=14) the flagship CRPS collapses (median 76.6 vs QRF 35.5), shown
in the reconciliation panel (g). Coverage panels (e) are RAW heads BEFORE the
conformal layer, which restores nominal ~0.90 (see Fig. on calibration).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sbc.viz import style

style.apply()
# Keep Arial as primary but enable per-glyph fallback to DejaVu Sans so the
# superscript-minus in style.UNIT_Q (m^3 s^-1) renders (Arial lacks U+207B).
import matplotlib as mpl
mpl.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

# --- data -------------------------------------------------------------------
CSV = Path(r"G:/MDPI Q1-2026/results/tables/cmal_uq_comparison_real_decadal.csv")
df = pd.read_csv(CSV)

# per-gauge CRPS for the PUR reconciliation panel (audit M2): the flagship's
# per-gauge CRPS collapses out-of-sample even though pooled CRPS does not.
PG = pd.read_csv(r"G:/MDPI Q1-2026/results/tables/per_gauge_canonical_decadal.csv")

MODELS = ["cmal", "regimeprobnet", "qrf"]          # CMAL, Gaussian mixture, QRF
SPLITS = ["temporal", "pur"]

# Fixed colour grammar (audit C3/C4): one CB-safe hue per head, legend == bars.
# regimeprobnet IS the Gaussian-mixture flagship -> flagship orange.
COLORS = {"cmal": style.SERIES_COLORS["cmal"],          # pink
          "regimeprobnet": style.SERIES_COLORS["flagship"],  # orange
          "qrf": style.SERIES_COLORS["qrf"]}            # green

MODEL_TICKS = {"cmal": "CMAL\nhead", "regimeprobnet": "Gaussian\nmixture",
               "qrf": "QRF"}

# in-panel sample sizes (pooled over timesteps)
N_BY_SPLIT = {sp: int(df[df.split == sp]["n"].iloc[0]) for sp in SPLITS}

# metric key, panel title, units, lower-is-better?, optional target line
METRICS = [
    ("crps",             "CRPS",              style.UNIT_Q, True,  None),
    ("twcrps_q90",       "twCRPS–Q90",   style.UNIT_Q, True,  None),
    ("winkler90",        "Winkler-90 score",  style.UNIT_Q, True,  None),
    ("width90",          "Mean 90% PI width", style.UNIT_Q, True,  None),
    ("cov90",            "Coverage-90",       "-",    None,  0.90),
    ("alpha_reliability","Alpha-reliability", "-",          False, None),
]

# --- layout -----------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(style.WIDTH_2COL, 6.4))
axes = axes.ravel()

x = np.arange(len(MODELS))
bw = 0.38
offs = {"temporal": -bw / 2, "pur": bw / 2}
HATCH = {"temporal": "", "pur": "////"}

for k, (key, title, unit, lower, target) in enumerate(METRICS):
    ax = axes[k]
    vals_all = []
    for sp in SPLITS:
        sub = df[df.split == sp].set_index("model")
        vals = [sub.loc[m, key] for m in MODELS]
        vals_all += vals
        ax.bar(x + offs[sp], vals, bw,
               color=[COLORS[m] for m in MODELS],
               edgecolor="white" if sp == "temporal" else "#222222",
               linewidth=0.6, hatch=HATCH[sp],
               alpha=1.0 if sp == "temporal" else 0.55, zorder=3)

    if target is not None:
        ax.axhline(target, color="#222222", ls="--", lw=1.1, zorder=4)
        ax.text(0.98, 0.97, f"nominal {target:.2f}", color="#222222", fontsize=6.5,
                va="top", ha="right", transform=ax.transAxes, zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec="#222222", lw=0.5, alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_TICKS[m] for m in MODELS])
    ax.set_ylabel(f"{title}\n({unit})" if unit != "-" else title, fontsize=8)

    # direction-of-merit tag
    if lower is True:
        merit = r"$\downarrow$ lower is better"
    elif lower is False:
        merit = r"$\uparrow$ higher is better"
    else:
        # coverage panel: the dashed-line box already labels the nominal target,
        # so the merit title says closeness-to-target (audit C: no double label).
        merit = r"closer to nominal $\rightarrow$ better"
    ax.set_title(merit, fontsize=7.5, color="#555555", pad=3)

    if key == "winkler90":
        ax.yaxis.set_major_formatter(style.thousands_formatter())

    # M1: zero-base every bar panel so truncation cannot exaggerate gaps.
    top = max(vals_all)
    if target is not None:
        top = max(top, target)
    ax.set_ylim(0, top * 1.20)
    ax.margins(x=0.06)

    style.panel(ax, "abcdef"[k])


# --- (g) per-gauge PUR CRPS reconciliation (audit M2) -----------------------
gax = axes[6]
pur_pg = PG[(PG.split == "pur") & (PG.model.isin(["regimeprobnet", "qrf"]))]
pg_models = ["regimeprobnet", "qrf"]
pg_x = {"regimeprobnet": 0, "qrf": 1}
rng = np.random.default_rng(0)
for m in pg_models:
    v = pur_pg[pur_pg.model == m]["crps"].to_numpy(float)
    v = v[np.isfinite(v)]
    jit = (rng.random(len(v)) - 0.5) * 0.22
    gax.scatter(np.full(len(v), pg_x[m]) + jit, v, s=16,
                color=COLORS[m], edgecolor="white", linewidth=0.4,
                alpha=0.9, zorder=3)
    med = float(np.median(v))
    gax.hlines(med, pg_x[m] - 0.28, pg_x[m] + 0.28, color="#222222",
               lw=1.6, zorder=4)
    gax.text(pg_x[m], gax.get_ylim()[1], "", )  # placeholder
    gax.annotate(f"median\n{med:.1f}", (pg_x[m], med), xytext=(0, 8),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color="#222222", zorder=6)
gax.set_xticks([0, 1])
gax.set_xticklabels(["Gaussian\nmixture", "QRF"])
gax.set_ylabel(f"Per-gauge CRPS\n({style.UNIT_Q})", fontsize=8)
gax.set_xlim(-0.6, 1.6)
gax.set_ylim(bottom=0)
gax.set_title("PUR, per gauge (n=14)", fontsize=7.5, color="#555555", pad=3)
style.panel(gax, "g")

# --- (h) legend -------------------------------------------------------------
lax = axes[7]
lax.axis("off")
LEG_LABELS = {"cmal": "CMAL head (sharpest)",
              "regimeprobnet": "Gaussian mixture\n(RegimeProbNet, flagship)",
              "qrf": "QRF (widest → best raw cov.)"}
model_handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=LEG_LABELS[m])
                 for m in MODELS]
split_handles = [
    Patch(facecolor="#888888", edgecolor="white", label="Temporal holdout"),
    Patch(facecolor="#888888", edgecolor="#222222", hatch="////", alpha=0.55,
          label="Prediction in ungauged regions (PUR)"),
]
leg1 = lax.legend(handles=model_handles, title="UQ head", loc="upper left",
                  bbox_to_anchor=(0.0, 1.02), handlelength=1.3,
                  borderaxespad=0.0, labelspacing=0.55, fontsize=7.5)
leg1.get_title().set_fontweight("bold")
lax.add_artist(leg1)
leg2 = lax.legend(handles=split_handles, title="Evaluation split",
                  loc="upper left", bbox_to_anchor=(0.0, 0.30),
                  handlelength=1.3, borderaxespad=0.0, labelspacing=0.55,
                  fontsize=7.5)
leg2.get_title().set_fontweight("bold")

# (i) reserved - caption / reconciliation text moved to the manuscript
cax = axes[8]
cax.axis("off")

fig.tight_layout(rect=(0, 0, 1, 0.99), w_pad=1.4, h_pad=2.2)

paths = style.savefig(fig, "fig08_uq_methods")
print("wrote", paths)
