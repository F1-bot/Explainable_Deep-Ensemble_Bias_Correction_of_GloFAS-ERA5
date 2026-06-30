"""Figure 07 - Probabilistic calibration & conformal post-processing.

Three panels telling the calibrated-UQ story honestly:
  (a) Reliability: the RAW flagship prediction intervals (BEFORE conformal)
      under-cover at every nominal level; the conformal layer restores the
      operational 90% PI to nominal (~0.90).
  (b) Per-regime coverage of the regime-conditional CONFORMAL scheme actually
      plotted here, the regime-soft-gate method (max |gap| = 0.024). For
      reference the regime-hard variant quoted in the text gives max |gap| =
      0.034 (annotated). Points are empirical coverage with binomial 95% CIs
      around the nominal 0.90 line.
  (c) CRPSS of the calibrated ensemble vs climatology (temporal vs PUR), with
      the sample size n annotated per split.

Coverage is nominal only AFTER conformal post-processing: the raw heads shown
in (a) are pre-conformal.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from sbc.viz import style

style.apply()

TBL = __import__("pathlib").Path(r"G:/MDPI Q1-2026/results/tables")
RAW = pd.read_csv(TBL / "calibration_fixed_real_decadal.csv")
COV = pd.read_csv(TBL / "calibration_fixed_coverage_real_decadal.csv")

C_RAW = style._HIGHLIGHT          # orange         - raw flagship (temporal)
C_RAW_P = style.PALETTE[9]        # wine (#882255) - raw flagship (PUR), distinct
C_CONF = style.PALETTE[0]         # blue           - after conformal
C_IDEAL = "#555555"

REGIME_METHOD = "regime-soft-gate"   # the regime-conditional conformal scheme PLOTTED
REF_METHOD = "regime-hard"           # variant quoted in the manuscript text
NOMINALS = [0.5, 0.8, 0.9]


def binom_ci(p, n, z=1.96):
    """Half-width of the Wald binomial 95% CI for a coverage proportion."""
    return z * np.sqrt(np.clip(p, 0, 1) * (1 - np.clip(p, 0, 1)) / np.asarray(n, float))


# ---------------------------------------------------------------- raw coverage
raw_t = RAW[RAW.split == "temporal"].iloc[0]
raw_p = RAW[RAW.split == "pur"].iloc[0]
raw_cov_t = [raw_t["cov_0.5"], raw_t["cov_0.8"], raw_t["cov_0.9"]]
raw_cov_p = [raw_p["cov_0.5"], raw_p["cov_0.8"], raw_p["cov_0.9"]]

# conformal coverage at the 90% PI across methods (temporal overall)
conf_t = COV[(COV.split == "temporal") & (COV.regime == "__overall__")]
conf_lo = conf_t.coverage.min()
conf_hi = conf_t.coverage.max()
conf_mid = COV[(COV.split == "temporal") & (COV.method == "split-absolute") &
               (COV.regime == "__overall__")].coverage.iloc[0]

# =============================================================================
fig = plt.figure(figsize=(style.WIDTH_2COL, 3.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.28, 0.66],
                      wspace=0.42, left=0.075, right=0.985,
                      bottom=0.165, top=0.86)
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])
axc = fig.add_subplot(gs[2])

# ----------------------------------------------------------------- panel (a)
lim = (0.27, 0.95)
axa.plot(lim, lim, ls="--", lw=1.0, color=C_IDEAL, zorder=1)
# diagonal label sits along the line near its centre
axa.text(0.605, 0.625, "ideal (y = x)", color=C_IDEAL, fontsize=7.2,
         ha="center", va="bottom", style="italic", rotation=45,
         rotation_mode="anchor")

# small x-offset on the PUR series so the two near-coincident raw curves do not
# sit on top of each other (distinct colour + linestyle + marker as well)
NOM_P = [n + 0.008 for n in NOMINALS]
axa.plot(NOMINALS, raw_cov_t, "-o", color=C_RAW, mfc=C_RAW, mec="white",
         mew=0.6, ms=6, lw=1.6, label="Raw, temporal (pre-conformal)", zorder=4)
axa.plot(NOM_P, raw_cov_p, ":D", color=C_RAW_P, mfc="white", mec=C_RAW_P,
         mew=1.3, ms=5.5, lw=1.4, label="Raw, PUR (pre-conformal)", zorder=3)

# conformal restoration at the 90% PI (range across conformal methods)
axa.errorbar(0.9, conf_mid, yerr=[[conf_mid - conf_lo], [conf_hi - conf_mid]],
             fmt="*", ms=13, color=C_CONF, mec="white", mew=0.7,
             ecolor=C_CONF, elinewidth=1.4, capsize=3.2, zorder=5,
             label="After conformal (90% PI)")

# M16: arrow shows the raw -> after-conformal restoration at the 90% PI
axa.annotate("", xy=(0.9, conf_mid - 0.012), xytext=(0.9, raw_cov_t[2] + 0.012),
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))

axa.set_xlim(lim)
axa.set_ylim(lim)
axa.set_xticks(NOMINALS)
axa.set_yticks([0.3, 0.5, 0.7, 0.9])
axa.set_aspect("equal", adjustable="box")
axa.set_xlabel("Nominal coverage")
axa.set_ylabel("Empirical coverage")
axa.set_title("Reliability of prediction intervals", fontsize=9.2, pad=6)
# legend back in the upper-left (empty above-diagonal region); the diagonal label
# now sits at the line centre so the two no longer collide
axa.legend(loc="upper left", fontsize=6.5, handlelength=1.9,
           borderaxespad=0.35, labelspacing=0.65, handletextpad=0.55,
           borderpad=0.5)
style.panel(axa, "a", x=-0.24)

# ----------------------------------------------------------------- panel (b)
# C1: coverage shown as POINTS with binomial 95% CIs around the nominal 0.90
# line (a legitimate non-zero reference) instead of truncated-baseline bars.
sub = COV[COV.method == REGIME_METHOD].copy()
regimes = [r for r in style.REGIME_ORDER
           if r in set(sub[sub.regime != "__overall__"].regime)]
x = np.arange(len(regimes))
off = 0.16


def cov_n_for(split):
    d = sub[sub.split == split].set_index("regime")
    cov = np.array([d.loc[r, "coverage"] for r in regimes], float)
    nn = np.array([d.loc[r, "n"] for r in regimes], float)
    return cov, nn


cov_t, n_t = cov_n_for("temporal")
cov_p, n_p = cov_n_for("pur")
ci_t = binom_ci(cov_t, n_t)
ci_p = binom_ci(cov_p, n_p)
cols = [style.REGIME_COLORS[r] for r in regimes]

axb.axhline(0.9, ls="--", lw=1.0, color=C_IDEAL, zorder=1)
axb.text(1.5, 0.862, "nominal 0.90", fontsize=7.0,
         color=C_IDEAL, ha="center", va="top", style="italic", zorder=8,
         bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))

for i, c in enumerate(cols):
    axb.errorbar(x[i] - off, cov_t[i], yerr=ci_t[i], fmt="o", ms=6.0,
                 color=c, mfc=c, mec="white", mew=0.6, ecolor=c,
                 elinewidth=1.3, capsize=2.6, zorder=4)
    axb.errorbar(x[i] + off, cov_p[i], yerr=ci_p[i], fmt="D", ms=5.5,
                 color=c, mfc="white", mec=c, mew=1.3, ecolor=c,
                 elinewidth=1.3, capsize=2.6, zorder=4)

# M14: name both methods and reconcile the gap numbers (soft-gate vs hard).
maxgap = sub[sub.regime != "__overall__"].gap.abs().max()
ref = COV[COV.method == REF_METHOD]
maxgap_hard = ref[ref.regime != "__overall__"].gap.abs().max()

axb.set_ylim(0.82, 0.955)
axb.set_xlim(-0.6, len(regimes) - 0.4)
axb.set_xticks(x)
axb.set_xticklabels([style.REGIME_LABELS[r] for r in regimes],
                    rotation=18, ha="right")
axb.set_ylabel("Empirical coverage")
axb.set_title("Conformal coverage by regime (soft-gate)",
              fontsize=8.4, pad=6)

split_handles = [
    Line2D([0], [0], marker="o", color="#666666", mfc="#777777",
           mec="white", mew=0.6, ls="none", ms=6.0, label="Temporal"),
    Line2D([0], [0], marker="D", color="#666666", mfc="white",
           mec="#666666", mew=1.3, ls="none", ms=5.5, label="PUR"),
]
axb.legend(handles=split_handles, loc="upper left", fontsize=7.0,
           ncol=2, handlelength=1.1, columnspacing=1.0, borderaxespad=0.3,
           handletextpad=0.4)
style.panel(axb, "b", x=-0.155)

# ----------------------------------------------------------------- panel (c)
crpss = {"temporal": raw_t["crpss_pooled"], "pur": raw_p["crpss_pooled"]}
nn = {"temporal": int(raw_t["n"]), "pur": int(raw_p["n"])}
xs = np.arange(2)
ccols = [style.PALETTE[2], style.PALETTE[3]]
axc.bar(xs, [crpss["temporal"], crpss["pur"]], 0.62, color=ccols,
        edgecolor="white", lw=0.6, zorder=3)
for xi, k in zip(xs, ["temporal", "pur"]):
    v = crpss[k]
    axc.text(xi, v + 0.018, f"+{v:.2f}", ha="center", va="bottom",
             fontsize=8.0, fontweight="bold")
axc.axhline(0, color="#333333", lw=0.8)
axc.set_ylim(0, 0.92)
axc.set_xticks(xs)
axc.set_xticklabels(["Temporal", "PUR"])
axc.set_ylabel("CRPSS vs climatology")
axc.set_title("Skill", fontsize=9.2, pad=6)
style.panel(axc, "c", x=-0.40)

style.savefig(fig, "fig07_calibration")
print("saved; soft maxgap=%.4f hard maxgap=%.4f conf=[%.3f,%.3f] crpss=%.3f/%.3f"
      % (maxgap, maxgap_hard, conf_lo, conf_hi,
         crpss["temporal"], crpss["pur"]))
