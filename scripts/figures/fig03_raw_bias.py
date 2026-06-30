"""Figure 03 - Raw GloFAS-ERA5 bias characterization across 74 gauges.

Three panels from per_gauge_canonical_decadal.csv (split=temporal, model=lgbm
gives one row per gauge of the model-independent RAW values):
  (a) distribution of raw KGE-prime  (median ~0.39)
  (b) distribution of raw PBIAS      (median ~-13%, systematic under-bias)
  (c) KGE-prime decomposition (r, beta, gamma) vs the ideal value 1.

Message: GloFAS-ERA5 is systematically negatively biased with degraded
correlation in these snow-influenced headwaters.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sbc.viz import style

style.apply()

# --- data -------------------------------------------------------------------
CSV = r"G:/MDPI Q1-2026/results/tables/per_gauge_canonical_decadal.csv"
df = pd.read_csv(CSV)
d = df[(df["split"] == "temporal") & (df["model"] == "lgbm")].copy()
assert d["code"].nunique() == len(d), "expected one row per gauge"
n_gauges = len(d)

kge = d["kge_raw"].to_numpy()
pbias = d["pbias_raw"].to_numpy()
decomp = {
    "r": d["kge_r_raw"].to_numpy(),
    "beta": d["kge_beta_raw"].to_numpy(),
    "gamma": d["kge_gamma_raw"].to_numpy(),
}

C_HIST = "#0072B2"
C_MED = "#D55E00"

fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_2COL, 2.55))
fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.90, wspace=0.33)

# --- (a) raw KGE-prime ------------------------------------------------------
ax = axes[0]
med_kge = np.median(kge)
bins = np.linspace(-1.0, 1.0, 21)
kge_c = np.clip(kge, bins[0], bins[-1])
ax.hist(kge_c, bins=bins, color=C_HIST, alpha=0.55, edgecolor="white", linewidth=0.5)
# KDE overlay (Gaussian KDE on the un-clipped values), scaled density -> counts
kde = gaussian_kde(kge)
xs = np.linspace(-1.0, 1.0, 200)
width = bins[1] - bins[0]
ax.plot(xs, kde(xs) * len(kge) * width, color=C_HIST, linewidth=1.6)
ax.axvline(med_kge, color=C_MED, linewidth=1.6, linestyle="--")
ax.text(med_kge - 0.04, ax.get_ylim()[1] * 0.96, f"median = {style.num(med_kge, 2)}",
        color=C_MED, ha="right", va="top", fontsize=8, fontweight="bold")
ax.set_xlabel(f"Raw {style.KGE_PRIME} (-)")
ax.set_ylabel(f"Gauges (n = {n_gauges})")
ax.set_xlim(-1.0, 1.0)
ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
style.panel(ax, "a")

# --- (b) raw PBIAS ----------------------------------------------------------
ax = axes[1]
med_pb = np.median(pbias)
lo, hi = -80, 80
bins = np.linspace(lo, hi, 21)
pb_c = np.clip(pbias, lo, hi)
ax.hist(pb_c, bins=bins, color=C_HIST, alpha=0.55, edgecolor="white", linewidth=0.5)
ax.axvline(0.0, color="#444444", linewidth=1.1, linestyle="-")
ax.axvline(med_pb, color=C_MED, linewidth=1.6, linestyle="--")
ax.text(med_pb - 3, ax.get_ylim()[1] * 0.96, f"median = {style.num(med_pb, 1, pct=True)}",
        color=C_MED, ha="right", va="top", fontsize=8, fontweight="bold")
# shade the under-bias region for narrative
ax.axvspan(lo, 0, color="#D55E00", alpha=0.05, zorder=0)
ax.set_xlabel("Raw PBIAS (%)")
ax.set_ylabel(f"Gauges (n = {n_gauges})")
ax.set_xlim(lo, hi)
ax.set_xticks(np.arange(-80, 81, 40))
style.panel(ax, "b")

# --- (c) KGE' decomposition -------------------------------------------------
ax = axes[2]
keys = ["r", "beta", "gamma"]
labels = [r"$r$", r"$\beta$", r"$\gamma$"]
data = [decomp[k] for k in keys]
positions = [1, 2, 3]
bp = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False,
                patch_artist=True, medianprops=dict(color="#222222", linewidth=1.4),
                whiskerprops=dict(color="#555555", linewidth=1.0),
                capprops=dict(color="#555555", linewidth=1.0),
                boxprops=dict(linewidth=0.8, edgecolor="#555555"))
box_colors = ["#56B4E9", "#009E73", "#E69F00"]
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)
# jittered points
rng = np.random.default_rng(7)
for pos, arr in zip(positions, data):
    jx = pos + rng.uniform(-0.16, 0.16, size=len(arr))
    ax.scatter(jx, arr, s=6, color="#333333", alpha=0.35, linewidths=0, zorder=3)
ax.axhline(1.0, color=C_MED, linewidth=1.4, linestyle="--", zorder=2)
ax.text(0.6, 1.05, "ideal = 1", color=C_MED, ha="left", va="bottom",
        fontsize=8, fontweight="bold")
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_xlim(0.5, 3.5)
ax.set_ylim(0.0, 3.0)
ax.set_ylabel("Component value (-)")
ax.set_xlabel(f"{style.KGE_PRIME} component")
style.panel(ax, "c")

style.savefig(fig, "fig03_raw_bias")
print("saved; medians: kge=%.3f pbias=%.2f r=%.3f beta=%.3f gamma=%.3f" % (
    med_kge, med_pb, np.median(decomp["r"]),
    np.median(decomp["beta"]), np.median(decomp["gamma"])))
