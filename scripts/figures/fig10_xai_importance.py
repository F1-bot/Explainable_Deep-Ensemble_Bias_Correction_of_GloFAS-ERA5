"""Figure 10 - Explainability: drivers of the GloFAS bias correction.

Panel (a): top-12 global mean|SHAP| features of the deployed flagship, with
snow-physics and the land-surface soil-moisture driver highlighted against
GloFAS discharge memory and seasonality/static catchment terms (greys).
Whiskers show per-seed spread; the per-feature ORDER is seed-unstable
(Kendall tau~0.27) — only group-level attribution is stable (tau=1.0), so the
ranks 5-12 tail is de-emphasised.
Panel (b): regime-conditional, driver-level attribution as a TRUE per-feature
mean (family sum / n_features, n annotated per row) for the snow-physics
families plus the soil-moisture land-surface term, across the four hydrological
regimes. Soil moisture (swvl1) is a land-surface, not a snow-physics, driver.

Attribution is computed on the deployed flagship (RegimeProbNet), not a proxy.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from sbc.viz import style

style.apply()

TAB = r"G:/MDPI Q1-2026/results/tables"

# ---------------------------------------------------------------- colours -----
C_MEM = "#5B5B5B"     # GloFAS discharge memory (the anchor)
C_SNOW = "#56B4E9"    # snow-physics drivers (highlight)
C_LAND = "#009E73"    # land-surface drivers (soil moisture) — NOT snow physics
C_SEAS = "#B0B0B0"    # seasonality / static catchment terms
SNOW_CMAP = "Blues"

# ----------------------------------------------------- feature humanising -----
HUMAN = {
    "f_log_qglofas": "GloFAS discharge (log)",
    "f_log_qglofas_lag1": "GloFAS discharge, lag 1",
    "f_log_qglofas_lag2": "GloFAS discharge, lag 2",
    "f_log_qglofas_lag3": "GloFAS discharge, lag 3",
    "f_log_qglofas_roc": "GloFAS rate of change",
    "f_log_qglofas_rmean3": "GloFAS 3-step mean",
    "f_log_qglofas_rmean6": "GloFAS 6-step mean",
    "f_log_qglofas_rmean9": "GloFAS 9-step mean",
    "f_log_qglofas_rstd3": "GloFAS 3-step variability",
    "f_log_qglofas_rstd9": "GloFAS 9-step variability",
    "t2m_mean": "Air temperature (2 m)",
    "swvl1": "Soil moisture (top layer)",
    "swe": "Snow water equivalent",
    "smlt": "Snowmelt",
    "f_doy_cos": "Season (annual cycle)",
    "f_doy_sin": "Season (annual phase)",
    "f_decade_sin": "Decadal phase",
    "q_mm": "Catchment yield (mm)",
    "q_km3a": "Mean flow (km³ a⁻¹)",
    "q_m3s": f"Mean flow ({style.UNIT_Q})",
    "h_min": "Minimum elevation",
    "h_max": "Maximum elevation",
    "f_swe_lag3": "Snow water equiv., lag 3",
    "f_scf_cold": "Snow cover (cold season)",
    "f_scf_amp": "Snow cover (amplitude)",
    "regime_id": "Hydrological regime",
    "hurs_range": "Humidity range",
}


def humanise(f: str) -> str:
    return HUMAN.get(f, f.replace("_", " "))


# membership tests for the panel (a) three-way colouring
def is_memory(f: str) -> bool:
    return f.startswith("f_log_qglofas")


# snow-physics keywords — soil moisture (swvl) is deliberately EXCLUDED here and
# treated as a land-surface driver instead (audit M18).
SNOW_KW = ("t2m", "tas", "pdd", "freeze_thaw", "ft_crossings",
           "swe", "smlt", "melt", "scf", "snow")
LAND_KW = ("swvl",)   # soil moisture — land-surface, not snow physics


def is_snow(f: str) -> bool:
    return any(kw in f for kw in SNOW_KW)


def is_land(f: str) -> bool:
    return any(kw in f for kw in LAND_KW)


def category(f: str) -> str:
    if is_memory(f):
        return "memory"
    if is_land(f):
        return "land"
    if is_snow(f):
        return "snow"
    return "other"


CAT_COLOR = {"memory": C_MEM, "snow": C_SNOW, "land": C_LAND, "other": C_SEAS}

# driver families for panel (b): snow-physics families plus the soil-moisture
# land-surface term (kept for comparison, flagged separately — audit M18).
DRIVERS = {
    "Air temperature": ("t2m", "tas", "pdd", "freeze_thaw", "ft_crossings"),
    "Snow water equiv.": ("swe",),
    "Soil moisture": ("swvl",),
    "Snowmelt": ("smlt", "melt_season"),
    "Snow cover": ("scf",),
}
# which families are snow-physics vs land-surface (for the y-tick annotation)
DRIVER_KIND = {"Air temperature": "snow", "Snow water equiv.": "snow",
               "Soil moisture": "land", "Snowmelt": "snow", "Snow cover": "snow"}
REGIMES = ["accumulation", "melt_freshet", "rain_on_snow", "recession"]

# ============================================================= load data ======
imp = pd.read_csv(f"{TAB}/flagship_shap_importance.csv")
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
top = imp.head(12).copy()
top["cat"] = top["feature"].map(category)
top["label"] = top["feature"].map(humanise)

# Per-seed instability (audit M7): the multi-seed run reports a coefficient of
# variation (CV = sd / mean) per feature. CV is scale-free, so we map it onto the
# deployed-run mean|SHAP| to draw a per-seed spread band (mean × CV ≈ 1-sigma).
STAB = (f"{TAB}/shap_stability_regimeprobnet_table_finalrigor_dryrun_decadal.csv")
stab = pd.read_csv(STAB).set_index("feature")["cv"]
_cv_med = float(stab.median())
top["cv"] = top["feature"].map(stab).fillna(_cv_med)
top["err"] = top["mean_abs_shap"] * top["cv"]

# Headline seed-stability summary (from shap_stability_finalrigor_dryrun_decadal.csv)
TAU_FEATURE = 0.27   # Kendall tau, per-feature ranking (regimeprobnet, n=2 seeds)
TAU_GROUP = 1.0      # Kendall tau, group-level attribution (stable)

reg = pd.read_csv(f"{TAB}/flagship_shap_regime.csv")
reg = reg[reg["regime"].isin(REGIMES)].copy()


def driver_matrix():
    """Return (mean-per-feature matrix, per-family feature count).

    audit M18: the family aggregate must be a TRUE MEAN (sum / n_features), not a
    raw family SUM, because family sizes are unequal (Air temp n=11 vs Soil
    moisture n=1) and a sum rewards large families by construction.
    """
    M = np.zeros((len(DRIVERS), len(REGIMES)))
    counts = np.zeros(len(DRIVERS), dtype=int)
    for i, (_, kws) in enumerate(DRIVERS.items()):
        fam = reg[reg["feature"].apply(lambda f: any(k in f for k in kws))]
        counts[i] = fam["feature"].nunique()
        for j, rg in enumerate(REGIMES):
            s = fam[fam["regime"] == rg]["mean_abs_shap"]
            M[i, j] = s.sum() / max(counts[i], 1)   # true per-feature mean
    return M, counts


M, FAM_N = driver_matrix()

# ================================================================ figure =======
fig = plt.figure(figsize=(style.WIDTH_2COL, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.62,
                      left=0.005, right=0.985, top=0.88, bottom=0.16)

# ---------- panel (a): global top-12 horizontal bars --------------------------
axa = fig.add_subplot(gs[0, 0])
ypos = np.arange(len(top))[::-1]  # largest at top
colors = [CAT_COLOR[c] for c in top["cat"]]
# solid bars at full colour so every bar matches its legend swatch exactly
bars = axa.barh(ypos, top["mean_abs_shap"], color=colors, edgecolor="white",
                linewidth=0.5, height=0.74, zorder=3)
# per-seed spread bands (audit M7): horizontal error bars = mean × CV from re-seeding
axa.errorbar(top["mean_abs_shap"], ypos, xerr=top["err"], fmt="none",
             ecolor="#333333", elinewidth=0.8, capsize=2.0, capthick=0.8,
             alpha=0.85, zorder=4)
axa.set_yticks(ypos)
axa.set_yticklabels(top["label"])
axa.set_xlim(0, (top["mean_abs_shap"] + top["err"]).max() * 1.06)
axa.set_ylim(-0.7, len(top) - 1 + 0.6)
axa.set_xlabel(r"Global mean $|\mathrm{SHAP}|$ (log-discharge units)")
axa.tick_params(axis="y", length=0)
axa.grid(axis="y", visible=False)
axa.margins(y=0.012)

legend_a = [
    Patch(facecolor=C_MEM, label="GloFAS discharge memory"),
    Patch(facecolor=C_SNOW, label="Snow-physics driver"),
    Patch(facecolor=C_LAND, label="Land-surface (soil moisture)"),
    Patch(facecolor=C_SEAS, label="Seasonality / static"),
]
axa.legend(handles=legend_a, loc="lower right", fontsize=7.0,
           handlelength=1.1, handleheight=1.1, borderpad=0.5,
           labelspacing=0.4, frameon=True, framealpha=0.92,
           edgecolor="#CCCCCC")
style.panel(axa, "a", x=-0.58, y=1.03)

# ---------- panel (b): regime-conditional driver heatmap ----------------------
axb = fig.add_subplot(gs[0, 1])
vmax = M.max()
im = axb.imshow(M, aspect="auto", cmap=SNOW_CMAP, vmin=0, vmax=vmax)
axb.set_xticks(range(len(REGIMES)))
axb.set_xticklabels([style.REGIME_LABELS[r] for r in REGIMES],
                    rotation=30, ha="right")
# y labels carry per-family feature count n and the snow/land kind (audit M18)
ylabels = []
for name, n in zip(DRIVERS.keys(), FAM_N):
    tag = "land-surf." if DRIVER_KIND[name] == "land" else "snow"
    ylabels.append(f"{name}  (n={n}, {tag})")
axb.set_yticks(range(len(DRIVERS)))
axb.set_yticklabels(ylabels)
for lab, name in zip(axb.get_yticklabels(), DRIVERS.keys()):
    if DRIVER_KIND[name] == "land":
        lab.set_color(C_LAND)
axb.tick_params(length=0)
axb.grid(False)
# minor gridlines as cell separators
axb.set_xticks(np.arange(-0.5, len(REGIMES), 1), minor=True)
axb.set_yticks(np.arange(-0.5, len(DRIVERS), 1), minor=True)
axb.grid(which="minor", color="white", linewidth=1.4)
for sp in axb.spines.values():
    sp.set_visible(False)

# annotate cells — high-contrast text (audit M18): white on dark cells, near-black
# on light cells, switching at ~60% of the Blues ramp.
thr = vmax * 0.60
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        axb.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                 fontsize=6.9, fontweight="medium",
                 color="white" if M[i, j] > thr else "#1A1A1A")

cbar = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.04)
# audit M18: this is a TRUE per-feature mean (family sum / n_features), not a sum
cbar.set_label(r"Per-feature mean $|\mathrm{SHAP}|$" "\n(family sum ÷ n)",
               fontsize=7.4)
cbar.ax.tick_params(labelsize=7, length=2)
cbar.outline.set_visible(False)
style.panel(axb, "b", x=-0.42, y=1.03)

out = style.savefig(fig, "fig10_xai_importance")
print("WROTE", out)
