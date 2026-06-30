"""Figure 11 - PUR transferability LIMIT (honest attribution).

PUR (prediction in ungauged regions = transfer to Amu Darya) per-gauge skill
change of the flagship RegimeProbNet, screened against static catchment
attributes. The transfer is net-negative: 8 of 14 transfer gauges DEGRADE
(median delta-KGE' = -0.11, mean = -0.14). Degradation correlates with glacier
fraction and drainage area, but these two attributes are themselves strongly
collinear (Spearman r = 0.76) - a single size/glacier gradient, not two
independent controls - and neither correlation survives a Bonferroni correction
for the 5 attributes screened (0.05 / 5 = 0.01). The association is also levered
by 2 extreme points (see leave-2-out r). Framed strictly as a transferability
limit, not a success.

(a) delta-KGE' vs glacier fraction   (Spearman r = -0.63, robust Theil-Sen line)
(b) delta-KGE' vs drainage area      (Spearman r = -0.63, robust Theil-Sen line)
(c) full screening: Spearman r (with 95% bootstrap CI) for all 5 attributes,
    against the Bonferroni and uncorrected significance thresholds.
n = 14 transfer gauges - all points shown honestly, small sample.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r"G:/MDPI Q1-2026/src")
from sbc.viz import style  # noqa: E402

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

# Keep axes.unicode_minus = True (set in style.apply): axis ticks render the
# typographic minus (U+2212) so text annotations via style.num() match exactly.

ROOT = "G:/MDPI Q1-2026"
FAIL = f"{ROOT}/results/tables/pur_failure_gauges_v2_real_decadal.csv"
REG = f"{ROOT}/results/tables/pur_attr_regression_v2_real_decadal.csv"

GREY = "#555555"        # neutral annotation colour (audit M19)
GREY_L = "#888888"
KP = style.KGE_PRIME    # "KGE" + prime, set-wide metric glyph

# ---------------------------------------------------------------- load + merge
fg = pd.read_csv(FAIL)
fg["code"] = fg["code"].astype(str)
# area_km2 in this table is already log10(drainage area [km2]); recover raw km2.
fg["area_raw"] = 10.0 ** fg["area_km2"]
fg["gl_pct"] = 100.0 * fg["glacier_frac"]
d = fg["delta_kge"].values
n = len(fg)

# ---------------------------------------------------------------- headline stats
med = float(np.median(d))
mean = float(np.mean(d))
n_deg = int((d < 0).sum())
# glacier / area collinearity (one confounded gradient, not two; audit M19)
r_coll, _ = stats.spearmanr(fg["gl_pct"].values, fg["area_km2"].values)

# the 2 high-leverage points = the 2 most extreme delta-KGE' gauges
lev_idx = np.argsort(np.abs(d - np.median(d)))[-2:]
lev_codes = set(fg.iloc[lev_idx]["code"])


def boot_ci_r(x, y, nboot=5000, seed=0):
    """95% bootstrap percentile CI for Spearman r."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rs = []
    for _ in range(nboot):
        s = rng.choice(idx, len(x), replace=True)
        if len(np.unique(x[s])) < 3:
            continue
        rs.append(stats.spearmanr(x[s], y[s])[0])
    rs = np.asarray(rs)
    return np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)


# ---------------------------------------------------------------- regression tbl
reg = pd.read_csv(REG)
reg = reg[(reg["target"] == "delta_kge") & (reg["model"] == "regimeprobnet")].copy()
# real per-attribute x arrays for bootstrap CIs in panel (c)
ATTR_X = {
    "glacier_frac": fg["gl_pct"].values,
    "area_km2": fg["area_km2"].values,
    "elevation": fg["elevation"].values,
    "snow_frac": fg["snow_frac"].values,
    "aridity": fg["aridity"].values,
}
ATTR_NAME = {"glacier_frac": "Glacier fraction", "area_km2": "Drainage area (log)",
             "elevation": "Mean elevation", "snow_frac": "Snow fraction",
             "aridity": "Aridity index"}
reg = reg.sort_values("abs_spearman", ascending=True)  # weakest at bottom of lollipop

# Bonferroni: 5 attributes screened -> alpha = 0.05 / 5 = 0.01.
N_ATTR = 5
ALPHA_B = 0.05 / N_ATTR


def crit_r(alpha, nn):
    """Two-sided critical |Spearman r| (t-approximation, df = n-2)."""
    tc = stats.t.ppf(1 - alpha / 2, nn - 2)
    return float(np.sqrt(tc**2 / (tc**2 + (nn - 2))))


RC_05 = crit_r(0.05, n)       # uncorrected 0.05 threshold
RC_BONF = crit_r(ALPHA_B, n)  # Bonferroni 0.01 threshold

# ---------------------------------------------------------------- basin colours
basins = ["ZERAFSHAN", "PYANDZH", "VAKSH", "KOFARNIKHAN", "SURKHANDARYA"]
blabel = {b: b.title() for b in basins}
bcol = {
    "ZERAFSHAN": style.PALETTE[0],
    "PYANDZH": style.PALETTE[1],
    "VAKSH": style.PALETTE[2],
    "KOFARNIKHAN": style.PALETTE[3],
    "SURKHANDARYA": style.PALETTE[4],
}


def trend_band(ax, x, y, xfit, xplot=None):
    """OLS trend line + 95% mean-response confidence band - the smooth shaded
    curve (narrow at the data centre, widening at the extremes). The in-panel
    Spearman r is the rank-based, robust statistic; the band is an OLS visual
    guide whose leverage sensitivity is reported in the manuscript text."""
    x = np.asarray(x, float)
    yv = np.asarray(y, float)
    nn = len(x)
    sl, ic = np.polyfit(x, yv, 1)
    resid = yv - (ic + sl * x)
    s2 = float(np.sum(resid ** 2) / (nn - 2))
    xbar = float(x.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    tval = float(stats.t.ppf(0.975, nn - 2))
    yfit = ic + sl * xfit
    half = tval * np.sqrt(s2 * (1.0 / nn + (xfit - xbar) ** 2 / sxx))
    xd = xfit if xplot is None else xplot
    ax.fill_between(xd, yfit - half, yfit + half, color=GREY, alpha=0.15,
                    lw=0, zorder=1)
    ax.plot(xd, yfit, color=GREY, lw=1.6, zorder=2, label="_nolegend_")


def annot(ax, x):
    """A single concise Spearman r label; the full 5-attribute screen (CI,
    Bonferroni, leverage robustness) is reported in the manuscript text."""
    r, _ = stats.spearmanr(x, d)
    ax.text(0.965, 0.86, f"Spearman $r$ = {style.num(r, 2)}",
            transform=ax.transAxes, fontsize=7.8, color="#333333",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))


def scatter(ax, x):
    for b in basins:
        m = (fg["basin"] == b).values
        ax.scatter(x[m], d[m], s=54, c=bcol[b], edgecolors="white",
                   linewidths=0.6, zorder=4, label=blabel[b])


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(style.WIDTH_2COL, 3.9))
gs = fig.add_gridspec(1, 2, wspace=0.24)
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])

YL = (-1.45, 1.20)
for ax in (axa, axb):
    ax.axhline(0.0, color=GREY, lw=0.9, ls=(0, (5, 3)), zorder=0)
    ax.set_ylim(*YL)
    ax.set_ylabel(f"$\\Delta${KP} (corrected {style.MINUS} raw GloFAS)")

# improve / degrade guidance - neutral grey (audit M19)
for ax in (axa, axb):
    ax.text(0.97, 0.965, "improves", transform=ax.transAxes, fontsize=7,
            color=GREY_L, style="italic", va="top", ha="right")
    ax.text(0.97, 0.035, "degrades", transform=ax.transAxes, fontsize=7,
            color=GREY_L, style="italic", va="bottom", ha="right")

# --- (a) glacier fraction
xa = fg["gl_pct"].values
scatter(axa, xa)
trend_band(axa, xa, d, np.linspace(-0.4, xa.max() + 0.6, 50))
axa.set_xlim(-0.5, xa.max() + 0.8)
axa.set_xlabel("Glacier fraction (%)")
annot(axa, xa)
style.panel(axa, "a")

# --- (b) drainage area (log axis)
xb = fg["area_raw"].values
axb.set_xscale("log")
scatter(axb, xb)
lx = np.log10(xb)
xfit = np.linspace(lx.min() - 0.1, lx.max() + 0.1, 50)
trend_band(axb, lx, d, xfit, xplot=10.0 ** xfit)
axb.set_xlim(10 ** (lx.min() - 0.15), 10 ** (lx.max() + 0.15))
axb.set_xlabel("Drainage area (km²)")
axb.xaxis.set_major_formatter(style.thousands_formatter())
annot(axb, xb)
style.panel(axb, "b")

# shared basin legend below
handles, labels = axa.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.0,
           title="Amu Darya transfer basin (outside training support)",
           fontsize=7.8, title_fontsize=8)

fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.20, wspace=0.24)

style.savefig(fig, "fig11_pur_attribution")
print("saved fig11_pur_attribution")
print(f"median dKGE={med:.3f} mean={mean:.3f} degrade={n_deg}/{n} "
      f"collinearity r={r_coll:.2f} crit05={RC_05:.2f} critBonf={RC_BONF:.2f}")
print(f"high-leverage gauges: {sorted(lev_codes)}")
