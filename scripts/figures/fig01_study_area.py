"""Figure 01 - Study area and PUR design.

Study-area map of snow-influenced transboundary Central Asia. Core basins
(Syr Darya / Chu / Talas / Naryn / Chirchik / Qashqadarya / Akhangaran) are the
calibration domain; the Amu Darya system is held out as the transfer / PUR
(prediction in ungauged regions) domain. Gauges are sized by mean reference
discharge; the two daily Naryn gauges (16055, 16068) are marked distinctly.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle
import matplotlib.patheffects as pe

from sbc.viz import style

style.apply()

CORE = style.DOMAIN_COLORS["core"]
TRANSFER = style.DOMAIN_COLORS["transfer"]
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# ---------------------------------------------------------------- load data ---
basins = gpd.read_parquet(r"G:/MDPI Q1-2026/datasets/processed/basins.parquet")
gauges = pd.read_parquet(r"G:/MDPI Q1-2026/datasets/processed/gauges.parquet")

LON0, LAT0, LON1, LAT1 = basins.total_bounds
pad = 0.25
LON0, LON1 = LON0 - pad, LON1 + pad
LAT0, LAT1 = LAT0 - pad, LAT1 + pad
MID_LAT = 0.5 * (LAT0 + LAT1)
ASPECT = 1.0 / np.cos(np.deg2rad(MID_LAT))

# ---------------------------------------------------------------- figure ------
fig = plt.figure(figsize=(style.WIDTH_2COL, style.WIDTH_2COL * 0.74))
ax = fig.add_axes([0.07, 0.07, 0.90, 0.90])

# faint country borders for geographic context (real Natural Earth coastlines)
try:
    _world = gpd.read_file(style.NE_50M)
    _world.plot(ax=ax, facecolor="#F4F1EC", edgecolor="#C8C2B8",
                linewidth=0.4, zorder=0)
except Exception as _e:  # pragma: no cover
    print("NE context skipped:", _e)

# basin fills by domain (light tint) with thin sub-catchment edges
for dom, col in (("core", CORE), ("transfer", TRANSFER)):
    sub = basins[basins.domain == dom]
    sub.plot(ax=ax, facecolor=col, edgecolor="white", linewidth=0.35, alpha=0.40)
# strong domain outline
for dom, col in (("core", CORE), ("transfer", TRANSFER)):
    basins[basins.domain == dom].dissolve("domain").plot(
        ax=ax, facecolor="none", edgecolor=col, linewidth=1.3)

# ---------------------------------------------------------------- gauges ------
def msize(q):
    return 14.0 + 9.0 * np.sqrt(q)

daily_mask = gauges.scale == "daily"
nan_mask = gauges.q_mean_ref.isna()
for dom, col in (("core", CORE), ("transfer", TRANSFER)):
    sub = gauges[(gauges.domain == dom) & (~daily_mask) & (~nan_mask)]
    ax.scatter(sub.lon, sub.lat, s=msize(sub.q_mean_ref), facecolor=col,
               edgecolor="#222222", linewidth=0.5, alpha=0.92, zorder=5)

# gauges with no reference discharge (e.g. 16107) would give msize(NaN) and be
# silently dropped; plot them explicitly at minimum size with a distinct hollow
# diamond so the network is never undercounted (74/74 decadal markers shown)
nan_g = gauges[nan_mask & (~daily_mask)]
ax.scatter(nan_g.lon, nan_g.lat, s=msize(0.0), facecolor="white",
           edgecolor="#222222", linewidth=0.9, marker="D", alpha=0.97, zorder=6)
for _, r in nan_g.iterrows():
    ax.annotate(f"{r.code}", (r.lon, r.lat), xytext=(7, -1),
                textcoords="offset points", fontsize=6.2, color="#444444",
                ha="left", va="center", zorder=8, path_effects=HALO)

# daily Naryn gauges - distinct star markers
dg = gauges[daily_mask].set_index("code")
ax.scatter(dg.lon, dg.lat, s=msize(dg.q_mean_ref) * 2.2, marker="*",
           facecolor="#F0E442", edgecolor="#222222", linewidth=0.8, zorder=7)
_off = {"16055": (-8, -9, "right", "top"), "16068": (9, -9, "left", "top")}
for code, r in dg.iterrows():
    dx, dy, ha, va = _off[code]
    ax.annotate(code, (r.lon, r.lat), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.8, fontweight="bold",
                color="#222222", ha=ha, va=va, zorder=8, path_effects=HALO)

# ---------------------------------------------------------------- labels ------
CORE_LABELS = {
    "Naryn": (74.55, 41.85), "Syr Darya": (73.81, 40.30), "Chu": (75.55, 42.35),
    "Talas": (72.30, 42.55), "Chirchik": (70.10, 41.95),
    "Akhangaran": (69.70, 40.95), "Qashqadarya": (66.95, 38.55),
}
for name, (x, y) in CORE_LABELS.items():
    ax.text(x, y, name, fontsize=7.6, fontweight="bold", color="#0A3A5A",
            ha="center", va="center", zorder=9, path_effects=HALO)
ax.text(72.0, 36.6, "Amu Darya / Zerafshan\n(transfer / PUR hold-out)", fontsize=8.2,
        fontweight="bold", color="#8A3A00", ha="center", va="center", zorder=9,
        path_effects=HALO)

# ---------------------------------------------------------------- graticule ---
ax.set_xlim(LON0, LON1)
ax.set_ylim(LAT0, LAT1)
ax.set_aspect(ASPECT)
xticks = np.arange(np.ceil(LON0), np.floor(LON1) + 1, 2)
yticks = np.arange(np.ceil(LAT0), np.floor(LAT1) + 1, 1)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f"{v:g}°E" for v in xticks])
ax.set_yticklabels([f"{v:g}°N" for v in yticks])
ax.tick_params(labelsize=7.5)
ax.grid(True, color="#BBBBBB", linewidth=0.5, alpha=0.7, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_edgecolor("#333333")

# ---------------------------------------------------------------- scale bar ---
# accurate length: at the map mean latitude, 1 deg lon = 111.320*cos(lat) km
KM_PER_DEG_LON = 111.320 * np.cos(np.deg2rad(MID_LAT))
km = 200.0                                    # exact round value
deg = km / KM_PER_DEG_LON
x0, y0 = LON0 + 0.45, LAT0 + 0.40
ax.plot([x0, x0 + deg], [y0, y0], color="#222222", lw=2.6, solid_capstyle="butt",
        zorder=10)
ax.plot([x0, x0], [y0 - 0.05, y0 + 0.05], color="#222222", lw=1.2, zorder=10)
ax.plot([x0 + deg, x0 + deg], [y0 - 0.05, y0 + 0.05], color="#222222", lw=1.2,
        zorder=10)
ax.text(x0 + deg / 2, y0 + 0.12, f"{km:,.0f} km", fontsize=7.0, ha="center",
        va="bottom", color="#222222", path_effects=HALO)
# EPSG:4326 has no single true scale: 1 deg lon shrinks with latitude, so the
# bar length is exact only at the latitude used to size it. State that latitude.
ax.text(x0 + deg / 2, y0 - 0.16,
        "WGS84 (EPSG:4326)", fontsize=6.0,
        ha="center", va="top", color="#555555", path_effects=HALO)

# ---------------------------------------------------------------- north arrow -
nx, ny = LON0 + 0.62, LAT0 + 1.25   # empty SW corner, clear of basins/scale bar
ax.add_patch(FancyArrow(nx, ny, 0, 0.55, width=0.0, head_width=0.22,
                        head_length=0.22, length_includes_head=True,
                        color="#222222", zorder=10))
ax.text(nx, ny + 0.78, "N", fontsize=10, fontweight="bold", ha="center",
        va="bottom", color="#222222", path_effects=HALO)

# ---------------------------------------------------------------- legends -----
# domain legend (upper-left)
dom_handles = [
    mpatches.Patch(facecolor=CORE, edgecolor=CORE, alpha=0.55,
                   label="Core domain (calibration)"),
    mpatches.Patch(facecolor=TRANSFER, edgecolor=TRANSFER, alpha=0.55,
                   label="Transfer domain (PUR hold-out)"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#F0E442",
           markeredgecolor="#222222", markersize=11,
           label="Daily Naryn gauge (16055, 16068)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
           markeredgecolor="#222222", markeredgewidth=0.9, markersize=6.5,
           label="Decadal gauge, no discharge data (16107)"),
]
leg1 = ax.legend(handles=dom_handles, loc="upper left", fontsize=7.3,
                 frameon=True, framealpha=0.92, edgecolor="#BBBBBB",
                 borderpad=0.6, handlelength=1.4, labelspacing=0.5)
leg1.set_zorder(12)
ax.add_artist(leg1)

# gauge size key - placed INSIDE the map frame (right-centre, a clear area east
# of the basins and below the daily Naryn gauges); tighter spacing, unit in bounds
size_vals = [10, 100, 1000]
keyax = ax.inset_axes([0.795, 0.330, 0.185, 0.285])
keyax.set_xlim(0, 1)
keyax.set_ylim(0, 1)
keyax.set_xticks([])
keyax.set_yticks([])
keyax.set_facecolor("white")
keyax.patch.set_alpha(0.90)
for sp in keyax.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor("#BBBBBB")
    sp.set_linewidth(0.8)
keyax.set_zorder(12)
# title goes INSIDE the box (set_title renders above the border, outside the
# block); DejaVu Sans carries the U+207B superscript-minus that Arial lacks
keyax.text(0.5, 0.965, "Mean discharge\n(" + style.UNIT_Q + ")",
           transform=keyax.transAxes, ha="center", va="top",
           fontsize=6.5, fontfamily="DejaVu Sans", linespacing=1.1)
_cx = 0.30
_rows_y = [0.595, 0.375, 0.185]         # tight group directly below the title
for v, ry in zip(sorted(size_vals, reverse=True), _rows_y):
    keyax.scatter([_cx], [ry], s=msize(v), facecolor="#888888",
                  edgecolor="#222222", linewidth=0.5, zorder=3)
    keyax.text(0.56, ry, f"{v:,}", fontsize=6.8, ha="left", va="center",
               color="#222222")

# ---------------------------------------------------------------- inset -------
# real regional-context map from Natural Earth admin-0 country polygons
AS_LON0, AS_LON1, AS_LAT0, AS_LAT1 = 45, 100, 20, 55
axins = fig.add_axes([0.715, 0.090, 0.235, 0.235])
axins.set_xlim(AS_LON0, AS_LON1)
axins.set_ylim(AS_LAT0, AS_LAT1)
axins.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (AS_LAT0 + AS_LAT1))))
if "_world" not in dir():
    _world = gpd.read_file(style.NE_50M)
# ocean/background fill
axins.set_facecolor("#DCE8F0")
# light country fills with thin borders, clipped to the inset extent
_ctx = _world.cx[AS_LON0:AS_LON1, AS_LAT0:AS_LAT1]
_ctx.plot(ax=axins, facecolor="#F0ECE3", edgecolor="#9A948A", linewidth=0.45,
          zorder=1)
# study-area highlight rectangle at the study bbox
S_LON0, S_LAT0, S_LON1, S_LAT1 = 66.4, 35.4, 78.4, 43.0
axins.add_patch(Rectangle((S_LON0, S_LAT0), S_LON1 - S_LON0, S_LAT1 - S_LAT0,
                          facecolor="none", edgecolor="#D7191C", linewidth=1.3,
                          zorder=4))
_scx, _scy = 0.5 * (S_LON0 + S_LON1), 0.5 * (S_LAT0 + S_LAT1)
axins.plot(_scx, _scy, marker="o", markersize=3.2, color="#D7191C", zorder=5)
axins.annotate("Study area", (S_LON1, S_LAT1), xytext=(8, 9),
               textcoords="offset points", fontsize=6.4, color="#D7191C",
               fontweight="bold", ha="left", va="bottom", zorder=6,
               path_effects=HALO,
               arrowprops=dict(arrowstyle="-", color="#D7191C", lw=0.7))
axins.set_xticks([])
axins.set_yticks([])
for sp in axins.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor("#555555")
    sp.set_linewidth(0.9)
axins.set_title("Regional context", fontsize=6.8, pad=2)

ax.set_xlabel("Longitude", fontsize=8.5)
ax.set_ylabel("Latitude", fontsize=8.5)

style.savefig(fig, "fig01_study_area")
print("saved")
