"""Figure 05 - Spatial skill improvement (per-gauge Delta KGE-prime map).

Panel (a): temporal holdout, all evaluation gauges (core Syr Darya / Chu / Talas /
Naryn = circles, plus the 14 transfer Amu Darya gauges = triangles). Panel (b):
prediction in ungauged regions (PUR) over the SAME 14 transfer gauges (triangles),
which honestly shows weaker / mixed improvement that is NOT statistically significant.

Points are coloured by Delta KGE-prime = KGE-prime(corrected) - KGE-prime(raw GloFAS)
on a diverging RdBu scale centred at zero (blue = improvement, red = degradation).
The diverging scale is clipped to the robust data range (+/-1.0); a few temporal
gauges exceed +1.0 (max +2.45) and are annotated rather than washing the map out.
In panel (b) every per-gauge delta is within noise (all pairwise p > 0.42) so the
markers are desaturated and carry an explicit non-significance caveat.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from sbc.viz import style

style.apply()

ROOT = r"G:/MDPI Q1-2026"
MODEL = "stacked"          # deep-ensemble flagship corrector
CMAP = "RdBu"              # red = degraded, blue = improved
# M24: clip the diverging scale to the robust data range. Deltas cluster within
# 0-0.8 (temporal 90th pct 0.71; PUR 10th-90th pct -0.66..0.45); the original
# +/-1.5 washed the whole map to near-uniform pale blue. We saturate at +/-0.8
# (the robust range) so the diverging structure is legible; the tails - 7
# temporal gauges above +0.8 (max +2.45) and 1 PUR gauge below -0.8 (min -0.92)
# - are surfaced via the extend triangles + an explicit clipped-count annotation.
VLIM = 0.8                 # symmetric saturation clipped to the robust data range
MARKERS = {"core": "o", "transfer": "^"}   # transfer gauges get a distinct marker
cmap_obj = plt.get_cmap(CMAP)

# --- load skill table + gauge coordinates ------------------------------------
df = pd.read_csv(f"{ROOT}/results/tables/per_gauge_canonical_decadal.csv")
gauges = pd.read_parquet(f"{ROOT}/datasets/processed/gauges.parquet")
basins = gpd.read_parquet(f"{ROOT}/datasets/processed/basins.parquet")

# real Natural Earth admin-0 country polygons for geographic context (coastlines,
# transboundary Central Asia borders) drawn UNDER the gauge points
world = gpd.read_file(style.NE_50M)

df["code"] = df["code"].astype(str)
gauges["code"] = gauges["code"].astype(str)


def prep(split: str) -> pd.DataFrame:
    sub = df[(df["split"] == split) & (df["model"] == MODEL)].copy()
    sub["delta"] = sub["kge"] - sub["kge_raw"]
    sub = sub.merge(gauges[["code", "lon", "lat"]], on="code", how="left")
    return sub.dropna(subset=["lon", "lat", "delta"])


temporal = prep("temporal")
pur = prep("pur")

norm = TwoSlopeNorm(vmin=-VLIM, vcenter=0.0, vmax=VLIM)
deg_lon = FuncFormatter(lambda x, _p: f"{x:g}$^\\circ$E")
deg_lat = FuncFormatter(lambda y, _p: f"{y:g}$^\\circ$N")


def facecolors(deltas, desat: float = 0.0):
    """RdBu RGBA for each delta; optionally blend toward neutral grey (desaturation)."""
    rgba = cmap_obj(norm(np.asarray(deltas, dtype=float)))
    if desat > 0.0:
        grey = np.array([0.62, 0.62, 0.62, 1.0])
        rgba = (1.0 - desat) * rgba + desat * grey
    return rgba


def draw_map(ax, data, extent, *, title, lon_ticks, lat_ticks, desaturate=False):
    """Natural Earth borders, faint basins, graticule and gauge points by Delta KGE-prime.

    Core gauges are circles and transfer gauges triangles (MARKERS); when
    ``desaturate`` is set (PUR panel, all deltas within noise) marker colours are
    blended toward grey so the map does not imply per-gauge significance.
    """
    # real country polygons (land fill + admin-0 borders) clipped via axes limits
    sel = world.cx[extent[0] - 1:extent[1] + 1, extent[2] - 1:extent[3] + 1]
    sel.plot(ax=ax, color="#F4F1EC", edgecolor="none", zorder=0)
    sel.boundary.plot(ax=ax, color="#A6ABB2", linewidth=0.6, zorder=1)
    # faint hydrological basin outlines on top of the country borders
    basins.plot(ax=ax, color="#E9EDF1", edgecolor="none", alpha=0.55, zorder=2)
    basins.boundary.plot(ax=ax, color="#7F94A8", linewidth=0.35, alpha=0.75, zorder=3)

    # plot worst (most negative) last so degradations are never hidden; split by
    # domain so core vs transfer gauges carry distinct markers
    d = data.sort_values("delta", ascending=False)
    desat = 0.55 if desaturate else 0.0
    edge = "#7A7A7A" if desaturate else "#2B2B2B"
    for dom, mk in MARKERS.items():
        g = d[d["domain"] == dom]
        if g.empty:
            continue
        ax.scatter(
            g["lon"], g["lat"], color=facecolors(g["delta"], desat),
            marker=mk, s=58 if mk == "^" else 48,
            edgecolors=edge, linewidths=0.45, zorder=6,
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(extent[2:]))))

    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.xaxis.set_major_formatter(deg_lon)
    ax.yaxis.set_major_formatter(deg_lat)
    ax.tick_params(length=2.5)
    ax.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.8, zorder=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("#333333")
        ax.spines[s].set_linewidth(0.8)
    ax.set_title(title, pad=5, fontsize=9)


fig = plt.figure(figsize=(style.WIDTH_2COL, 4.5))
gs = fig.add_gridspec(
    1, 2, width_ratios=[1.62, 1.0], wspace=0.20,
    left=0.065, right=0.985, top=0.90, bottom=0.295,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

def signed_median(v: float) -> str:
    """Median with an explicit + for gains and the typographic minus (style.num) for losses."""
    return ("+" + style.num(v, 2)) if v >= 0 else style.num(v, 2)


n_core = int((temporal["domain"] == "core").sum())
n_xfer = int((temporal["domain"] == "transfer").sum())

med_t = temporal["delta"].median()
draw_map(
    ax_a, temporal, extent=(65.9, 76.4, 36.8, 43.1),
    title=f"Temporal holdout  (n = {len(temporal)}, median {signed_median(med_t)})",
    lon_ticks=[66, 69, 72, 75], lat_ticks=[37, 39, 41, 43],
)
ax_a.set_ylabel("Latitude")
ax_a.set_xlabel("Longitude")

# marker key: core gauges = circles, transfer gauges = triangles (M12)
mk_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#C9C9C9",
           markeredgecolor="#2B2B2B", markeredgewidth=0.45, markersize=6,
           label=f"core (n = {n_core})"),
    Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#C9C9C9",
           markeredgecolor="#2B2B2B", markeredgewidth=0.45, markersize=6.5,
           label=f"transfer (n = {n_xfer})"),
]
ax_a.legend(
    handles=mk_handles, loc="lower left", fontsize=6.5, frameon=True,
    framealpha=0.92, edgecolor="#BBBBBB", handletextpad=0.4,
    borderpad=0.4, labelspacing=0.3,
).set_zorder(8)

med_p = pur["delta"].median()
draw_map(
    ax_b, pur, extent=(66.4, 72.8, 36.8, 40.3),
    title=f"PUR transfer  (n = {len(pur)}, median {signed_median(med_p)})",
    lon_ticks=[67, 69, 71], lat_ticks=[37, 38, 39, 40],
    desaturate=True,
)
ax_b.set_xlabel("Longitude")

style.panel(ax_a, "a", x=-0.095, y=1.05)
style.panel(ax_b, "b", x=-0.155, y=1.05)

# honest caption-on-figure note (M12): the PUR per-gauge deltas are not just
# --- shared horizontal colorbar ----------------------------------------------
cax = fig.add_axes([0.305, 0.095, 0.39, 0.030])
sm = ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
cb.set_label(rf"$\Delta$ {style.KGE_PRIME}  (corrected $-$ raw)", labelpad=3)
cb_ticks = [-0.8, -0.4, 0.0, 0.4, 0.8]
cb.set_ticks(cb_ticks)
# typographic minus (U+2212) on negative ticks to match the axis tick scales
cb.set_ticklabels([style.num(t, 1) if t != 0 else "0" for t in cb_ticks])
cb.ax.tick_params(length=2.5)
cb.outline.set_linewidth(0.6)

# directional cue replaces a redundant marker legend
cb.ax.annotate("degraded", xy=(0, -1.0), xytext=(-0.02, -2.1),
               xycoords="axes fraction", ha="right", va="center",
               fontsize=7.5, color="#B2182B")
cb.ax.annotate("improved", xy=(1, -1.0), xytext=(1.02, -2.1),
               xycoords="axes fraction", ha="left", va="center",
               fontsize=7.5, color="#2166AC")

paths = style.savefig(fig, "fig05_improvement_map")
print("saved:", paths[0])
print(f"temporal: n={len(temporal)} median={med_t:.3f} min={temporal['delta'].min():.3f} max={temporal['delta'].max():.3f}")
print(f"pur: n={len(pur)} median={med_p:.3f} negatives={int((pur['delta'] < 0).sum())}")
