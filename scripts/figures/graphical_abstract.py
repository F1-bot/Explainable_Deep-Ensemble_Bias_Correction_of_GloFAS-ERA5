"""Graphical abstract for the MDPI *Water* paper — a clean, high-resolution,
three-panel layout in the house teal/green palette.

Panel 1  Problem & Inputs            — global product vs. local fidelity
Panel 2  Explainable Deep-Ensemble   — the RegimeProbNet flagship architecture
Panel 3  Key Findings & Outputs      — skill, calibration, interpretability

The figure is recreated from scratch (matplotlib + numpy; scipy for the soft
background, PIL for the final RGB flatten). All numbers are the verified,
honest values from the manuscript. No over-claims, no "Graphical Abstract"
heading inside the artwork.

Run:  python scripts/figures/graphical_abstract.py
Out:  results/figures/publication/graphical_abstract.png  (2600 x 1300, 300 dpi)
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Ellipse, Polygon, Rectangle
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------- #
#  Canvas + palette
# --------------------------------------------------------------------------- #
W, H = 2600, 1300

TEAL_DARK = "#0B6E72"     # header gradient start, footer, deep accents
TEAL      = "#0E8C86"     # primary teal (icons, lines)
TEAL_MID  = "#1AA39A"
GREEN     = "#2FA76C"     # header gradient end
GREEN2    = "#3FB07A"
TEAL_L    = "#BFE0DC"     # chip / box borders
PALE      = "#EAF6F3"     # chip / box fills
PALE2     = "#DCEFEB"     # alt fill
BG        = "#F3F8F7"     # page background base
INK       = "#15333A"     # primary text
GREY      = "#566B6E"     # secondary text
RAW       = "#9AA7AC"     # raw / reference series
BAND      = "#1AA39A"     # uncertainty band

# typographic glyphs
PRIME = "′"          # KGE prime
ARROW = "→"          # right arrow
DOT   = "·"          # middle dot
UNIT  = "m³ s⁻¹"
KGE   = "KGE" + PRIME

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": True,
    "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# --------------------------------------------------------------------------- #
#  Geometry of the three panels (pixel coordinates, y up)
# --------------------------------------------------------------------------- #
M, GAP = 28, 30
PW = (W - 2 * M - 2 * GAP) / 3.0
PTOP, PBOT = H - M, M
HH, R = 176, 24                       # header height, corner radius
PX = [M, M + PW + GAP, M + 2 * (PW + GAP)]
PCX = [x + PW / 2 for x in PX]
HDR_BOT = PTOP - HH                    # y of header/body divider


# --------------------------------------------------------------------------- #
#  Path helpers
# --------------------------------------------------------------------------- #
def rounded_rect(x0, y0, w, h, r):
    r = min(r, w / 2, h / 2)
    v = [(x0 + r, y0), (x0 + w - r, y0), (x0 + w, y0), (x0 + w, y0 + r),
         (x0 + w, y0 + h - r), (x0 + w, y0 + h), (x0 + w - r, y0 + h),
         (x0 + r, y0 + h), (x0, y0 + h), (x0, y0 + h - r),
         (x0, y0 + r), (x0, y0), (x0 + r, y0)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
         Path.LINETO, Path.CURVE3, Path.CLOSEPOLY]
    return Path(v, c)


def round_top_rect(x0, y0, w, h, r):
    """Rectangle with only the two TOP corners rounded (square bottom)."""
    v = [(x0, y0), (x0, y0 + h - r), (x0, y0 + h), (x0 + r, y0 + h),
         (x0 + w - r, y0 + h), (x0 + w, y0 + h), (x0 + w, y0 + h - r),
         (x0 + w, y0), (x0, y0)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    return Path(v, c)


def add_round(ax, x0, y0, w, h, r, **kw):
    p = PathPatch(rounded_rect(x0, y0, w, h, r), **kw)
    ax.add_patch(p)
    return p


# --------------------------------------------------------------------------- #
#  Icon library (simple filled / line glyphs, drawn at centre (cx,cy), scale s)
# --------------------------------------------------------------------------- #
def ic_droplet(ax, cx, cy, s, col, z=8):
    ax.add_patch(Circle((cx, cy - 0.18 * s), 0.62 * s, fc=col, ec="none", zorder=z))
    ax.add_patch(Polygon([(cx - 0.44 * s, cy + 0.10 * s), (cx, cy + 0.98 * s),
                          (cx + 0.44 * s, cy + 0.10 * s)], fc=col, ec="none", zorder=z))
    ax.add_patch(Circle((cx - 0.18 * s, cy - 0.30 * s), 0.16 * s, fc="white",
                        ec="none", alpha=0.55, zorder=z + 1))


def ic_cloud(ax, cx, cy, s, col, z=8):
    for dx, dy, r in [(-0.42, 0.02, 0.42), (0.40, 0.04, 0.46), (0.0, 0.30, 0.50)]:
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), r * s, fc=col, ec="none", zorder=z))
    ax.add_patch(Rectangle((cx - 0.82 * s, cy - 0.10 * s), 1.64 * s, 0.42 * s,
                           fc=col, ec="none", zorder=z))
    # falling snow under the cloud
    for dx in (-0.45, 0.0, 0.45):
        ax.plot([cx + dx * s], [cy - 0.55 * s], marker="*", ms=4.6, color=col, zorder=z + 1)


def ic_database(ax, cx, cy, s, col, z=8):
    w, h = 1.0 * s, 1.2 * s
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + 0.12 * s), w, h - 0.24 * s,
                           fc=col, ec="none", zorder=z))
    for dy in (h / 2 - 0.12 * s, 0.12 * s, -0.18 * s):
        ax.add_patch(Ellipse((cx, cy + dy), w, 0.30 * s, fc=col, ec="none", zorder=z))
    for dy in (0.12 * s, -0.18 * s):
        ax.add_patch(Ellipse((cx, cy + dy), w, 0.30 * s, fc="none", ec="white",
                            lw=1.3, zorder=z + 1))
    ax.add_patch(Ellipse((cx, cy + h / 2 - 0.12 * s), w, 0.30 * s, fc=col,
                        ec="white", lw=1.3, zorder=z + 1))


def ic_mountain(ax, cx, cy, s, col, z=8):
    ax.add_patch(Polygon([(cx - 1.0 * s, cy - 0.55 * s), (cx - 0.32 * s, cy + 0.62 * s),
                          (cx + 0.30 * s, cy - 0.55 * s)], fc=col, ec="none", zorder=z))
    ax.add_patch(Polygon([(cx - 0.18 * s, cy - 0.55 * s), (cx + 0.5 * s, cy + 0.86 * s),
                          (cx + 1.05 * s, cy - 0.55 * s)], fc=TEAL_MID, ec="none", zorder=z))
    # snow caps
    ax.add_patch(Polygon([(cx - 0.46 * s, cy + 0.30 * s), (cx - 0.32 * s, cy + 0.62 * s),
                          (cx - 0.17 * s, cy + 0.30 * s)], fc="white", ec="none", zorder=z + 1))
    ax.add_patch(Polygon([(cx + 0.33 * s, cy + 0.52 * s), (cx + 0.5 * s, cy + 0.86 * s),
                          (cx + 0.67 * s, cy + 0.52 * s)], fc="white", ec="none", zorder=z + 1))


def ic_snowflake(ax, cx, cy, s, col, z=8):
    for a in range(0, 180, 60):
        rad = np.radians(a)
        dx, dy = np.cos(rad) * s, np.sin(rad) * s
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=col, lw=2.4,
                solid_capstyle="round", zorder=z)
        for t in (0.55, -0.55):
            bx, by = cx + dx * t, cy + dy * t
            px, py = -dy / s * 0.28 * s, dx / s * 0.28 * s
            ax.plot([bx - px, bx + px], [by - py, by + py], color=col, lw=2.0,
                    solid_capstyle="round", zorder=z)


def ic_network(ax, cx, cy, s, col, z=8):
    layers = [(-0.85, [-0.6, 0.0, 0.6]), (0.0, [-0.35, 0.35]), (0.85, [0.0])]
    pts = {}
    for li, (lx, ys) in enumerate(layers):
        pts[li] = [(cx + lx * s, cy + yy * s) for yy in ys]
    for a, b in [(0, 1), (1, 2)]:
        for p in pts[a]:
            for q in pts[b]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.9, alpha=0.55, zorder=z)
    for li in pts:
        for p in pts[li]:
            ax.add_patch(Circle(p, 0.13 * s, fc=col, ec="white", lw=1.0, zorder=z + 1))


def ic_gate(ax, cx, cy, s, col, z=8):
    # one input fanning into three experts (mixture of experts)
    ax.add_patch(Circle((cx - 0.8 * s, cy), 0.16 * s, fc=col, ec="none", zorder=z + 1))
    outs = [0.65 * s, 0.0, -0.65 * s]
    for oy in outs:
        ax.plot([cx - 0.8 * s, cx + 0.35 * s], [cy, cy + oy], color=col, lw=1.4,
                alpha=0.7, zorder=z)
        ax.add_patch(Rectangle((cx + 0.35 * s, cy + oy - 0.16 * s), 0.42 * s, 0.32 * s,
                               fc=col, ec="none", zorder=z + 1))


def ic_gauss(ax, cx, cy, s, col, z=8):
    xx = np.linspace(-1.15, 1.15, 80)
    yy = np.exp(-(xx ** 2) / 0.42)
    ax.plot(cx + xx * s, cy - 0.55 * s + yy * 1.15 * s, color=col, lw=2.4,
            solid_capstyle="round", zorder=z + 1)
    ax.plot([cx - 1.15 * s, cx + 1.15 * s], [cy - 0.55 * s, cy - 0.55 * s],
            color=col, lw=1.4, zorder=z)


def ic_bulb(ax, cx, cy, s, col, z=8):
    ax.add_patch(Circle((cx, cy + 0.18 * s), 0.62 * s, fc=col, ec="none", zorder=z))
    ax.add_patch(Rectangle((cx - 0.26 * s, cy - 0.62 * s), 0.52 * s, 0.34 * s,
                           fc=col, ec="none", zorder=z))
    for dy in (-0.40 * s, -0.52 * s):
        ax.plot([cx - 0.24 * s, cx + 0.24 * s], [cy + dy, cy + dy], color="white",
                lw=1.3, zorder=z + 1)
    ax.plot([cx, cx], [cy + 0.0, cy + 0.5 * s], color="white", lw=1.4, zorder=z + 1)


def ic_shield(ax, cx, cy, s, col, z=8):
    v = [(cx, cy + 0.95 * s), (cx + 0.78 * s, cy + 0.55 * s),
         (cx + 0.78 * s, cy - 0.20 * s), (cx, cy - 0.95 * s),
         (cx - 0.78 * s, cy - 0.20 * s), (cx - 0.78 * s, cy + 0.55 * s)]
    ax.add_patch(Polygon(v, closed=True, fc=col, ec="none", zorder=z))
    ax.plot([cx - 0.34 * s, cx - 0.06 * s, cx + 0.40 * s],
            [cy + 0.02 * s, cy - 0.28 * s, cy + 0.36 * s],
            color="white", lw=2.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=z + 1)


def ic_check(ax, cx, cy, s, col=TEAL, z=8):
    ax.add_patch(Circle((cx, cy), s, fc=PALE, ec=TEAL_L, lw=1.4, zorder=z))
    ax.plot([cx - 0.42 * s, cx - 0.10 * s, cx + 0.45 * s],
            [cy + 0.02 * s, cy - 0.32 * s, cy + 0.38 * s],
            color=TEAL, lw=2.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=z + 1)


# --------------------------------------------------------------------------- #
#  Composite helpers
# --------------------------------------------------------------------------- #
def chip(ax, cx, cy, size, icon_fn, fc=PALE, ec=TEAL_L, icol=TEAL, iscale=0.42):
    add_round(ax, cx - size / 2, cy - size / 2, size, size, size * 0.28,
              fc=fc, ec=ec, lw=1.6, zorder=6)
    icon_fn(ax, cx, cy, size * iscale, icol)


def two_line(ax, x, cy, title, sub=None, tfs=9.5, sfs=8.0, color=INK,
             scolor=GREY, weight="bold", ha="left"):
    if sub:
        ax.text(x, cy + 14, title, va="center", ha=ha, fontsize=tfs,
                fontweight=weight, color=color, zorder=9)
        ax.text(x, cy - 15, sub, va="center", ha=ha, fontsize=sfs,
                color=scolor, zorder=9)
    else:
        ax.text(x, cy, title, va="center", ha=ha, fontsize=tfs,
                fontweight=weight, color=color, zorder=9)


def chevron(ax, cx, cy, w=14, h=11, col=TEAL):
    ax.plot([cx - w, cx, cx + w], [cy + h, cy - h, cy + h], color=col, lw=3.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=7)


def header(ax, num, title_lines, subtitle, px0):
    patch = PathPatch(round_top_rect(px0, HDR_BOT, PW, HH, R), fc=TEAL_DARK,
                      ec="none", zorder=3)
    ax.add_patch(patch)
    c0, c1 = np.array(to_rgb(TEAL_DARK)), np.array(to_rgb(GREEN))
    grad = (c0[None, :] + (c1 - c0)[None, :] * np.linspace(0, 1, 256)[:, None])[None]
    im = ax.imshow(grad, extent=[px0, px0 + PW, HDR_BOT, PTOP], aspect="auto",
                   zorder=3, interpolation="bilinear")
    im.set_clip_path(patch)
    # number badge
    bx = px0 + 50
    n = len(title_lines)
    ty_top = PTOP - 56
    by = ty_top - (n - 1) * 21
    ax.add_patch(Circle((bx, by), 25, fc="white", ec="none", alpha=0.96, zorder=4))
    ax.text(bx, by, str(num), ha="center", va="center", color=TEAL_DARK,
            fontsize=14, fontweight="bold", zorder=5)
    tx = bx + 46
    for i, line in enumerate(title_lines):
        ax.text(tx, ty_top - i * 42, line, ha="left", va="center", color="white",
                fontsize=14, fontweight="bold", zorder=5)
    ax.text(tx, HDR_BOT + 30, subtitle, ha="left", va="center", color="#EAFBF7",
            fontsize=9.6, style="italic", zorder=5)


def card(ax, x0, y0, w, h, **kw):
    kw.setdefault("fc", "white")
    kw.setdefault("ec", TEAL_L)
    kw.setdefault("lw", 1.4)
    add_round(ax, x0, y0, w, h, 16, zorder=6, **kw)


def frac(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


# --------------------------------------------------------------------------- #
#  Figure + soft background
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(W / 300, H / 300), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

xx, yy = np.meshgrid(np.linspace(0, 1, W // 2), np.linspace(0, 1, H // 2))
bg = np.ones((H // 2, W // 2, 3)) * np.array(to_rgb(BG))
for cx, cy, r, col, a in [(0.20, 0.85, 0.6, "#E2F1EE", 0.7),
                          (0.85, 0.80, 0.6, "#E6F1F4", 0.6),
                          (0.5, 0.1, 0.7, "#EAF4F1", 0.5)]:
    wgt = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))) * a
    bg = bg * (1 - wgt[..., None]) + np.array(to_rgb(col)) * wgt[..., None]
bg = gaussian_filter(bg, sigma=(12, 12, 0))
ax.imshow(bg, extent=[0, W, 0, H], aspect="auto", zorder=0, interpolation="bilinear")

# panel bodies (white rounded cards) + soft shadow
for px0 in PX:
    add_round(ax, px0 + 5, PBOT - 6, PW, PH := (PTOP - PBOT), R,
              fc="#C9D8D6", ec="none", alpha=0.30, zorder=1)
    add_round(ax, px0, PBOT, PW, PTOP - PBOT, R, fc="white", ec="#D6E6E3",
              lw=1.4, zorder=2)

header(ax, 1, ["Problem & Inputs"], "Global product vs. local fidelity", PX[0])
header(ax, 2, ["Explainable Deep-", "Ensemble Corrector"],
       "Flagship architecture (RegimeProbNet)", PX[1])
header(ax, 3, ["Key Findings & Outputs"],
       f"Skill {DOT} Calibration {DOT} Interpretability", PX[2])

# =========================================================================== #
#  PANEL 1 — Problem & Inputs
# =========================================================================== #
p0 = PX[0]
# basin banner
chip(ax, p0 + 88, 1052, 74, ic_mountain, fc=PALE, icol=TEAL)
two_line(ax, p0 + 138, 1052,
         "Transboundary Central Asian basins",
         f"Syr Darya  {DOT}  Amu Darya  {DOT}  snow-fed",
         tfs=9.6, sfs=8.4)

# hydrograph inset card: raw underestimates the freshet
hx, hy, hw, hh = p0 + 38, 758, PW - 76, 210
card(ax, hx, hy, hw, hh, fc="#FBFEFD")
ax.text(hx + 16, hy + hh - 14, "Raw product underestimates the freshet",
        fontsize=8.6, fontweight="bold", color=INK, va="top", zorder=9)
ax1 = fig.add_axes(frac(hx + 44, hy + 16, hw - 120, hh - 62))
ax1.set_facecolor("none")
t = np.linspace(0, 12, 400)


def bell(pk, tc, wd, base, wob=0.0):
    return base + pk * np.exp(-0.5 * ((t - tc) / wd) ** 2) * (1 + wob * np.sin(2.0 * t))


obs = bell(100, 6.4, 1.55, 8, 0.05)
raw = bell(34, 5.5, 2.15, 7)
ax1.fill_between(t, raw, obs, where=obs > raw, color="#C9633A", alpha=0.13, lw=0)
ax1.plot(t, raw, color=RAW, lw=2.0, ls=(0, (5, 2.4)))
ax1.plot(t, obs, color=INK, lw=2.0)
ax1.annotate("", xy=(6.4, 96), xytext=(6.4, 46),
             arrowprops=dict(arrowstyle="-|>", color="#C9633A", lw=1.6))
ax1.text(7.0, 66, "missed\npeak", fontsize=7.0, color="#A84D2A", va="center", ha="left")
ax1.text(0.4, 112, "observed", fontsize=7.0, color=INK, ha="left", va="top", fontweight="bold")
ax1.text(0.4, 26, "raw GloFAS-ERA5", fontsize=7.0, color=GREY, ha="left", va="center")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color("#BBC9C8")
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 128)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel(f"Discharge ({UNIT})", fontsize=6.6, color=GREY, labelpad=2)

# three labelled inputs
inputs = [
    (ic_droplet, "Raw GloFAS-ERA5 discharge", "global product (to be corrected)"),
    (ic_cloud,   "ERA5-Land forcing", f"meteorology {DOT} SWE {DOT} snowmelt"),
    (ic_database, "HydroATLAS", "static basin attributes"),
]
ys = [600, 392, 184]
ax.text(p0 + 40, 690, "INPUTS", fontsize=8.2, fontweight="bold", color=TEAL,
        va="center", zorder=9)
ax.plot([p0 + 118, p0 + PW - 40], [690, 690], color=TEAL_L, lw=1.2, zorder=6)
for (fn, ttl, sub), yc in zip(inputs, ys):
    chip(ax, p0 + 96, yc, 84, fn, fc=PALE, icol=TEAL)
    two_line(ax, p0 + 158, yc, ttl, sub, tfs=10.0, sfs=8.4)

# =========================================================================== #
#  PANEL 2 — Explainable Deep-Ensemble Corrector
# =========================================================================== #
p1 = PX[1]
cx = PCX[1]
bw = PW - 64
bx0 = cx - bw / 2

ax.text(p1 + 40, 1052, "ARCHITECTURE FLOW", fontsize=8.2, fontweight="bold",
        color=TEAL, va="center", zorder=9)
ax.plot([p1 + 255, p1 + PW - 40], [1052, 1052], color=TEAL_L, lw=1.2, zorder=6)

flow = [
    (ic_network, ["Entity-Aware LSTM", "(EA-LSTM)"], 990),
    (ic_gate,    ["Regime-Gated Mixture-", "of-Experts (MoE)"], 858),
    (ic_gauss,   ["Closed-form Gaussian /", "CMAL CRPS loss"], 726),
]
bh = 96
# faint flow spine behind boxes
ax.plot([cx, cx], [726 - bh / 2, 990 + bh / 2], color=TEAL_L, lw=2.0, zorder=4)
for fn, lines, yc in flow:
    add_round(ax, bx0, yc - bh / 2, bw, bh, 16, fc=PALE, ec=TEAL, lw=1.7, zorder=6)
    chip(ax, bx0 + 52, yc, 64, fn, fc="white", ec=TEAL_L, icol=TEAL, iscale=0.46)
    ax.text(bx0 + 100, yc + 15, lines[0], va="center", ha="left", fontsize=10.0,
            fontweight="bold", color=INK, zorder=9)
    ax.text(bx0 + 100, yc - 15, lines[1], va="center", ha="left", fontsize=10.0,
            fontweight="bold", color=INK, zorder=9)
chevron(ax, cx, 990 - bh / 2 - 17)
chevron(ax, cx, 858 - bh / 2 - 17)

# snow-physics constraints pill
sy = 612
add_round(ax, bx0, sy - 44, bw, 88, 16, fc=PALE2, ec=TEAL_MID, lw=1.6, zorder=6)
chip(ax, bx0 + 52, sy, 60, ic_snowflake, fc="white", ec=TEAL_L, icol=TEAL_MID, iscale=0.46)
ax.text(bx0 + 100, sy + 15, "Snow-physics constraints", va="center", ha="left",
        fontsize=10.0, fontweight="bold", color=TEAL_DARK, zorder=9)
ax.text(bx0 + 100, sy - 15, "soft & hard monotonicity", va="center", ha="left",
        fontsize=8.6, color=GREY, zorder=9)

# two supporting capabilities
ax.text(p1 + 40, 512, "TRUSTWORTHY BY DESIGN", fontsize=8.2, fontweight="bold",
        color=TEAL, va="center", zorder=9)
ax.plot([p1 + 290, p1 + PW - 40], [512, 512], color=TEAL_L, lw=1.2, zorder=6)
caps = [
    (ic_bulb, 430, ["Explainability (XAI)", "grouped Shapley & ALE"]),
    (ic_shield, 300, ["Trustworthy UQ — regime-conditional",
                      "(Mondrian) conformal intervals"]),
]
for fn, yc, lines in caps:
    add_round(ax, bx0, yc - 46, bw, 92, 16, fc="white", ec=TEAL_L, lw=1.4, zorder=6)
    chip(ax, bx0 + 52, yc, 64, fn, fc=PALE, ec=TEAL_L, icol=TEAL, iscale=0.46)
    ax.text(bx0 + 100, yc + 16, lines[0], va="center", ha="left", fontsize=9.6,
            fontweight="bold", color=INK, zorder=9)
    ax.text(bx0 + 100, yc - 16, lines[1], va="center", ha="left", fontsize=9.0,
            color=GREY, zorder=9)

# =========================================================================== #
#  PANEL 3 — Key Findings & Outputs
# =========================================================================== #
p2 = PX[2]
# hydrograph inset: corrected inside calibrated 90% band
hx, hy, hw, hh = p2 + 38, 800, PW - 76, 212
card(ax, hx, hy, hw, hh, fc="#FBFEFD")
ax.text(hx + 16, hy + hh - 14, f"Corrected flow {DOT} calibrated 90% interval",
        fontsize=8.6, fontweight="bold", color=INK, va="top", zorder=9)
ax3 = fig.add_axes(frac(hx + 44, hy + 16, hw - 120, hh - 62))
ax3.set_facecolor("none")
cor = bell(95, 6.35, 1.58, 8, 0.045)
band = 7 + 11 * np.exp(-0.5 * ((t - 6.0) / 2.4) ** 2) + 0.06 * cor
ax3.fill_between(t, np.clip(cor - band, 0, None), cor + band, color=BAND, alpha=0.22, lw=0)
ax3.plot(t, obs, color=INK, lw=1.6, ls=(0, (4, 2)))
ax3.plot(t, cor, color="white", lw=4.5, alpha=0.7, solid_capstyle="round")
ax3.plot(t, cor, color=TEAL, lw=2.4, solid_capstyle="round")
ax3.text(0.4, 112, "corrected", fontsize=7.0, color=TEAL, ha="left", va="top",
         fontweight="bold")
ax3.text(11.7, 26, "observed", fontsize=7.0, color=INK, ha="right", va="center")
ax3.text(9.3, cor[310] + band[310] + 5, "90% interval", fontsize=6.8,
         color="#0E7F79", ha="left", va="bottom")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax3.spines[s].set_color("#BBC9C8")
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 128)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel(f"Discharge ({UNIT})", fontsize=6.6, color=GREY, labelpad=2)

# headline KGE' gain
ky = 700
add_round(ax, p2 + 38, ky - 56, PW - 76, 112, 16, fc=PALE, ec=TEAL, lw=1.7, zorder=6)
ax.text(PCX[2], ky + 36, f"Median {KGE}  {DOT}  all 74 gauges skilful",
        fontsize=9.6, fontweight="bold", color=TEAL_DARK, va="center",
        ha="center", zorder=9)
ax.text(PCX[2], ky - 12, f"0.386 {ARROW} 0.825", fontsize=25, fontweight="bold",
        color=TEAL, va="center", ha="center", zorder=9)

# findings checklist
finds = [
    ["Temporal skill statistically tied",
     "with gradient boosting (honest parity)"],
    ["Calibrated 90% intervals after",
     f"conformal (coverage 0.68 {ARROW} 0.90)"],
    ["Ungauged transfer (PUR): an honest,",
     "physically-attributed transfer limit"],
]
fy = [582, 432, 282]
for lines, yc in zip(finds, fy):
    ic_check(ax, p2 + 68, yc, 26)
    ax.text(p2 + 108, yc + 16, lines[0], va="center", ha="left", fontsize=9.2,
            fontweight="bold", color=INK, zorder=9)
    ax.text(p2 + 108, yc - 16, lines[1], va="center", ha="left", fontsize=8.6,
            color=GREY, zorder=9)

# deployable footer band
fyb = 150
add_round(ax, p2 + 38, PBOT + 22, PW - 76, fyb - 22, 16, fc=TEAL_DARK, ec="none", zorder=6)
gradf = (np.array(to_rgb(TEAL_DARK))[None] +
         (np.array(to_rgb(GREEN)) - np.array(to_rgb(TEAL_DARK)))[None] *
         np.linspace(0, 1, 256)[:, None])[None]
fpatch = add_round(ax, p2 + 38, PBOT + 22, PW - 76, fyb - 22, 16, fc="none",
                   ec="none", zorder=6)
imf = ax.imshow(gradf, extent=[p2 + 38, p2 + PW - 38, PBOT + 22, PBOT + fyb],
                aspect="auto", zorder=6, interpolation="bilinear")
imf.set_clip_path(fpatch)
ax.text(PCX[2], PBOT + (22 + fyb) / 2 + 12, "Deployable by a national",
        ha="center", va="center", color="white", fontsize=11.5,
        fontweight="bold", zorder=9)
ax.text(PCX[2], PBOT + (22 + fyb) / 2 - 16, "hydrometeorological service",
        ha="center", va="center", color="#EAFBF7", fontsize=11.5,
        fontweight="bold", zorder=9)

# --------------------------------------------------------------------------- #
#  Save
# --------------------------------------------------------------------------- #
try:
    from sbc.viz import style
    out_dir = style.PUB_DIR
except Exception:
    from pathlib import Path as _P
    out_dir = _P(r"G:/MDPI Q1-2026/results/figures/publication")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "graphical_abstract.png"
fig.savefig(out, dpi=300, facecolor="white")
plt.close(fig)

from PIL import Image
im = Image.open(out).convert("RGB")
im.save(out, dpi=(300, 300))
print("saved", out, "size (WxH px):", im.size)
