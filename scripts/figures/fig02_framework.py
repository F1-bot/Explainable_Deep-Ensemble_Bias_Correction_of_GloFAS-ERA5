"""Figure 02 - Methodological framework (schematic flowchart).

Left-to-right explainable deep-ensemble bias-correction pipeline for GloFAS-ERA5
daily/decadal discharge in snow-influenced transboundary Central Asia.

Built modularly from two helpers - rounded_block() (FancyBboxPatch) and
connect() (FancyArrowPatch, arrowstyle "-|>") - composed into four labelled
column groups: DATA & FEATURES | TARGET DEFINITION | MODELING FRAMEWORK |
EVALUATION.  Solid arrows = main data flow; dashed green = regime-gating
feedback.  Drawn through the shared MDPI style.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"G:/MDPI Q1-2026/src")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sbc.viz import style

style.apply()

# --- restrained colour-blind-safe scheme grouping the pipeline stages ---------
C_DATA = {"face": "#DCE9F5", "edge": "#0072B2", "title": "#0a3d62"}   # data (blue)
C_FEAT = {"face": "#E5F0FA", "edge": "#0072B2", "title": "#0a3d62"}   # features (blue)
C_REG = {"face": "#D7EFE7", "edge": "#009E73", "title": "#0b5345"}    # target / regime (green)
C_MODEL = {"face": "#FDF1DC", "edge": "#E69F00", "title": "#7a4f00"}  # model zoo (amber)
C_CAND = {"face": "#FBE7C8", "edge": "#E69F00", "title": "#7a4f00"}   # candidate sub-box
# flagship: toned-down light fill + normal border (no longer reads as superiority);
# edge keyed to the set-wide flagship hue so the colour means ONE thing everywhere.
C_FLAG = {"face": "#FBE0C7", "edge": style.SERIES_COLORS["flagship"],
          "title": "#7a2e00"}                                        # flagship (#D55E00)
C_EVAL = {"face": "#E7DEF1", "edge": "#7B5EA7", "title": "#3f2a63"}   # evaluation (purple)
# stacked ensemble: distinct modeling/integration colour keyed to the set-wide
# "stacked" hue (sky-blue) - was wrongly the same purple the key labels Evaluation.
C_ENS = {"face": "#CFE8F8", "edge": style.SERIES_COLORS["stacked"],
         "title": "#15547a"}                                         # integration (#56B4E9)

INK = "#262626"

# === canvas: equal units/inch (20 per inch) so rounded corners stay circular ==
fig, ax = plt.subplots(figsize=(style.WIDTH_2COL, 4.65))
ax.set_xlim(0, 144)
ax.set_ylim(0, 93)
ax.axis("off")
fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)


# === modular helpers =========================================================
def rounded_block(x, y, w, h, scheme, *, title=None, body=None, dashed=False,
                  lw=1.2, rounding=2.4, title_fs=8.0, body_fs=6.8, z=2,
                  face=None, edge=None, body_color=None, body_it=False,
                  title_top=True, body_dy=0.0):
    """One rounded, colour-grouped block with an optional bold title and a
    centred (multi-line) body string."""
    ls = (0, (5, 2.5)) if dashed else "-"
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge or scheme["edge"],
        facecolor=face or scheme["face"], linestyle=ls,
        mutation_aspect=1.0, zorder=z,
    )
    ax.add_patch(p)
    cx = x + w / 2
    if title is not None:
        ty = (y + h - 3.0) if title_top else (y + h / 2)
        ax.text(cx, ty, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=scheme["title"], zorder=z + 3)
    if body is not None:
        by = (y + h / 2 - 1.0 + body_dy) if title_top else (y + h / 2 + body_dy)
        ax.text(cx, by, body, ha="center", va="center", fontsize=body_fs,
                color=body_color or INK, zorder=z + 3, linespacing=1.35,
                style="italic" if body_it else "normal")
    return p


def bullet_list(x_left, y_top, items, fs, color=INK, line_h=2.6, gap=1.4, z=6):
    """Left-aligned bulleted list; continuation lines (\\n) are indented."""
    y = y_top
    for it in items:
        n = it.count("\n") + 1
        ax.text(x_left, y, "•  " + it.replace("\n", "\n     "),
                ha="left", va="top", fontsize=fs, color=color,
                zorder=z, linespacing=1.2)
        y -= n * line_h + gap


def connect(a, b, *, color="#5a5a5a", lw=1.7, dashed=False, rad=0.0, z=1,
            mscale=13):
    ls = (0, (5, 2.5)) if dashed else "-"
    p = FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=mscale,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2.5, shrinkB=2.5, zorder=z,
    )
    ax.add_patch(p)


def col_header(x0, x1, text, color, y=84.5):
    cx = (x0 + x1) / 2
    ax.text(cx, y, text, ha="center", va="center", fontsize=8.4,
            fontweight="bold", color=color, zorder=5)
    ax.plot([x0, x1], [y - 3.0, y - 3.0], color=color, lw=1.1,
            solid_capstyle="round", zorder=4)


# === figure title ============================================================
ax.text(72, 90.5, "Regime-gated hydrological modeling framework",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
ax.text(72, 87.0,
        "Flagship is tied with LightGBM on point skill; its contribution is "
        "calibration + interpretability, not point-skill dominance.",
        ha="center", va="center", fontsize=6.6, style="italic", color="#555555")

# === column-group headers ====================================================
col_header(3, 38, "DATA & FEATURES", C_DATA["title"])
col_header(42, 59, "TARGET DEFINITION", C_REG["title"])
col_header(63, 115, "MODELING FRAMEWORK", C_MODEL["title"])
col_header(119, 142, "EVALUATION", C_EVAL["title"])

YC = 44  # main-spine vertical centre

# === (1) DATA & FEATURES column ==============================================
# data sources
rounded_block(3, 55, 35, 25, C_DATA, title="Data sources", lw=1.3)
bullet_list(5.5, 71, [
    "GloFAS-ERA5 v4 discharge",
    "ERA5-Land snow & meteo",
    "CA discharge observations",
    "HydroATLAS attributes",
], fs=7.0, color="#0a3d62", line_h=2.6, gap=1.5)

# feature engineering
rounded_block(3, 28, 35, 24, C_FEAT, title="Feature engineering", lw=1.3)
bullet_list(5.5, 44, [
    "GloFAS-memory lags",
    "SWE / melt, degree-days",
    "Rain-on-snow flags",
    "Seasonality",
], fs=7.0, color="#0a3d62", line_h=2.6, gap=1.5)

# hydrological regime classifier (green - it drives the gate)
rounded_block(3, 11, 35, 14, C_REG, title=None, lw=1.3,
              body="Hydrological regime\nclassifier", title_top=False,
              body_fs=7.2, body_color=C_REG["title"])

# === (2) TARGET DEFINITION column ============================================
rounded_block(42, 35, 17, 18, C_REG, lw=1.5)
ax.text(50.5, 49, "Target", ha="center", va="center", fontsize=8.0,
        fontweight="bold", color=C_REG["title"], zorder=5)
ax.text(50.5, 43.6, "log-residual", ha="center", va="center", fontsize=6.6,
        color=C_REG["title"], zorder=5)
ax.text(50.5, 39.4,
        r"log $q_\mathregular{obs}$ " + style.MINUS + r" log $q_\mathregular{sim}$",
        ha="center", va="center", fontsize=7.0, color=C_REG["title"], zorder=5)

# === (3) MODELING FRAMEWORK column ===========================================
# model zoo container
rounded_block(63, 10, 35, 70, C_MODEL, lw=1.5, rounding=2.8)
ax.text(80.5, 76.5, "Model zoo", ha="center", va="center", fontsize=8.4,
        fontweight="bold", color=C_MODEL["title"], zorder=5)

# candidate models (nested)
rounded_block(65, 47, 31, 26, C_CAND, title="Candidate models", lw=1.0,
              rounding=1.8, title_fs=7.2, z=3)
bullet_list(67.5, 67, [
    "Quantile-mapping &\nscaling baselines",
    "XGBoost / LightGBM /\nCatBoost",
    "EA-LSTM",
], fs=6.8, color="#5a3a00", line_h=2.7, gap=1.4, z=6)

# flagship (nested) - normal border weight, matching the model-zoo family
rounded_block(64, 12, 33, 33, C_FLAG, lw=1.3, rounding=2.0, z=3)
ax.text(80.5, 41, "RegimeProbNet (flagship)", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color=C_FLAG["title"], zorder=6)
bullet_list(66.5, 36, [
    "regime-gated mixture-of-experts",
    "Gaussian / CMAL CRPS heads",
    "soft & hard SWE–T monotonicity",
], fs=6.6, color="#6b2600", line_h=2.7, gap=1.6, z=6)

# leakage-safe stacked ensemble (integration step -> feeds Validation only)
rounded_block(101, 52.5, 14, 21, C_ENS, lw=1.5)
ax.text(108, 69.6, "Leakage-safe", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=C_ENS["title"], zorder=5)
ax.text(108, 66.3, "stacked", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=C_ENS["title"], zorder=5)
ax.text(108, 63.0, "ensemble", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=C_ENS["title"], zorder=5)
ax.text(108, 57.6, "out-of-fold\nmeta-learner", ha="center", va="center",
        fontsize=6.8, color=C_ENS["title"], zorder=5, linespacing=1.3)

# === (4) EVALUATION column ===================================================
eval_specs = [
    (66, "Validation", "temporal · LOBO · PUR"),
    (42, "Uncertainty", "CRPS, conformal\n(regime-conditional),\ncalibration"),
    (18, "Explainability", "SHAP / IG / ALE,\nregime gates"),
]
eval_cy = []
for cy, t, body in eval_specs:
    rounded_block(119, cy - 9, 23, 18, C_EVAL, title=t, lw=1.2, title_fs=7.4)
    ax.text(130.5, cy - 3.4, body, ha="center", va="center", fontsize=6.8,
            color=C_EVAL['title'], zorder=5, linespacing=1.35)
    eval_cy.append(cy)

# === arrows: solid main data-flow spine ======================================
connect((20.5, 55), (20.5, 52))                       # data -> features
connect((38, 41), (42, 44), rad=0.06)                 # features -> target
connect((59, 44), (63, 44))                           # target -> model zoo
# main spine: model zoo -> stacked ensemble -> Validation (the integrated prediction)
connect((98, 63), (101, 63))                          # model zoo -> stacked ensemble
connect((115, 63), (119, eval_cy[0]), rad=-0.05, lw=1.45, mscale=11)  # ensemble -> Validation
# UQ + XAI branch from the FLAGSHIP / model zoo (CRPS/conformal/SHAP are model
# properties, NOT meta-learner outputs); coloured to read as flagship-sourced.
connect((97, 39), (119, eval_cy[1]), color=C_FLAG["edge"], rad=-0.05,
        lw=1.45, mscale=11)                           # flagship -> Uncertainty
connect((97, 22.5), (119, eval_cy[2]), color=C_FLAG["edge"], rad=0.08,
        lw=1.45, mscale=11)                           # flagship -> Explainability

# === dashed green regime-gating feedback -> flagship =========================
connect((38, 19), (64, 20), color=C_REG["edge"], lw=1.4, dashed=True,
        rad=-0.10, z=4, mscale=12)
ax.text(51, 23.5, "Regime gating", ha="center", va="bottom", fontsize=6.6,
        color=C_REG["title"], style="italic", zorder=6)

# === integrated colour key (slim caption-row, not a detached box) ============
key = [("Data", C_DATA), ("Features", C_FEAT), ("Target / regime", C_REG),
       ("Model zoo", C_MODEL), ("Flagship", C_FLAG),
       ("Stacked ensemble", C_ENS), ("Evaluation", C_EVAL)]
ax.plot([3, 142], [7.0, 7.0], color="#cccccc", lw=0.8, zorder=1)
ax.text(3, 4.2, "Key:", ha="left", va="center", fontsize=6.6,
        fontweight="bold", color="#555555", zorder=6)
kx = 11.0
for name, sc in key:
    rounded_block(kx, 2.7, 3.2, 3.2, sc, lw=0.9, rounding=0.8, z=5)
    ax.text(kx + 4.0, 4.3, name, ha="left", va="center", fontsize=6.5,
            color="#3a3a3a", zorder=6)
    kx += 4.5 + 1.3 * len(name)

style.savefig(fig, "fig02_framework")
print("saved fig02_framework")
