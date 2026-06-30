"""Shared publication style for every figure — the consistency backbone.

Targets the MDPI *Water* figure standard: >= 600 dpi PNG, RGB color, English labels,
correct math symbols (use '-' not the em dash), a comma thousands-separator for
numbers with five or more digits, short axis labels with units, and (a)/(b) panel
tags. Import :func:`apply` first in every figure script and save with :func:`savefig`
so all figures share fonts, sizes, palette, model colours/order and DPI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

from ..config import PATHS

# --- canonical layout widths (MDPI single / 1.5 / double column, inches) -----
# MDPI Water text block is 170 mm; keep full-width <= 6.69 in so figures place
# at 100% without downscaling (audit C7).
WIDTH_1COL = 3.35     # ~85 mm
WIDTH_15COL = 5.00    # ~127 mm
WIDTH_2COL = 6.69     # ~170 mm (full text-block width)
DPI = 600

PUB_DIR = PATHS.figures / "publication"

# --- colour-blind-safe categorical palette (Okabe-Ito + extensions) ----------
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000", "#999999", "#882255", "#44AA99"]

# --- canonical model display order, pretty labels and stable colours ---------
MODEL_ORDER = ["raw", "scaling", "qmap", "donor", "glofas_lstm", "qrf",
               "xgb", "lgbm", "catboost", "ealstm", "regimeprobnet",
               "deepens", "stacked", "cmal", "gen_resid", "diffsnow",
               "probnet_hardmono", "transfer_lstm"]
MODEL_LABELS = {
    "raw": "Raw GloFAS-ERA5", "scaling": "Linear scaling", "qmap": "Quantile mapping",
    "donor": "Donor (SABER-style)", "glofas_lstm": "LSTM-on-GloFAS (Hunt-style)",
    "qrf": "QRF", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost",
    "ealstm": "EA-LSTM", "regimeprobnet": "RegimeProbNet (flagship)",
    "deepens": "Deep ensemble", "stacked": "Stacked ensemble",
    "cmal": "CMAL head", "gen_resid": "Generative IQN", "diffsnow": "Diff. snow",
    "probnet_hardmono": "Hard-monotonic", "transfer_lstm": "Transfer-LSTM",
}
SPLIT_LABELS = {"temporal": "Temporal holdout", "lobo": "Leave-one-basin-out",
                "pur": "Prediction in ungauged regions"}

# --- hydrological-regime colours (snow narrative) ----------------------------
REGIME_ORDER = ["accumulation", "melt_freshet", "rain_on_snow", "glacier_melt", "recession"]
REGIME_LABELS = {"accumulation": "Accumulation", "melt_freshet": "Melt freshet",
                 "rain_on_snow": "Rain-on-snow", "glacier_melt": "Glacier melt",
                 "recession": "Recession"}
REGIME_COLORS = {"accumulation": "#56B4E9", "melt_freshet": "#0072B2",
                 "rain_on_snow": "#009E73", "glacier_melt": "#D55E00",
                 "recession": "#999999"}
DOMAIN_COLORS = {"core": "#0072B2", "transfer": "#D55E00"}

# --- fixed colour grammar so a hue means ONE thing set-wide (audit C3/C4) -----
# Validation splits: use these identically in every panel of every figure.
SPLIT_COLORS = {"temporal": "#0072B2", "lobo": "#009E73", "pur": "#D55E00"}
# Series identity: observed / raw reference / generic corrected / the flagship /
# named UQ baselines. Name the plotted model in the caption (audit C9).
SERIES_COLORS = {"observed": "#000000", "raw": "#9A9A9A", "corrected": "#0072B2",
                 "flagship": "#D55E00", "qrf": "#009E73", "cmal": "#CC79A7",
                 "lgbm": "#E69F00", "stacked": "#56B4E9"}

# Metric glyph: bonded prime (U+2032), matching the manuscript body (audit C5).
KGE_PRIME = "KGE′"

_HIGHLIGHT = "#D55E00"  # the flagship / key series accent


def apply() -> None:
    """Set global rcParams for the MDPI Water figure standard."""
    mpl.use("Agg", force=True)
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        # Concrete family list (not the generic "sans-serif" alias) so matplotlib
        # builds a per-glyph fallback chain: Arial stays primary for every glyph,
        # DejaVu Sans only supplies glyphs Arial lacks (e.g. U+207B superscript
        # minus in UNIT_Q "m^3 s^-1"), avoiding tofu boxes.
        "font.family": ["Arial", "DejaVu Sans"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
        "grid.alpha": 0.8, "axes.axisbelow": True,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "figure.autolayout": False,
        "axes.unicode_minus": True,   # ticks use U+2212; annotations must match (style.num)
        "mathtext.default": "regular", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# --- typography helpers: keep negative numbers + units consistent ------------
MINUS = "−"                 # typographic minus (U+2212), matches axis ticks
ARROW = "→"                 # right arrow, use instead of '->'
UNIT_Q = "m³ s⁻¹"  # m^3 s^-1 via unicode super/subscripts (low, even baseline)


def num(x, dec: int = 2, pct: bool = False, thousands: bool = True) -> str:
    """Format a number with the typographic minus (U+2212) and a comma thousands
    separator, so text annotations match matplotlib's axis tick labels."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    neg = x < 0
    v = abs(float(x))
    s = (f"{v:,.{dec}f}" if thousands else f"{v:.{dec}f}")
    if pct:
        s += "%"
    return (MINUS + s) if neg else s


def discharge_label(prefix: str = "Discharge") -> str:
    """Axis label like 'Discharge (m^3 s^-1)' with clean unicode superscripts."""
    return f"{prefix} ({UNIT_Q})"


# --- real map geometry (Natural Earth admin-0 country polygons) --------------
NE_50M = PATHS.datasets / "naturalearth" / "ne_50m_admin_0_countries.geojson"
NE_110M = PATHS.datasets / "naturalearth" / "ne_110m_admin_0_countries.geojson"


def model_color(name: str) -> str:
    try:
        return PALETTE[MODEL_ORDER.index(name) % len(PALETTE)]
    except ValueError:
        return "#666666"


def order_models(models) -> list[str]:
    s = set(models)
    return [m for m in MODEL_ORDER if m in s] + [m for m in sorted(s) if m not in MODEL_ORDER]


def label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def thousands(x, _pos=None) -> str:
    """Tick formatter: comma thousands-separator (MDPI: numbers with >= 5 digits)."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    if abs(x) >= 10000:
        return f"{x:,.0f}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:g}"


def thousands_formatter():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(thousands)


def panel(ax, letter: str, x: float = -0.16, y: float = 1.04, **kw) -> None:
    """Bold panel tag, e.g. (a), in axes-fraction coords."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", **kw)


def savefig(fig, name: str, formats=("png",)) -> list[Path]:
    """Save a figure to results/figures/publication/<name>.<fmt> at 600 dpi.

    PNGs are flattened RGBA -> RGB on white (MDPI wants RGB, no stray alpha; audit C6).
    """
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = PUB_DIR / f"{name}.{fmt}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
        if fmt == "png":
            from PIL import Image
            im = Image.open(p)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.split()[-1])
                bg.save(p, dpi=(DPI, DPI))
        out.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out
