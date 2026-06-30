"""Publication figures and tables from a finished experiment's result tables.

Reads ``results/tables/per_gauge_<tag>.parquet`` (default tag ``real_decadal``)
and writes figures to ``results/figures`` and a Markdown summary to
``results/tables``.

Usage::
    PYTHONPATH=src python scripts/make_figures.py --tag real_decadal
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sbc.config import PATHS
from sbc.utils import get_logger

log = get_logger("figures")
MODEL_ORDER = ["scaling", "qmap", "xgb", "lgbm", "catboost", "ealstm",
               "regimeprobnet", "stacked"]


def _ordered(models):
    present = [m for m in MODEL_ORDER if m in set(models)]
    return present + [m for m in sorted(set(models)) if m not in present]


def fig_kge_by_split(res: pd.DataFrame, out):
    splits = [s for s in ["temporal", "lobo", "pur"] if s in res["split"].unique()]
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        sub = res[res.split == split]
        models = _ordered(sub.model.unique())
        data = [sub[sub.model == m]["kge"].dropna().values for m in models]
        ax.boxplot(data, labels=models, vert=False, showfliers=False)
        ax.axvline(sub["kge_raw"].median(), color="crimson", ls="--", lw=1.5,
                   label="raw GloFAS (median)")
        ax.set_title(f"KGE′ — {split}")
        ax.set_xlim(-0.6, 1.0); ax.grid(alpha=0.3, axis="x"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    log.info("wrote %s", out)


def fig_improvement_map(res: pd.DataFrame, gauges: pd.DataFrame, out, model="lgbm"):
    sub = res[(res.model == model) & (res.split == "temporal")].copy()
    sub["d_kge"] = sub["kge"] - sub["kge_raw"]
    m = sub.merge(gauges[["code", "lon", "lat", "basin"]].drop_duplicates("code"),
                  on="code", how="left").dropna(subset=["lon", "lat"])
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(m.lon, m.lat, c=m.d_kge, cmap="RdYlGn", vmin=-0.3, vmax=0.6,
                    s=60, edgecolor="k", linewidth=0.4)
    fig.colorbar(sc, label=f"ΔKGE′ ({model} − raw GloFAS)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Per-gauge skill improvement (temporal holdout)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    log.info("wrote %s", out)


def fig_decomposition(res: pd.DataFrame, out, model="regimeprobnet"):
    sub = res[(res.model == model) & (res.split == "temporal")]
    if sub.empty:
        return
    comps = [("kge_r", "kge_r_raw", "r"), ("kge_beta", "kge_beta_raw", "β"),
             ("kge_gamma", "kge_gamma_raw", "γ")]
    comps = [c for c in comps if c[0] in sub.columns and c[1] in sub.columns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = np.arange(len(comps))
    for i, (cc, rc, lab) in enumerate(comps):
        ax.boxplot(sub[rc].dropna(), positions=[i - 0.18], widths=0.3, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="lightgray"))
        ax.boxplot(sub[cc].dropna(), positions=[i + 0.18], widths=0.3, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="mediumseagreen"))
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(pos); ax.set_xticklabels([c[2] for c in comps])
    ax.set_title(f"KGE′ components: raw (grey) vs {model} (green)")
    ax.set_ylabel("component value (ideal = 1)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    log.info("wrote %s", out)


def fig_shap(out):
    p = PATHS.shap_dir / "global_importance.parquet"
    if not p.exists():
        return
    g = pd.read_parquet(p).head(15).iloc[::-1]
    col = "mean_abs_shap" if "mean_abs_shap" in g.columns else g.columns[-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(g.iloc[:, 0].astype(str), g[col], color="steelblue")
    ax.set_xlabel("mean |SHAP|"); ax.set_title("Global feature importance (top 15)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real_decadal")
    a = ap.parse_args()
    res = pd.read_parquet(PATHS.tables / f"per_gauge_{a.tag}.parquet")
    gauges = pd.read_parquet(PATHS.processed / "gauges.parquet")

    fig_kge_by_split(res, PATHS.figures / f"kge_by_split_{a.tag}.png")
    fig_improvement_map(res, gauges, PATHS.figures / f"improvement_map_{a.tag}.png")
    fig_decomposition(res, PATHS.figures / f"kge_decomposition_{a.tag}.png")
    fig_shap(PATHS.figures / f"shap_importance_{a.tag}.png")

    from sbc.validation.cv import summarise
    summ = summarise(res)
    try:
        md = summ.to_markdown(index=False)
    except Exception:  # tabulate not installed
        md = summ.to_string(index=False)
    (PATHS.tables / f"summary_{a.tag}.md").write_text(md, encoding="utf-8")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
