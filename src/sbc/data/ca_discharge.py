"""Extraction of ground-truth discharge from the CA-discharge GeoPackage.

CA-discharge (Marti, Siegfried, Yakovlev, Karger, Ragettli et al., 2023,
*Scientific Data* 10:579, doi:10.1038/s41597-023-02474-8; Zenodo 8147591) is a
consolidated, quality-controlled archive of Central-Asian river-discharge
observations.  The single ``CA-discharge.gpkg`` container holds:

* ``gauges``                gauge metadata + point geometry (297 stations)
* ``basins``                contributing-area polygons (295)
* ``basin_attributes``      ~1100 HydroATLAS-style static attributes
* ``discharge_time_series`` long-format series, column ``res`` in
                            {``day``, ``decade``, ``month``}
* ``quality_flags``         per-gauge QC flags

This module turns that container into tidy, analysis-ready tables restricted to
the study domain (see :mod:`sbc.data.domain`) and to the GloFAS overlap period.

Run as a script::

    PYTHONPATH=src python -m sbc.data.ca_discharge
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger, save_json, save_table
from . import domain as D

log = get_logger(__name__)

# CA-discharge ``res`` label -> framework scale name
RES_TO_SCALE = {"decade": "decadal", "day": "daily", "month": "monthly"}
SCALE_TO_RES = {v: k for k, v in RES_TO_SCALE.items()}

# Curated static attributes (subset of the ~1100 available) that are physically
# relevant for snow-influenced bias correction.  Substring match, case-insensitive.
STATIC_ATTR_KEYS = (
    "area", "ele", "elev", "slp", "slope", "snow", "swe", "glac", "ice",
    "ari", "aridity", "pre", "prec", "tmp", "temp", "for", "forest",
    "soil", "perm", "lake", "rev", "clz", "clay", "snd", "sand", "lat", "lon",
)


def _gpkg_path() -> Path:
    p = PATHS.ca_discharge / "CA-discharge.gpkg"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it from Zenodo (record 8147591) into "
            f"{PATHS.ca_discharge} — see scripts/download_ca_discharge.py."
        )
    return p


def load_tables() -> dict[str, pd.DataFrame]:
    """Load the tabular layers of the GeoPackage with sqlite3."""
    path = _gpkg_path()
    with sqlite3.connect(path) as con:
        gauges = pd.read_sql("SELECT * FROM gauges", con)
        ts = pd.read_sql("SELECT CODE, res, date, value FROM discharge_time_series", con)
        attrs = pd.read_sql("SELECT * FROM basin_attributes", con)
        quality = pd.read_sql("SELECT * FROM quality_flags", con)
        try:
            attr_desc = pd.read_sql("SELECT * FROM basin_attribute_description", con)
        except Exception:
            attr_desc = pd.DataFrame()
    for df in (gauges, attrs, quality):
        df.drop(columns=[c for c in ("geom", "fid") if c in df.columns],
                inplace=True, errors="ignore")
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    log.info("Loaded gpkg: %d gauges, %d ts rows, %d attribute cols",
             len(gauges), len(ts), attrs.shape[1])
    return {"gauges": gauges, "ts": ts, "attrs": attrs,
            "quality": quality, "attr_desc": attr_desc}


def availability(ts: pd.DataFrame, period_start: str) -> pd.DataFrame:
    """Per (CODE, res) valid-record count and span within the overlap period."""
    start = pd.Timestamp(period_start)
    post = ts[(ts["date"] >= start) & ts["value"].notna()]
    agg = (post.groupby(["CODE", "res"])
           .agg(n_valid=("value", "size"),
                ts_start=("date", "min"),
                ts_end=("date", "max"))
           .reset_index())
    agg["years"] = (agg["ts_end"] - agg["ts_start"]).dt.days / 365.25
    return agg


def select_gauges(gauges: pd.DataFrame, avail: pd.DataFrame,
                  scale: str, min_years: float) -> pd.DataFrame:
    """Study-domain gauges meeting the minimum-length criterion at ``scale``."""
    res = SCALE_TO_RES[scale]
    sub = avail[(avail["res"] == res) & (avail["years"] >= min_years)]
    g = gauges.merge(sub, on="CODE", how="inner")
    g["domain"] = g["BASIN"].map(D.domain_of)
    g = g[g["domain"].notna()].copy()
    g["scale"] = scale
    return g


def _tidy_meta(g: pd.DataFrame) -> pd.DataFrame:
    keep = {
        "CODE": "code", "RIVER": "river", "BASIN": "basin", "COUNTRY": "country",
        "LON": "lon", "LAT": "lat", "NAME_ENG": "name", "q_m3s": "q_mean_ref",
        "domain": "domain", "scale": "scale", "n_valid": "n_valid",
        "ts_start": "ts_start", "ts_end": "ts_end", "years": "years",
    }
    out = g[[c for c in keep if c in g.columns]].rename(columns=keep)
    out["code"] = out["code"].astype(str)
    out["basin"] = out["basin"].astype(str).str.upper()
    return out.reset_index(drop=True)


def _tidy_timeseries(ts: pd.DataFrame, codes: list[str], scale: str,
                     period_start: str, period_end: str) -> pd.DataFrame:
    res = SCALE_TO_RES[scale]
    start, end = pd.Timestamp(period_start), pd.Timestamp(period_end)
    sub = ts[(ts["res"] == res) & ts["CODE"].isin(codes)
             & (ts["date"] >= start) & (ts["date"] <= end) & ts["value"].notna()]
    out = (sub[["CODE", "date", "value"]]
           .rename(columns={"CODE": "code", "value": "q_obs"})
           .sort_values(["code", "date"]).reset_index(drop=True))
    out["code"] = out["code"].astype(str)
    out["q_obs"] = out["q_obs"].astype(float)
    return out


def _select_static(attrs: pd.DataFrame, codes: list[str],
                   min_coverage: float = 0.5) -> pd.DataFrame:
    """Return every numeric static attribute with sufficient coverage.

    The CA-discharge ``basin_attributes`` table carries ~1100 columns, including
    monthly climatologies of precipitation (``pr_*``), air temperature
    (``tas_*``) and MODIS snow-cover fraction (``scf_*``), plus terrain, glacier
    (``gl_*``), bioclimatic (``bio*``) and growing-degree-day (``gdd*``)
    indices.  All numeric attributes are kept here; the feature layer
    (:mod:`sbc.features`) collapses the monthly groups into compact,
    physically-meaningful seasonal descriptors before modelling.
    """
    code_col = "CODE" if "CODE" in attrs.columns else attrs.columns[0]
    drop = {"fid", "EASTING", "NORTHING", "SOURCE", "REGION", "BASIN", "NAME_ENG"}
    sub = attrs[attrs[code_col].astype(str).isin(codes)].copy()
    code_series = sub[code_col].astype(str).values
    feat = sub.drop(columns=[c for c in drop | {code_col} if c in sub.columns],
                    errors="ignore")
    num = feat.apply(pd.to_numeric, errors="coerce")
    keep = num.columns[num.notna().mean() > min_coverage]
    num = num[keep].copy()
    num.insert(0, "code", code_series)
    return num.reset_index(drop=True)


def extract() -> dict:
    """Run the full extraction and persist analysis-ready tables."""
    PATHS.ensure()
    t = load_tables()
    avail = availability(t["ts"], D.PERIOD["start"])

    meta_frames, ts_frames = [], {}
    summary: dict = {"period": D.PERIOD, "scales": {}}

    for scale, min_years in D.MIN_YEARS.items():
        g = select_gauges(t["gauges"], avail, scale, min_years)
        if g.empty:
            log.warning("No gauges for scale=%s", scale)
            continue
        meta = _tidy_meta(g)
        codes = meta["code"].tolist()
        tsd = _tidy_timeseries(t["ts"], codes, scale,
                               D.PERIOD["start"], D.PERIOD["end"])
        ts_frames[scale] = tsd
        meta_frames.append(meta)

        save_table(tsd, PATHS.processed / f"discharge_{scale}.parquet", csv_mirror=True)
        by_dom = meta.groupby("domain")["code"].nunique().to_dict()
        by_basin = meta.groupby("basin")["code"].nunique().to_dict()
        summary["scales"][scale] = {
            "n_gauges": int(meta["code"].nunique()),
            "n_records": int(len(tsd)),
            "by_domain": by_dom,
            "by_basin": by_basin,
            "median_years": float(meta["years"].median()),
        }
        log.info("scale=%-8s gauges=%-3d records=%-6d basins=%d",
                 scale, meta["code"].nunique(), len(tsd), meta["basin"].nunique())

    meta_all = (pd.concat(meta_frames, ignore_index=True)
                .sort_values(["scale", "domain", "basin", "code"]))
    save_table(meta_all, PATHS.processed / "gauges.parquet", csv_mirror=True)

    all_codes = meta_all["code"].unique().tolist()
    static = _select_static(t["attrs"], all_codes)
    save_table(static, PATHS.processed / "static_attributes.parquet", csv_mirror=True)
    summary["n_static_attributes"] = int(static.shape[1] - 1)

    if not t["attr_desc"].empty:
        save_table(t["attr_desc"], PATHS.processed / "attribute_descriptions.parquet")

    # quality flags for the retained gauges
    q = t["quality"].copy()
    if "CODE" in q.columns:
        q = q.rename(columns={"CODE": "code"})
        q["code"] = q["code"].astype(str)
        q = q[q["code"].isin(all_codes)]
        save_table(q, PATHS.processed / "quality_flags.parquet", csv_mirror=True)

    save_json(summary, PATHS.processed / "domain_summary.json")
    log.info("Static attributes retained: %d", summary["n_static_attributes"])
    log.info("Wrote processed tables to %s", PATHS.processed)
    return summary


def export_basin_polygons() -> Path | None:
    """Save study-domain basin polygons as GeoParquet (needs geopandas)."""
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        log.warning("geopandas unavailable (%s); skipping basin polygons", exc)
        return None
    gdf = gpd.read_file(_gpkg_path(), layer="basins")
    gdf["BASIN"] = gdf["BASIN"].astype(str).str.upper()
    gdf = gdf[gdf["BASIN"].isin(D.ALL_STUDY_BASINS)].copy()
    gdf["domain"] = gdf["BASIN"].map(D.domain_of)
    out = PATHS.processed / "basins.parquet"
    gdf.to_parquet(out)
    log.info("Saved %d study basin polygons to %s", len(gdf), out)
    return out


def main() -> None:
    summary = extract()
    export_basin_polygons()
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
