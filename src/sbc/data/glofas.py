"""GloFAS-ERA5 historical reanalysis: download and per-gauge pixel extraction.

Predictor product to be bias-corrected.  Source: Copernicus Early Warning Data
Store, dataset ``cems-glofas-historical`` (v4.0, 0.05 deg), variable
``river_discharge_in_the_last_24_hours`` (LISFLOOD forced by ERA5/HTESSEL).

Download is per hydrological year and resumable.  Pixel extraction picks, within
a small search window around each gauge, the river cell whose GloFAS upstream
area best matches the gauge's contributing area (falling back to the
highest-discharge neighbour), the standard way to avoid snapping to a tributary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger, save_table
from . import domain as D
from ._cds import make_client, retrieve_with_retry

log = get_logger(__name__)

DATASET = "cems-glofas-historical"
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def _area(bbox: dict) -> list[float]:
    return [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]


def download_glofas(years: list[int] | None = None,
                    bbox: dict | None = None,
                    outdir: Path | None = None,
                    system_version: str = "version_4_0",
                    data_format: str = "netcdf",
                    test: bool = False) -> list[Path]:
    """Download GloFAS daily discharge, one NetCDF per hydrological year."""
    bbox = bbox or D.STUDY_BBOX
    outdir = Path(outdir or PATHS.glofas)
    outdir.mkdir(parents=True, exist_ok=True)
    if years is None:
        y0 = int(D.PERIOD["start"][:4])
        y1 = int(D.PERIOD["end"][:4])
        years = list(range(y0, y1 + 1))
    if test:
        years = years[:1]

    client = make_client("ewds")
    written: list[Path] = []
    for y in years:
        suffix = "test" if test else str(y)
        target = outdir / f"glofas_{system_version}_{suffix}.nc"
        if target.exists() and target.stat().st_size > 1024 and not test:
            log.info("skip existing %s", target.name)
            written.append(target)
            continue
        req = {
            "system_version": system_version,
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "variable": "river_discharge_in_the_last_24_hours",
            "hyear": [str(y)],
            "hmonth": ["06"] if test else MONTHS,
            "hday": ["15"] if test else DAYS,
            "data_format": data_format,
            "download_format": "unarchived",
            "area": _area(bbox),
        }
        log.info("requesting GloFAS %s -> %s", y, target.name)
        retrieve_with_retry(client, DATASET, req, str(target))
        written.append(target)
    return written


# --------------------------------------------------------------------------- #
#  Pixel extraction                                                            #
# --------------------------------------------------------------------------- #
def _open_glofas(files: list[Path]):
    import xarray as xr

    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords",
                           engine="netcdf4")
    # normalise coordinate / variable names across GloFAS encodings
    rename = {}
    for cand in ("dis24", "river_discharge_in_the_last_24_hours", "rivdis"):
        if cand in ds.variables:
            rename[cand] = "dis"
    for cand in ("latitude", "Latitude"):
        if cand in ds.coords:
            rename[cand] = "lat"
    for cand in ("longitude", "Longitude"):
        if cand in ds.coords:
            rename[cand] = "lon"
    for cand in ("time", "valid_time"):
        if cand in ds.coords:
            rename[cand] = "time"
    ds = ds.rename(rename)
    return ds


def extract_pixels(gauges: pd.DataFrame,
                   files: list[Path] | None = None,
                   uparea: "xr.DataArray | None" = None,  # noqa: F821
                   window: float = 0.15) -> pd.DataFrame:
    """Extract a daily GloFAS series at each gauge.

    Parameters
    ----------
    gauges : DataFrame with columns ``code, lon, lat`` and (optional)
        ``area_km2`` used for upstream-area matching.
    files  : list of downloaded GloFAS NetCDF files (defaults to all in
        ``datasets/glofas_era5``).
    uparea : optional GloFAS upstream-area DataArray (km2) on the same grid.
    window : half-width [deg] of the candidate search box around the gauge.
    """
    import xarray as xr  # noqa: F401

    files = files or sorted(PATHS.glofas.glob("glofas_*.nc"))
    if not files:
        raise FileNotFoundError("No GloFAS NetCDF files found; run download first.")
    ds = _open_glofas(files)
    dis = ds["dis"]

    records = []
    for _, g in gauges.iterrows():
        lon, lat = float(g["lon"]), float(g["lat"])
        area = float(g["area_km2"]) if "area_km2" in g and np.isfinite(
            g.get("area_km2", np.nan)) else np.nan
        qref = float(g["q_mean_ref"]) if "q_mean_ref" in g and np.isfinite(
            g.get("q_mean_ref", np.nan)) else np.nan
        # Snap the gauge to its contributing river cell.  Preference order:
        #   1) GloFAS upstream-area closest to the catchment area (if available);
        #   2) long-term mean discharge closest (log space) to the gauge's
        #      reference norm discharge -- robust to GloFAS's systematic bias;
        #   3) otherwise the highest-flow neighbour.
        # The window is widened once if the best match is implausibly low
        # (GloFAS < 40 % of the reference), i.e. the main stem lies further out.
        cell = None
        for w in (window, window * 2.5):
            sub = dis.sel(lat=slice(lat + w, lat - w), lon=slice(lon - w, lon + w))
            if sub.lat.size == 0 or sub.lon.size == 0:
                continue
            mflow = sub.mean("time").values
            if uparea is not None and np.isfinite(area):
                ua = uparea.sel(lat=sub.lat, lon=sub.lon).values
                score = np.where(np.isfinite(ua), np.abs(ua - area), np.inf)
            elif np.isfinite(qref) and qref > 0:
                lm = np.log(np.clip(mflow, 1e-6, None))
                score = np.where(np.isfinite(mflow), np.abs(lm - np.log(qref)), np.inf)
            else:
                score = np.where(np.isfinite(mflow), -mflow, np.inf)
            jbest = np.unravel_index(np.nanargmin(score), score.shape)
            cell = sub.isel(lat=jbest[0], lon=jbest[1])
            best_mean = float(mflow[jbest])
            good = uparea is not None or not (np.isfinite(qref) and qref > 0) \
                or best_mean >= 0.4 * qref
            if good:
                break
        if cell is None:
            cell = dis.sel(lat=lat, lon=lon, method="nearest")
        s = cell.to_series()
        rec = pd.DataFrame({"code": str(g["code"]),
                            "date": s.index.values,
                            "q_glofas": s.values})
        rec["sel_lat"] = float(cell["lat"])
        rec["sel_lon"] = float(cell["lon"])
        records.append(rec)
    out = pd.concat(records, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


AUX_DATASET = "cems-glofas-auxiliary-data"


def download_uparea(outdir: Path | None = None,
                    system_version: str = "version_4_0") -> Path:
    """Download the GloFAS static upstream-area map (km2 grid) for pixel snapping.

    Uses the ``cems-glofas-auxiliary-data`` collection on EWDS.  This is the
    authoritative way to match a gauge to its main-stem river cell.  Run it once
    the per-year discharge download has finished to avoid EWDS request contention.
    """
    outdir = Path(outdir or PATHS.glofas)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "glofas_uparea.nc"
    if target.exists() and target.stat().st_size > 1024:
        return target
    client = make_client("ewds")
    req = {
        "system_version": system_version,
        "variable": "upstream_area",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    retrieve_with_retry(client, AUX_DATASET, req, str(target))
    return target


def load_uparea(file: Path | None = None):
    """Load the GloFAS upstream-area map as a DataArray in km2 (lat/lon coords)."""
    import xarray as xr

    file = Path(file or PATHS.glofas / "glofas_uparea.nc")
    if not file.exists():
        return None
    ds = xr.open_dataset(file)
    ren = {}
    for c in ("latitude", "Latitude"):
        if c in ds.coords:
            ren[c] = "lat"
    for c in ("longitude", "Longitude"):
        if c in ds.coords:
            ren[c] = "lon"
    ds = ds.rename(ren)
    name = [v for v in ds.data_vars][0]
    da = ds[name]
    # GloFAS upstream area is in m2 -> convert to km2 if the magnitude is large
    if float(da.max()) > 1e9:
        da = da / 1e6
    return da


def main_extract() -> Path:
    gauges = pd.read_parquet(PATHS.processed / "gauges.parquet")
    static = pd.read_parquet(PATHS.processed / "static_attributes.parquet")
    if "area_km2" in static.columns:
        gauges = gauges.merge(static[["code", "area_km2"]], on="code", how="left")
    uparea = load_uparea()
    if uparea is not None:
        log.info("using GloFAS upstream-area map for pixel snapping")
    daily = extract_pixels(gauges.drop_duplicates("code"), uparea=uparea)
    out = save_table(daily, PATHS.interim / "glofas_daily_at_gauges.parquet")
    log.info("Saved GloFAS pixel series: %s rows -> %s", len(daily), out)
    return out


if __name__ == "__main__":
    main_extract()
