"""ERA5-Land dynamic forcing: download and basin aggregation.

Auxiliary meteorological / snow forcing for the bias-correction model (Copernicus
CDS, ERA5-Land, 0.1 deg).  The default path uses **monthly means**
(``reanalysis-era5-land-monthly-means``): it is fast, complete for 1979-present
and small, and provides the snow state (snow-water-equivalent), snowmelt,
temperature, precipitation, snowfall and soil moisture needed to explain the
GloFAS bias.  Monthly forcing is broadcast onto the decadal/daily target periods
during assembly; the sub-monthly discharge timing is supplied by GloFAS itself.

A higher-resolution daily path (``derived-era5-land-daily-statistics``) is
provided by :func:`download_era5_land_daily` for a future upgrade, but that
product is aggregated server-side and is far slower to retrieve.

Snow-relevant variables: snow-depth water equivalent, snowmelt, snowfall, 2 m
temperature, total precipitation and top-layer soil moisture.
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

MONTHLY_DATASET = "reanalysis-era5-land-monthly-means"
DAILY_DATASET = "derived-era5-land-daily-statistics"
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

# (cds variable, output tag) used for the monthly forcing
VARIABLES = [
    ("2m_temperature", "t2m_mean"),
    ("total_precipitation", "tp"),
    ("snow_depth_water_equivalent", "swe"),
    ("snowmelt", "smlt"),
    ("snowfall", "sf"),
    ("volumetric_soil_water_layer_1", "swvl1"),
]


def _area(bbox: dict) -> list[float]:
    return [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]


def download_era5_land(years: list[int] | None = None,
                       bbox: dict | None = None,
                       outdir: Path | None = None,
                       variables: list[tuple[str, str]] | None = None,
                       test: bool = False) -> list[Path]:
    """Download ERA5-Land **monthly-mean** forcing, one NetCDF per variable."""
    bbox = bbox or D.STUDY_BBOX
    outdir = Path(outdir or PATHS.era5_land)
    outdir.mkdir(parents=True, exist_ok=True)
    variables = variables or VARIABLES
    if years is None:
        years = list(range(int(D.PERIOD["start"][:4]), int(D.PERIOD["end"][:4]) + 1))

    client = make_client("cds")
    written: list[Path] = []
    for var, tag in variables:
        target = outdir / (f"era5land_{tag}_test.nc" if test else f"era5land_{tag}.nc")
        if target.exists() and target.stat().st_size > 1024 and not test:
            log.info("skip existing %s", target.name)
            written.append(target)
            continue
        req = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [var],
            "year": ["1979"] if test else [str(y) for y in years],
            "month": ["06"] if test else MONTHS,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": _area(bbox),
        }
        log.info("requesting ERA5-Land monthly %s -> %s", var, target.name)
        retrieve_with_retry(client, MONTHLY_DATASET, req, str(target))
        written.append(target)
    return written


def download_era5_land_daily(years=None, bbox=None, outdir=None,
                             variable_stats=None, test=False) -> list[Path]:
    """Optional high-resolution path: ERA5-Land daily statistics (slow)."""
    bbox = bbox or D.STUDY_BBOX
    outdir = Path(outdir or PATHS.era5_land)
    outdir.mkdir(parents=True, exist_ok=True)
    variable_stats = variable_stats or [
        ("2m_temperature", "daily_mean", "t2m_mean"),
        ("total_precipitation", "daily_maximum", "tp"),
        ("snow_depth_water_equivalent", "daily_mean", "swe"),
        ("snowmelt", "daily_maximum", "smlt"),
    ]
    if years is None:
        years = list(range(int(D.PERIOD["start"][:4]), int(D.PERIOD["end"][:4]) + 1))
    if test:
        years, variable_stats = years[:1], variable_stats[:1]
    client = make_client("cds")
    written: list[Path] = []
    for var, stat, tag in variable_stats:
        for y in years:
            target = outdir / (f"era5landD_{tag}_test.nc" if test
                               else f"era5landD_{tag}_{y}.nc")
            if target.exists() and target.stat().st_size > 1024 and not test:
                written.append(target)
                continue
            req = {"variable": [var], "year": [str(y)],
                   "month": ["06"] if test else MONTHS,
                   "day": ["15"] if test else DAYS,
                   "daily_statistic": stat, "time_zone": "utc+00:00",
                   "frequency": "1_hourly", "area": _area(bbox)}
            log.info("requesting ERA5-Land daily %s/%s %s", var, stat, y)
            retrieve_with_retry(client, DAILY_DATASET, req, str(target))
            written.append(target)
    return written


# --------------------------------------------------------------------------- #
#  Basin aggregation                                                           #
# --------------------------------------------------------------------------- #
def _zonal_or_nearest(da, lon: float, lat: float, geom=None, window: float = 0.15):
    """Mean over grid cells inside ``geom`` (if given) else a small window."""
    if geom is not None:
        from shapely.geometry import Point

        sub = da.sel(lat=slice(lat + 1.0, lat - 1.0), lon=slice(lon - 1.0, lon + 1.0))
        lons, lats = np.meshgrid(sub.lon.values, sub.lat.values)
        mask = np.array([geom.contains(Point(x, y))
                         for x, y in zip(lons.ravel(), lats.ravel())]).reshape(lons.shape)
        if mask.any():
            idx = np.where(mask)
            return sub.isel(lat=("z", idx[0]), lon=("z", idx[1])).mean("z")
    sub = da.sel(lat=slice(lat + window, lat - window),
                 lon=slice(lon - window, lon + window))
    if sub.lat.size and sub.lon.size:
        return sub.mean(["lat", "lon"])
    return da.sel(lat=lat, lon=lon, method="nearest")


def _open_norm(files: list[Path]):
    import xarray as xr

    ds = xr.open_mfdataset([str(f) for f in sorted(files)],
                           combine="by_coords", engine="netcdf4")
    ren = {}
    for c in ("latitude", "Latitude"):
        if c in ds.coords:
            ren[c] = "lat"
    for c in ("longitude", "Longitude"):
        if c in ds.coords:
            ren[c] = "lon"
    for c in ("valid_time",):
        if c in ds.coords:
            ren[c] = "time"
    ds = ds.rename(ren)
    # collapse a multi-valued expver (recent ERA5T months) to the first product
    if "expver" in ds.dims and ds.sizes.get("expver", 1) > 1:
        ds = ds.isel(expver=0)
    # remove scalar ensemble / version / surface helpers, whether they appear as
    # singleton dimensions (squeeze) or as scalar coordinates (drop).
    for extra in ("number", "surface", "depthBelowLandLayer", "expver"):
        if extra in ds.dims and ds.sizes.get(extra, 1) == 1:
            ds = ds.squeeze(extra, drop=True)
        elif extra in ds.coords:
            ds = ds.drop_vars(extra, errors="ignore")
    return ds


def extract_forcing(gauges: pd.DataFrame, files: list[Path] | None = None,
                    basins=None) -> pd.DataFrame:
    """Aggregate ERA5-Land forcing to each gauge's basin -> wide monthly table."""
    import re

    files = files or sorted(PATHS.era5_land.glob("era5land_*.nc"))
    if not files:
        raise FileNotFoundError("No ERA5-Land files found; run download first.")

    by_tag: dict[str, list[Path]] = {}
    for f in files:
        tag = f.stem
        if tag.startswith("era5land_"):
            tag = tag[len("era5land_"):]
        tag = re.sub(r"_(test|\d{4})$", "", tag)
        by_tag.setdefault(tag, []).append(f)

    geom_by_code = {}
    if basins is not None:
        for _, b in basins.iterrows():
            geom_by_code[str(b.get("CODE", b.get("code")))] = b.geometry

    frames = []
    for tag, flist in by_tag.items():
        ds = _open_norm(flist)
        var = [v for v in ds.data_vars][0]
        da = ds[var]
        for _, g in gauges.iterrows():
            geom = geom_by_code.get(str(g["code"]))
            ser = _zonal_or_nearest(da, float(g["lon"]), float(g["lat"]), geom).to_series()
            frames.append(pd.DataFrame({"code": str(g["code"]),
                                        "date": pd.to_datetime(ser.index),
                                        "var": tag, "value": ser.values}))
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["code", "date"], columns="var",
                            values="value").reset_index()
    wide.columns.name = None

    # Normalise ERA5-Land units to the physical conventions the feature and
    # regime rules assume: temperature in degC (not K), water fluxes / SWE in mm
    # water equivalent (not m).  Soil moisture (swvl1) is already 0-1 m3/m3.
    for c in wide.columns:
        if c.startswith("t2m"):
            wide[c] = wide[c] - 273.15
        elif c in ("tp", "sf", "smlt", "swe"):
            wide[c] = wide[c] * 1000.0

    wide["year"] = wide["date"].dt.year
    wide["month"] = wide["date"].dt.month
    return wide


def main_extract() -> Path:
    gauges = pd.read_parquet(PATHS.processed / "gauges.parquet").drop_duplicates("code")
    basins = None
    bp = PATHS.processed / "basins.parquet"
    if bp.exists():
        try:
            import geopandas as gpd

            basins = gpd.read_parquet(bp)
        except Exception as exc:  # pragma: no cover
            log.warning("basin polygons unavailable: %s", exc)
    forcing = extract_forcing(gauges, basins=basins)
    out = save_table(forcing, PATHS.interim / "era5land_monthly_at_basins.parquet")
    log.info("Saved ERA5-Land forcing: %s rows x %s cols -> %s",
             forcing.shape[0], forcing.shape[1], out)
    return out


if __name__ == "__main__":
    main_extract()
