"""Download ERA5-Land daily-statistics forcing for the study domain.

One NetCDF per (variable, statistic) tag covering 1979-2020 (resumable).
Run::  PYTHONPATH=src python scripts/download_era5_land.py
"""
from sbc.data.era5_land import download_era5_land

if __name__ == "__main__":
    files = download_era5_land()
    print(f"ERA5-Land download complete: {len(files)} files", flush=True)
