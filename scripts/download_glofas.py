"""Download the full GloFAS-ERA5 v4.0 daily reanalysis for the study domain.

One NetCDF per hydrological year (resumable: existing years are skipped).
Run::  PYTHONPATH=src python scripts/download_glofas.py
"""
from sbc.data.glofas import download_glofas

if __name__ == "__main__":
    files = download_glofas()
    print(f"GloFAS download complete: {len(files)} yearly files", flush=True)
