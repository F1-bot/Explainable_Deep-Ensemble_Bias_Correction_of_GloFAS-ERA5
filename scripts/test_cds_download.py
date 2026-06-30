"""Validate Copernicus credentials + request schemas with tiny test downloads."""
import traceback

import xarray as xr

from sbc.data.glofas import download_glofas
from sbc.data.era5_land import download_era5_land

print("=" * 70, flush=True)
print("TEST 1/2 — GloFAS-ERA5 via EWDS (1 day, study bbox)", flush=True)
print("=" * 70, flush=True)
try:
    f = download_glofas(test=True)[0]
    ds = xr.open_dataset(f)
    print("GloFAS OK ->", f.name, "| sizes:", dict(ds.sizes),
          "| vars:", list(ds.data_vars), "| coords:", list(ds.coords), flush=True)
except Exception:
    print("GloFAS FAILED:", flush=True)
    traceback.print_exc()

print("\n" + "=" * 70, flush=True)
print("TEST 2/2 — ERA5-Land via CDS (1 var, 1 day, study bbox)", flush=True)
print("=" * 70, flush=True)
try:
    f = download_era5_land(test=True)[0]
    ds = xr.open_dataset(f)
    print("ERA5-Land OK ->", f.name, "| sizes:", dict(ds.sizes),
          "| vars:", list(ds.data_vars), "| coords:", list(ds.coords), flush=True)
except Exception:
    print("ERA5-Land FAILED:", flush=True)
    traceback.print_exc()
print("\nDONE", flush=True)
