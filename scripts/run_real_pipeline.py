"""End-to-end REAL-data pipeline: snap GloFAS pixels -> assemble -> experiment.

Usage::
    PYTHONPATH=src python scripts/run_real_pipeline.py            # full publication run
    PYTHONPATH=src python scripts/run_real_pipeline.py --quick    # fast smoke
    PYTHONPATH=src python scripts/run_real_pipeline.py --no-uparea # skip aux-map download

Steps
-----
1. (optional) download the GloFAS upstream-area map for hydrologically correct
   gauge -> river-pixel snapping (run only once the per-year discharge download
   has finished, to avoid EWDS request contention);
2. extract the GloFAS daily series at every gauge (uses only fully-written
   yearly NetCDFs, so it is safe to run while a download is still in progress);
3. assemble the decadal and daily modelling tables (ERA5-Land monthly forcing is
   merged if present);
4. run the validation-matrix experiment for each scale and print the summary.
"""
from __future__ import annotations

import argparse

import pandas as pd

from sbc.config import PATHS
from sbc.data import glofas
from sbc.data.assemble import assemble
from sbc.experiment import run_experiment
from sbc.utils import get_logger, save_table

log = get_logger("run_real")


def extract_glofas(min_size_mb: int = 10, use_uparea: bool = True):
    files = sorted(f for f in PATHS.glofas.glob("glofas_version_4_0_*.nc")
                   if f.stat().st_size > min_size_mb * 1_000_000)
    if not files:
        raise FileNotFoundError("No complete GloFAS NetCDFs found; run scripts/download_glofas.py")
    log.info("snapping pixels from %d complete GloFAS years (%s..%s)",
             len(files), files[0].stem[-4:], files[-1].stem[-4:])
    gauges = pd.read_parquet(PATHS.processed / "gauges.parquet").drop_duplicates("code")
    static = pd.read_parquet(PATHS.processed / "static_attributes.parquet")
    if "area_km2" in static.columns:
        gauges = gauges.merge(static[["code", "area_km2"]], on="code", how="left")
    uparea = glofas.load_uparea() if use_uparea else None
    if use_uparea and uparea is None:
        log.warning("no upstream-area map; falling back to discharge-matched snapping")
    daily = glofas.extract_pixels(gauges, files=files, uparea=uparea)
    save_table(daily, PATHS.interim / "glofas_daily_at_gauges.parquet")
    log.info("GloFAS pixel series: %d rows, %d gauges", len(daily), daily["code"].nunique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-uparea", action="store_true")
    ap.add_argument("--scales", default="decadal,daily")
    a = ap.parse_args()

    if not a.no_uparea:
        try:
            glofas.download_uparea()
        except Exception as exc:  # pragma: no cover
            log.warning("upstream-area download skipped (%s)", exc)

    extract_glofas(use_uparea=not a.no_uparea)

    for scale in a.scales.split(","):
        scale = scale.strip()
        table = assemble(scale)
        save_table(table, PATHS.processed / f"model_table_{scale}.parquet")
        log.info("assembled %s table: %s", scale, table.shape)
        out = run_experiment(scale=scale, source="real", quick=a.quick)
        print(f"\n===== REAL {scale} summary =====")
        print(out["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
