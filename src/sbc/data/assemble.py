"""Assemble the modelling table by merging the real data sources per scale.

This is the last data-layer step before feature engineering.  It fuses the four
analysis-ready products of the upstream extractors into the single tidy
*modelling table* defined in :mod:`sbc.schemas` — one row per (gauge, period):

* **observed discharge** (ground truth) — :mod:`sbc.data.ca_discharge`
  ``processed/discharge_{decadal,daily}.parquet`` (``code, date, q_obs``);
* **GloFAS-ERA5 discharge** (the predictor to be bias-corrected) —
  :mod:`sbc.data.glofas` ``interim/glofas_daily_at_gauges.parquet``
  (``code, date, q_glofas``), always extracted at daily resolution;
* **ERA5-Land dynamic forcing** (snow / melt / temperature / precipitation
  drivers) — :mod:`sbc.data.era5_land`
  ``interim/era5land_daily_at_basins.parquet`` (``code, date, <forcing tags>``),
  also daily;
* **static catchment attributes** (the full ~1090-column HydroATLAS-style
  table) — :mod:`sbc.data.ca_discharge` ``processed/static_attributes.parquet``
  (``code`` + numeric attributes), merged on ``code`` in its entirety
  (curation into compact descriptors happens later in
  :mod:`sbc.features.engineering`).

Because GloFAS and ERA5-Land are produced daily, the ``decadal`` scale first
aggregates both series to the Central-Asian decade (days 1-10, 11-20, 21-end,
represented by the 5th / 15th / 25th of the month; period mean) so they align
with the decadal observations before the join.  Rows missing either the
observation or the GloFAS value are dropped, and the log-residual target is
added through :func:`sbc.schemas.make_target`.

Run as a script to (re)build both tables::

    PYTHONPATH=src python -m sbc.data.assemble
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS, SCALES
from ..schemas import (
    ID_COLS,
    OBS_COL,
    SIM_COL,
    TARGET_COL,
    make_target,
    validate,
)
from ..utils import get_logger, save_table

log = get_logger(__name__)

# Producer module advertised in error messages when an input is absent.
_PRODUCERS = {
    "gauges.parquet": "sbc.data.ca_discharge",
    "discharge": "sbc.data.ca_discharge",
    "static_attributes.parquet": "sbc.data.ca_discharge",
    "glofas_daily_at_gauges.parquet": "sbc.data.glofas",
    "era5land_daily_at_basins.parquet": "sbc.data.era5_land",
}


# --------------------------------------------------------------------------- #
#  Decadal aggregation helpers                                                 #
# --------------------------------------------------------------------------- #
def _decadal_rep_date(dates: pd.Series) -> pd.DatetimeIndex:
    """Map daily dates onto Central-Asian decade representative dates.

    Days 1-10, 11-20 and 21-end of each month collapse to the 5th, 15th and
    25th of that month respectively.
    """
    dates = pd.to_datetime(dates)
    day = dates.dt.day.to_numpy()
    rep = np.where(day <= 10, 5, np.where(day <= 20, 15, 25))
    return pd.to_datetime({"year": dates.dt.year.to_numpy(),
                           "month": dates.dt.month.to_numpy(),
                           "day": rep})


def _to_decadal(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Aggregate a daily ``code, date, <value_cols>`` frame to decade means."""
    df = df[["code", "date"] + value_cols].copy()
    df["date"] = _decadal_rep_date(df["date"])
    return (df.groupby(["code", "date"], as_index=False)[value_cols]
            .mean())


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a dynamic frame to canonical ``code`` (str) / ``date`` dtypes."""
    df = df.copy()
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- #
#  Core merge (pure; no IO so it can be unit-/smoke-tested in memory)          #
# --------------------------------------------------------------------------- #
def _merge_tables(scale: str,
                  obs: pd.DataFrame,
                  glofas: pd.DataFrame,
                  forcing: pd.DataFrame,
                  static: pd.DataFrame,
                  gauges: pd.DataFrame) -> pd.DataFrame:
    """Fuse the source frames into a validated modelling table.

    Parameters
    ----------
    scale : {"decadal", "daily"}
        Target temporal scale.  For ``"decadal"`` the (daily) GloFAS and
        ERA5-Land frames are aggregated to decade means; ``obs`` is assumed to
        already be at the requested scale.
    obs : DataFrame
        ``code, date, q_obs`` at ``scale``.
    glofas : DataFrame
        ``code, date, q_glofas`` (daily; extra columns ignored).
    forcing : DataFrame
        ``code, date, <forcing tags>`` (daily).
    static : DataFrame
        ``code`` + numeric static attributes (one row per gauge).
    gauges : DataFrame
        Gauge metadata carrying at least ``code, basin, domain``.

    Returns
    -------
    DataFrame
        Modelling table with columns ``code, basin, domain, scale, date,
        q_obs, q_glofas, log_residual, <forcing>, <static>``.
    """
    obs = _norm(obs)
    glofas = _norm(glofas)
    forcing = _norm(forcing)

    # ERA5-Land forcing may be monthly (broadcast onto each period of its month)
    # or daily (aggregated to the target scale); detect which from its columns.
    is_monthly = {"year", "month"}.issubset(forcing.columns)
    forcing_cols = [c for c in forcing.columns
                    if c not in ("code", "date", "year", "month")]

    # Drop any columns from static that would clash with reserved names so the
    # full-attribute merge cannot shadow id / target / forcing columns.
    reserved = (set(ID_COLS) | {OBS_COL, SIM_COL, TARGET_COL} | set(forcing_cols)) - {"code"}
    static = static.drop(columns=[c for c in static.columns if c in reserved],
                         errors="ignore").copy()
    static["code"] = static["code"].astype(str)
    static = static.drop_duplicates("code")
    static_cols = [c for c in static.columns if c != "code"]

    # GloFAS is daily -> aggregate to decade means for the decadal scale.
    if scale == "decadal":
        glofas = _to_decadal(glofas[["code", "date", SIM_COL]], [SIM_COL])
    else:
        glofas = glofas[["code", "date", SIM_COL]]

    # Inner-join obs <-> GloFAS guarantees both q_obs and q_glofas are present.
    base = obs.merge(glofas, on=["code", "date"], how="inner")

    # Left-merge the dynamic forcing (curation handles its gaps later).
    if forcing_cols:
        if is_monthly:
            fwide = (forcing[["code", "year", "month"] + forcing_cols]
                     .drop_duplicates(["code", "year", "month"])
                     .rename(columns={"year": "_y", "month": "_m"}))
            base["_y"] = base["date"].dt.year
            base["_m"] = base["date"].dt.month
            base = base.merge(fwide, on=["code", "_y", "_m"], how="left") \
                       .drop(columns=["_y", "_m"])
        else:
            fc = (_to_decadal(forcing[["code", "date"] + forcing_cols], forcing_cols)
                  if scale == "decadal"
                  else forcing[["code", "date"] + forcing_cols])
            base = base.merge(fc, on=["code", "date"], how="left")

    base = base.merge(static, on="code", how="left")

    meta = gauges.copy()
    meta["code"] = meta["code"].astype(str)
    meta = meta.drop_duplicates("code")[["code", "basin", "domain"]]
    base = base.merge(meta, on="code", how="left")

    base["scale"] = scale
    base = base.dropna(subset=[OBS_COL, SIM_COL]).reset_index(drop=True)
    base[TARGET_COL] = make_target(base[OBS_COL], base[SIM_COL])

    lead = ID_COLS + [OBS_COL, SIM_COL, TARGET_COL]
    ordered = [c for c in lead + forcing_cols + static_cols if c in base.columns]
    ordered += [c for c in base.columns if c not in ordered]
    base = base[ordered]

    log.info("scale=%-8s rows=%-6d gauges=%-3d forcing=%d static=%d",
             scale, len(base), base["code"].nunique(),
             len(forcing_cols), len(static_cols))
    return validate(base)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
def _require(path: Path, key: str) -> pd.DataFrame:
    """Read a required parquet input, raising a clear error if it is absent."""
    path = Path(path)
    if not path.exists():
        producer = _PRODUCERS.get(key, "the corresponding extractor")
        raise FileNotFoundError(
            f"{path} not found. Produce it by running {producer} "
            f"(e.g. PYTHONPATH=src python -m {producer})."
        )
    return pd.read_parquet(path)


def assemble(scale: str) -> pd.DataFrame:
    """Build the modelling table for one temporal ``scale`` from real inputs.

    Parameters
    ----------
    scale : {"decadal", "daily"}
        Temporal scale to assemble.

    Returns
    -------
    pandas.DataFrame
        Validated modelling table (see :mod:`sbc.schemas`).

    Raises
    ------
    ValueError
        If ``scale`` is not one of :data:`sbc.config.SCALES`.
    FileNotFoundError
        If any required processed/interim input is missing.
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {SCALES}")

    gauges = _require(PATHS.processed / "gauges.parquet", "gauges.parquet")
    obs = _require(PATHS.processed / f"discharge_{scale}.parquet", "discharge")
    static = _require(PATHS.processed / "static_attributes.parquet",
                      "static_attributes.parquet")
    glofas = _require(PATHS.interim / "glofas_daily_at_gauges.parquet",
                      "glofas_daily_at_gauges.parquet")
    # ERA5-Land forcing is optional: the table still assembles from GloFAS +
    # static attributes if the dynamic forcing has not been extracted yet.
    fpath = PATHS.interim / "era5land_monthly_at_basins.parquet"
    if fpath.exists():
        forcing = pd.read_parquet(fpath)
    else:
        log.warning("%s absent; assembling without ERA5-Land dynamic forcing",
                    fpath.name)
        forcing = pd.DataFrame(columns=["code", "date"])

    if "scale" in gauges.columns:
        gauges = gauges[gauges["scale"].astype(str) == scale]

    return _merge_tables(scale, obs, glofas, forcing, static, gauges)


def main() -> None:
    """Assemble every scale whose inputs are present and persist the tables."""
    PATHS.ensure()
    for scale in SCALES:
        needed = [
            PATHS.processed / "gauges.parquet",
            PATHS.processed / f"discharge_{scale}.parquet",
            PATHS.processed / "static_attributes.parquet",
            PATHS.interim / "glofas_daily_at_gauges.parquet",
        ]  # ERA5-Land forcing is optional (see assemble())
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            log.warning("scale=%s: skipping; missing inputs %s", scale, missing)
            print(f"model_table_{scale}: SKIPPED (missing {missing})")
            continue
        table = assemble(scale)
        out = save_table(table, PATHS.processed / f"model_table_{scale}.parquet")
        print(f"model_table_{scale}: {table.shape[0]} rows x "
              f"{table.shape[1]} cols -> {out}")


# --------------------------------------------------------------------------- #
#  Self-test: synthesise tiny stand-in sources and exercise the merge logic.   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sbc.synthetic import FORCING, STATIC, generate

    # A tiny synthetic *daily* modelling table stands in for the real sources;
    # we split it back into the four input frames the public reader expects.
    t = generate(n_basins=2, gauges_per_basin=(2, 3), years=2,
                 scale="daily", seed=0)
    forcing_tags = [f for f in FORCING if f in t.columns]
    static_tags = [s for s in STATIC if s in t.columns]

    obs_daily = t[["code", "date", "q_obs"]]
    glofas_daily = t[["code", "date", "q_glofas"]]
    forcing_daily = t[["code", "date"] + forcing_tags]
    static_tbl = t[["code"] + static_tags].drop_duplicates("code")
    gauges_tbl = t[["code", "basin", "domain"]].drop_duplicates("code")

    # Daily path: straight inner-join on (code, date).
    daily_tbl = _merge_tables("daily", obs_daily, glofas_daily,
                              forcing_daily, static_tbl,
                              gauges_tbl.assign(scale="daily"))

    # Decadal path: GloFAS + ERA5-Land aggregated to decades, joined to the
    # (also decadal) observations built with the same decade mapping.
    obs_decadal = _to_decadal(_norm(obs_daily), ["q_obs"])
    decadal_tbl = _merge_tables("decadal", obs_decadal, glofas_daily,
                                forcing_daily, static_tbl,
                                gauges_tbl.assign(scale="decadal"))

    ok_cols = set(ID_COLS + [OBS_COL, SIM_COL, TARGET_COL]) <= set(daily_tbl.columns)
    ok_finite = bool(np.isfinite(daily_tbl[TARGET_COL]).all())
    print(f"[assemble self-test] daily={daily_tbl.shape} "
          f"decadal={decadal_tbl.shape} "
          f"forcing={len(forcing_tags)} static={len(static_tags)} "
          f"schema_ok={ok_cols} target_finite={ok_finite}")
