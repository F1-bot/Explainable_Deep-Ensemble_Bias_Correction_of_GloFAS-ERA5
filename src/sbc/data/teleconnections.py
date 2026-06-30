"""Hydroclimatic teleconnection predictors (ENSO / NAO / PDO / AO).

Large-scale ocean-atmosphere oscillations modulate Central-Asian snowpack and
spring-summer streamflow with multi-month *lead* times, and Umirbekov et al.
(2025) show that ENSO, NAO and PDO dominate long-lead discharge skill in exactly
the Syr Darya / Chu / Talas basins targeted here.  Because these indices *lead*
streamflow, their lagged values are physically motivated, leakage-safe
predictors for the bias-correction model.

This module fetches the monthly climate indices published by NOAA PSL
(``https://psl.noaa.gov/data/correlation/<name>.data``, ultimately sourced from
CPC) and merges lagged copies onto the modelling table by calendar year-month:

* **oni** -- Oceanic Nino Index / Nino 3.4 (ENSO state);
* **nao** -- North Atlantic Oscillation;
* **pdo** -- Pacific Decadal Oscillation;
* **ao**  -- Arctic Oscillation.

The PSL ``.data`` text files share a fixed layout: a header line with the first
and last year, one row per year holding 12 monthly values, then a trailing
missing-value sentinel (e.g. ``-99.9`` / ``-99.99`` / ``-999``) and a free-text
provenance block.  :func:`download_indices` parses that into a tidy monthly
DataFrame ``[date, oni, nao, pdo, ao]`` and caches it as
``datasets/teleconnections.parquet``; :func:`add_teleconnections` left-merges the
lagged columns onto a modelling table.

Run as a script to (re)download the indices and self-test the merge::

    PYTHONPATH=src python -m sbc.data.teleconnections
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from ..utils import get_logger, save_table

log = get_logger(__name__)

#: NOAA PSL monthly-correlation text-file root.
PSL_BASE = "https://psl.noaa.gov/data/correlation"

#: Output index column -> ordered candidate PSL file basenames.  The first
#: source that downloads and parses to any non-null value wins; the fallbacks
#: keep the pipeline running if a single endpoint is temporarily unreachable.
INDEX_SOURCES: dict[str, tuple[str, ...]] = {
    "oni": ("oni", "nina34"),   # ENSO: standardised ONI, raw Nino3.4 SST fallback
    "nao": ("nao",),
    "pdo": ("pdo",),
    "ao": ("ao",),
}

#: Canonical ordered index columns produced by this module.
INDEX_COLS: tuple[str, ...] = tuple(INDEX_SOURCES)

#: Any value at or below this is treated as a missing-data sentinel.  The PSL
#: products use -99.9 / -99.99 / -999 / -9999; no real climate index (or even
#: the raw Nino 3.4 SST in degC) approaches this magnitude.
_MISSING_THRESHOLD = -90.0


# --------------------------------------------------------------------------- #
#  Download + parse                                                            #
# --------------------------------------------------------------------------- #
def _download_one(name: str, timeout: float = 60.0) -> str | None:
    """Fetch one PSL ``.data`` file as text, or ``None`` if unreachable.

    Tries :mod:`urllib` first (Colab-safe, no external binary) and falls back to
    ``curl`` so the function still works where Python's SSL stack is restricted.
    """
    url = f"{PSL_BASE}/{name}.data"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sbc/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("urllib failed for %s (%s); trying curl", url, exc)
    try:
        out = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", str(int(timeout)), url],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
        log.warning("curl failed for %s (rc=%s)", url, out.returncode)
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("curl unavailable for %s (%s)", url, exc)
    return None


def _parse_psl(text: str, col: str) -> pd.DataFrame:
    """Parse a PSL fixed-width ``.data`` payload into ``[date, <col>]`` monthly.

    Parameters
    ----------
    text : str
        Raw file contents.
    col : str
        Output value-column name.

    Returns
    -------
    DataFrame
        One row per month (``date`` = month start) with the index value;
        sentinel values are mapped to ``NaN``.
    """
    start_year: int | None = None
    end_year: int | None = None
    sentinel: float | None = None
    rows: list[tuple[int, list[float]]] = []

    for line in text.splitlines():
        toks = line.split()
        if not toks:
            continue
        if start_year is None:
            # Header: "<start_year> <end_year>".
            try:
                start_year, end_year = int(toks[0]), int(toks[1])
            except (ValueError, IndexError):
                start_year = None
            continue
        # Data row: <year> + 12 monthly values.
        if len(toks) >= 13:
            try:
                year = int(toks[0])
                vals = [float(x) for x in toks[1:13]]
            except ValueError:
                continue
            if start_year <= year <= end_year:
                rows.append((year, vals))
                continue
        # A lone numeric line after the data block is the missing-value sentinel.
        if len(toks) == 1 and sentinel is None and rows:
            try:
                sentinel = float(toks[0])
            except ValueError:
                pass

    if not rows:
        raise ValueError(f"no data rows parsed for {col!r}")

    recs: list[tuple[pd.Timestamp, float]] = []
    for year, vals in rows:
        for month, value in enumerate(vals, start=1):
            recs.append((pd.Timestamp(year=year, month=month, day=1), value))
    out = pd.DataFrame(recs, columns=["date", col])

    miss = out[col] <= _MISSING_THRESHOLD
    if sentinel is not None:
        miss |= np.isclose(out[col].to_numpy(), sentinel)
    out.loc[miss, col] = np.nan
    return out


def download_indices(outdir: str | Path | None = None,
                     timeout: float = 60.0,
                     sources: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    """Download the monthly teleconnection indices and cache them to parquet.

    Parameters
    ----------
    outdir : path, optional
        Directory for ``teleconnections.parquet`` (default ``datasets/``).
    timeout : float
        Per-request timeout [s].
    sources : dict, optional
        Override of :data:`INDEX_SOURCES` (output column -> candidate basenames).

    Returns
    -------
    DataFrame
        Tidy monthly indices ``[date, oni, nao, pdo, ao]`` (columns present even
        if their source was unreachable, in which case the column is all-``NaN``).

    Raises
    ------
    RuntimeError
        If *no* index could be downloaded at all.
    """
    sources = sources or INDEX_SOURCES
    outdir = Path(outdir) if outdir is not None else PATHS.datasets
    outdir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for col, candidates in sources.items():
        parsed: pd.DataFrame | None = None
        for name in candidates:
            text = _download_one(name, timeout=timeout)
            if text is None:
                continue
            try:
                cand = _parse_psl(text, col)
            except Exception as exc:
                log.warning("parse failed for %s (%s): %s", name, col, exc)
                continue
            if cand[col].notna().any():
                parsed = cand
                log.info("%-4s <- %-7s %d months (%d non-null, %s..%s)",
                         col, name, len(cand), int(cand[col].notna().sum()),
                         cand["date"].min().date(), cand["date"].max().date())
                break
        if parsed is None:
            log.warning("no reachable source for index %r; continuing without it", col)
            continue
        frames.append(parsed)

    if not frames:
        raise RuntimeError(
            "could not download any teleconnection index from NOAA PSL; "
            "check network access to https://psl.noaa.gov/data/correlation/")

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    for col in sources:                       # guarantee canonical columns/order
        if col not in out.columns:
            out[col] = np.nan
    out = out.sort_values("date").reset_index(drop=True)
    out = out[["date", *sources]]

    path = save_table(out, outdir / "teleconnections.parquet")
    log.info("saved %d monthly rows x %d indices -> %s",
             len(out), len(sources), path)
    return out


def load_indices(path: str | Path | None = None) -> pd.DataFrame:
    """Read the cached monthly indices, raising a clear error if absent."""
    path = Path(path) if path is not None else PATHS.datasets / "teleconnections.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first with "
            f"`PYTHONPATH=src python -m sbc.data.teleconnections` "
            f"(or call sbc.data.teleconnections.download_indices()).")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- #
#  Merge onto the modelling table                                             #
# --------------------------------------------------------------------------- #
def add_teleconnections(df: pd.DataFrame,
                        lags: tuple[int, ...] = (0, 1, 2, 3, 6),
                        indices: pd.DataFrame | None = None,
                        path: str | Path | None = None) -> pd.DataFrame:
    """Left-merge lagged teleconnection indices onto a modelling table.

    For each index ``X`` and lag ``K`` a feature column ``f_X_lagK`` is added,
    holding the value of ``X`` ``K`` calendar months *before* each row's period.
    Because the indices lead streamflow, every lag (including 0) uses only
    information available before the discharge, so the predictors are
    leakage-safe.  The merge is by calendar ``(year, month)``.

    Parameters
    ----------
    df : DataFrame
        Modelling table with a datetime ``date`` column.
    lags : tuple of int
        Month lags to add (non-negative; 0 = contemporaneous month).
    indices : DataFrame, optional
        Pre-loaded monthly indices ``[date, oni, ...]``.  When ``None`` they are
        read from ``path`` (default ``datasets/teleconnections.parquet``).
    path : path, optional
        Location of the cached indices parquet.

    Returns
    -------
    DataFrame
        A new table (the input is not mutated) with the ``f_<index>_lag<K>``
        columns appended.
    """
    if indices is None:
        indices = load_indices(path)
    else:
        indices = indices.copy()
        indices["date"] = pd.to_datetime(indices["date"])

    index_cols = [c for c in indices.columns if c != "date"]
    if not index_cols:
        log.warning("no index columns in teleconnection table; returning input copy")
        return df.copy()

    # Continuous monthly PeriodIndex so shifting by a lag is an exact month step.
    per = indices["date"].dt.to_period("M")
    base = (pd.DataFrame(indices[index_cols].to_numpy(),
                         index=pd.PeriodIndex(per, freq="M"), columns=index_cols)
            .groupby(level=0).first()
            .sort_index())
    full = pd.period_range(base.index.min(), base.index.max(), freq="M")
    base = base.reindex(full)

    out = df.copy()
    row_ym = pd.to_datetime(out["date"]).dt.to_period("M")
    for col in index_cols:
        for lag in lags:
            if lag < 0:
                raise ValueError(f"lags must be non-negative, got {lag}")
            shifted = base[col].shift(lag)         # row at p now holds value of p-lag
            out[f"f_{col}_lag{lag}"] = row_ym.map(shifted).to_numpy()
    return out


# --------------------------------------------------------------------------- #
#  Self-test: download the real indices and merge onto real / synthetic dates  #
# --------------------------------------------------------------------------- #
def _self_test_frame() -> tuple[pd.DataFrame, str]:
    """Return a small (date-bearing) table for the self-test and its source tag."""
    for name in ("model_table_decadal.parquet", "discharge_decadal.parquet"):
        p = PATHS.processed / name
        if p.exists():
            df = pd.read_parquet(p, columns=None)
            df = df[["code", "date"]].copy() if {"code", "date"} <= set(df.columns) else df
            df["date"] = pd.to_datetime(df["date"])
            return df, name
    from ..synthetic import generate
    return generate(scale="decadal", years=8, n_basins=3), "synthetic(decadal)"


def main() -> None:
    """Download the indices, then self-test :func:`add_teleconnections`."""
    indices = download_indices()
    df, src = _self_test_frame()

    lags = (0, 1, 2, 3, 6)
    res = add_teleconnections(df, lags=lags, indices=indices)
    added = [c for c in res.columns if c not in df.columns]

    span = (pd.to_datetime(df["date"]).min().date(),
            pd.to_datetime(df["date"]).max().date())
    print(f"[teleconnections] source={src} rows={len(res)} "
          f"dates {span[0]}..{span[1]} lags={lags}")
    print(f"[teleconnections] added {len(added)} index columns "
          f"({len(INDEX_COLS)} indices x {len(lags)} lags)")
    for col in added:
        cov = float(res[col].notna().mean())
        print(f"    {col:<14s} non-null coverage = {cov:6.1%}")


if __name__ == "__main__":
    main()
