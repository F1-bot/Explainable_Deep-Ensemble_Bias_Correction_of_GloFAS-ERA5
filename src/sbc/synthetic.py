"""Physically-grounded synthetic benchmark with a *known* GloFAS-style bias.

This module generates an analysis-ready modelling table identical in schema to
the assembled real-data table (:mod:`sbc.data.assemble`).  Each gauge follows a
simple but physically meaningful nival-glacial water balance: a snow store
accumulates in winter and releases a degree-day-driven freshet in spring, a
glacier contributes summer melt, and soil moisture sustains baseflow.  A
*pseudo-GloFAS* series is then produced from the truth by injecting the error
signatures documented for GloFAS-ERA5 in snow-dominated catchments:

* a systematic negative volume bias (global median ~ -16 %, larger in
  small mountain basins),
* an early-shifted, damped snowmelt freshet (simplified snow physics),
* compressed flow variability.

Because the injected bias is known, the synthetic experiment lets us (i) smoke-
test the entire pipeline before the real reanalysis finishes downloading and
(ii) verify, in a controlled setting, that the framework recovers a known bias.

It is **not** a substitute for the real experiment and is always reported as a
controlled synthetic benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.domain import CORE_BASINS, TRANSFER_BASINS

# dynamic forcing columns (match ERA5-Land tags in sbc.data.era5_land)
FORCING = ["t2m_mean", "t2m_min", "t2m_max", "tp", "sf", "smlt", "swe", "swvl1"]
STATIC = ["area_km2", "elev_m", "slope_deg", "snow_frac", "glacier_frac", "aridity"]


def _gauge_params(rng: np.random.Generator) -> dict:
    elev = rng.uniform(1500, 3800)                      # mean catchment elevation [m]
    area = float(np.exp(rng.uniform(np.log(50), np.log(8000))))  # km2
    snow_frac = float(np.clip(rng.normal(0.5 + (elev - 1500) / 5000, 0.12), 0.1, 0.95))
    glacier_frac = float(np.clip(rng.exponential(0.04) * (elev > 2800), 0, 0.35))
    return {
        "area_km2": area, "elev_m": elev,
        "slope_deg": float(np.clip(rng.normal(12, 4), 2, 35)),
        "snow_frac": snow_frac, "glacier_frac": glacier_frac,
        "aridity": float(np.clip(rng.normal(1.2, 0.4), 0.4, 3.0)),
    }


def _simulate_gauge(dates: pd.DatetimeIndex, p: dict, rng: np.random.Generator):
    """Return (q_true, forcing_dict) for one gauge over ``dates``."""
    n = len(dates)
    doy = dates.dayofyear.values.astype(float)
    year_frac = 2 * np.pi * doy / 365.25

    # --- temperature: seasonal + elevation lapse + AR(1) weather noise --------
    lapse = (p["elev_m"] - 1500) * 0.0065
    t_season = 12.0 - lapse + 13.0 * np.cos(year_frac - np.pi)        # warm in summer
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 2.2)
    t2m = t_season + noise
    dtr = rng.uniform(6, 11)                                          # diurnal range
    t2m_min, t2m_max = t2m - dtr / 2, t2m + dtr / 2

    # --- precipitation: gamma occurrence, wetter in cold half ----------------
    p_wet = 0.25 + 0.15 * np.cos(year_frac - np.pi * 0.9)
    wet = rng.random(n) < p_wet
    precip = wet * rng.gamma(shape=0.7, scale=6.0, size=n)            # mm/day

    # --- snow store & degree-day melt ----------------------------------------
    swe = np.zeros(n); smlt = np.zeros(n); snowfall = np.zeros(n)
    ddf = rng.uniform(3.0, 6.0)                                       # mm/degC/day
    t_snow, t_melt = 1.0, 0.0
    s = 0.0
    for i in range(n):
        sf = precip[i] if t2m[i] < t_snow else 0.0
        rain = precip[i] - sf
        s += sf
        melt = min(s, ddf * max(t2m[i] - t_melt, 0.0))
        s -= melt
        swe[i], smlt[i], snowfall[i] = s, melt, sf
        precip[i] = rain + sf            # keep total precip; rain split implicit

    # --- glacier melt (summer, exposed ice once snow is gone) ----------------
    glac = p["glacier_frac"] * np.maximum(t2m - 2.0, 0.0) * (swe < 5.0) * 7.0

    # --- soil moisture bucket & runoff generation ----------------------------
    rain = np.where(t2m >= t_snow, precip - snowfall, 0.0)
    et = np.clip(0.2 * np.maximum(t2m, 0.0) / p["aridity"], 0, None)
    soil = np.zeros(n); fast = np.zeros(n); cap = 120.0
    sm = 40.0
    for i in range(n):
        sm += rain[i] + smlt[i] - et[i]
        over = max(sm - cap, 0.0); sm = min(sm, cap); sm = max(sm, 0.0)
        fast[i] = 0.35 * (rain[i] + smlt[i]) + over
        soil[i] = sm
    baseflow = 0.04 * soil
    runoff_mm = baseflow + 0.6 * smlt + glac + 0.5 * fast             # mm/day

    # linear-reservoir routing (smoothing)
    q = np.zeros(n); k = 0.45
    for i in range(1, n):
        q[i] = k * q[i - 1] + (1 - k) * runoff_mm[i]
    # convert mm/day over area to m3/s
    q_true = q * p["area_km2"] * 1e3 / 86.4
    q_true = np.clip(q_true, 1e-3, None)

    forcing = {
        "t2m_mean": t2m, "t2m_min": t2m_min, "t2m_max": t2m_max,
        "tp": precip, "sf": snowfall, "smlt": smlt, "swe": swe,
        "swvl1": np.clip(soil / cap, 0, 1),
    }
    return q_true, forcing


def _pseudo_glofas(dates, q_true, forcing, rng):
    """Inject GloFAS-style snow-region error signatures into the truth."""
    smlt = forcing["smlt"]
    # early-shifted melt: move a fraction of melt-driven flow ~12 days earlier
    shift = rng.integers(8, 18)
    melt_signal = 0.6 * smlt
    early = np.zeros_like(q_true)
    early[:-shift] = melt_signal[shift:] - melt_signal[:-shift]
    damp = rng.uniform(0.75, 0.9)                       # compress variability
    qm = q_true.mean()
    q = qm + damp * (q_true - qm) + 0.25 * early * qm / (melt_signal.mean() + 1e-6)
    bias = rng.uniform(0.60, 0.85)                      # systematic -15..-40 %
    q = bias * q * np.exp(rng.normal(0, 0.08, size=len(q)))
    return np.clip(q, 1e-3, None)


def generate(n_basins: int = 7, gauges_per_basin: tuple[int, int] = (3, 10),
             years: int = 25, start: str = "1990-01-01",
             scale: str = "daily", seed: int = 1234) -> pd.DataFrame:
    """Generate a synthetic modelling table with a known GloFAS-style bias."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=years * 365, freq="D")
    # core basins plus two held-out transfer basins so the PUR split is exercised
    basins = [(b, "core") for b in CORE_BASINS[:n_basins]] \
        + [(b, "transfer") for b in TRANSFER_BASINS[:2]]
    rows = []
    gid = 0
    for b, dom in basins:
        ng = int(rng.integers(gauges_per_basin[0], gauges_per_basin[1] + 1))
        for _ in range(ng):
            grng = np.random.default_rng(seed * 1000 + gid)
            p = _gauge_params(grng)
            q_true, forcing = _simulate_gauge(dates, p, grng)
            q_glofas = _pseudo_glofas(dates, q_true, forcing, grng)
            df = pd.DataFrame({"code": f"S{gid:03d}", "basin": b, "domain": dom,
                               "scale": scale, "date": dates,
                               "q_obs": q_true, "q_glofas": q_glofas})
            for k, v in forcing.items():
                df[k] = v
            for k in STATIC:
                df[k] = p[k]
            rows.append(df)
            gid += 1
    out = pd.concat(rows, ignore_index=True)
    if scale == "decadal":
        out = decadal_aggregate(out)
    return out


def decadal_aggregate(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily table to Central-Asian decades (days 1-10/11-20/21-end)."""
    df = daily.copy()
    d = df["date"].dt.day
    dec = np.where(d <= 10, 5, np.where(d <= 20, 15, 25))
    df["date"] = pd.to_datetime(dict(year=df["date"].dt.year,
                                     month=df["date"].dt.month, day=dec))
    static = set(STATIC) | {"basin", "domain", "scale"}
    agg = {c: "mean" for c in df.columns if c not in {"code", "date"} | static}
    agg.update({c: "first" for c in static})
    out = df.groupby(["code", "date"], as_index=False).agg(agg)
    out["scale"] = "decadal"
    return out
