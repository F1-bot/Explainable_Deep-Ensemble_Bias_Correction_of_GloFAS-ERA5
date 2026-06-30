"""Hydrological evaluation metrics (pure NumPy, NaN-aware).

Deterministic skill: KGE' (Kling 2012) with its r / beta / gamma decomposition,
NSE, log-NSE, PBIAS, RMSE, plus the flow-duration-curve signatures FHV / FMS /
FLV (Yilmaz et al., 2008) and an annual peak-timing error.  Probabilistic skill:
ensemble and Gaussian CRPS.  ``evaluate`` returns the full deterministic bundle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MEAN_FLOW_BENCHMARK_KGE = 1.0 - np.sqrt(2.0)  # KGE' of the mean-flow benchmark


def _clean(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def kge_prime(obs, sim) -> dict[str, float]:
    """Modified Kling-Gupta efficiency and its components."""
    obs, sim = _clean(obs, sim)
    if obs.size < 3 or obs.std() == 0:
        return {"kge": np.nan, "r": np.nan, "beta": np.nan, "gamma": np.nan}
    r = np.corrcoef(obs, sim)[0, 1]
    mo, ms = obs.mean(), sim.mean()
    beta = ms / mo if mo != 0 else np.nan
    cv_o = obs.std() / mo if mo != 0 else np.nan
    cv_s = sim.std() / ms if ms != 0 else np.nan
    gamma = cv_s / cv_o if cv_o not in (0, np.nan) else np.nan
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return {"kge": float(kge), "r": float(r), "beta": float(beta), "gamma": float(gamma)}


def nse(obs, sim) -> float:
    obs, sim = _clean(obs, sim)
    if obs.size < 2:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((sim - obs) ** 2) / denom) if denom > 0 else np.nan


def lognse(obs, sim, eps: float = 1e-3) -> float:
    obs, sim = _clean(obs, sim)
    if obs.size < 2:
        return np.nan
    return nse(np.log(obs + eps), np.log(sim + eps))


def pbias(obs, sim) -> float:
    obs, sim = _clean(obs, sim)
    s = obs.sum()
    return float(100.0 * (sim - obs).sum() / s) if s != 0 else np.nan


def rmse(obs, sim) -> float:
    obs, sim = _clean(obs, sim)
    return float(np.sqrt(np.mean((sim - obs) ** 2))) if obs.size else np.nan


def kge_skill_score(kge: float, reference_kge: float = _MEAN_FLOW_BENCHMARK_KGE) -> float:
    """(KGE - KGE_ref) / (1 - KGE_ref); >0 means better than the reference."""
    if not np.isfinite(kge):
        return np.nan
    return float((kge - reference_kge) / (1.0 - reference_kge))


# --------------------------------------------------------------------------- #
#  Flow-duration-curve signatures (Yilmaz et al., 2008, WRR)                   #
# --------------------------------------------------------------------------- #
def fdc_fhv(obs, sim, h: float = 0.02) -> float:
    """% bias in high-flow volume (top ``h`` exceedance fraction)."""
    obs, sim = _clean(obs, sim)
    if obs.size < 5:
        return np.nan
    o = np.sort(obs)[::-1]; s = np.sort(sim)[::-1]
    k = max(1, int(np.ceil(h * o.size)))
    return float(100.0 * (s[:k].sum() - o[:k].sum()) / o[:k].sum())


def fdc_flv(obs, sim, l: float = 0.3, eps: float = 1e-3) -> float:
    """% bias in low-flow volume (bottom ``l`` fraction, log space)."""
    obs, sim = _clean(obs, sim)
    if obs.size < 5:
        return np.nan
    o = np.sort(obs); s = np.sort(sim)
    k = max(1, int(np.ceil(l * o.size)))
    lo = np.log(o[:k] + eps); ls = np.log(s[:k] + eps)
    lo -= lo.min(); ls -= ls.min()
    denom = lo.sum()
    return float(-100.0 * (ls.sum() - lo.sum()) / denom) if denom != 0 else np.nan


def fdc_fms(obs, sim, lo: float = 0.2, hi: float = 0.7, eps: float = 1e-3) -> float:
    """% bias in the mid-segment slope of the flow-duration curve."""
    obs, sim = _clean(obs, sim)
    if obs.size < 5:
        return np.nan
    o = np.sort(obs)[::-1]; s = np.sort(sim)[::-1]
    i1, i2 = int(lo * o.size), int(hi * o.size)
    i2 = min(max(i2, i1 + 1), o.size - 1)
    slope_o = np.log(o[i1] + eps) - np.log(o[i2] + eps)
    slope_s = np.log(s[i1] + eps) - np.log(s[i2] + eps)
    return float(100.0 * (slope_s - slope_o) / slope_o) if slope_o != 0 else np.nan


def peak_timing_error(dates, obs, sim) -> float:
    """Mean absolute error (days) of the annual peak-flow date."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "obs": obs, "sim": sim}).dropna()
    if df.empty:
        return np.nan
    df["year"] = df["date"].dt.year
    errs = []
    for _, g in df.groupby("year"):
        if len(g) < 3:
            continue
        do = g.loc[g["obs"].idxmax(), "date"]
        dsim = g.loc[g["sim"].idxmax(), "date"]
        errs.append(abs((do - dsim).days))
    return float(np.mean(errs)) if errs else np.nan


# --------------------------------------------------------------------------- #
#  Probabilistic                                                              #
# --------------------------------------------------------------------------- #
def crps_ensemble(obs, ensemble) -> float:
    """Mean CRPS for an ensemble forecast (obs: (n,), ensemble: (n, m))."""
    obs = np.asarray(obs, float)
    ens = np.asarray(ensemble, float)
    if ens.ndim == 1:
        ens = ens[:, None]
    m = np.isfinite(obs)
    obs, ens = obs[m], ens[m]
    if obs.size == 0:
        return np.nan
    term1 = np.abs(ens - obs[:, None]).mean(axis=1)
    term2 = np.abs(ens[:, :, None] - ens[:, None, :]).mean(axis=(1, 2))
    return float(np.mean(term1 - 0.5 * term2))


def crps_gaussian(obs, mu, sigma) -> float:
    """Closed-form Gaussian CRPS."""
    from math import sqrt, pi

    obs = np.asarray(obs, float); mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-6, None)
    z = (obs - mu) / sigma
    from scipy.stats import norm

    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / sqrt(pi))
    return float(np.nanmean(crps))


# --------------------------------------------------------------------------- #
#  Bundles                                                                    #
# --------------------------------------------------------------------------- #
def evaluate(obs, sim, dates=None) -> dict[str, float]:
    """Full deterministic metric bundle for one (obs, sim) pair."""
    k = kge_prime(obs, sim)
    out = {
        "kge": k["kge"], "kge_r": k["r"], "kge_beta": k["beta"], "kge_gamma": k["gamma"],
        "kge_ss": kge_skill_score(k["kge"]),
        "nse": nse(obs, sim), "lognse": lognse(obs, sim),
        "pbias": pbias(obs, sim), "rmse": rmse(obs, sim),
        "fhv": fdc_fhv(obs, sim), "fms": fdc_fms(obs, sim), "flv": fdc_flv(obs, sim),
        "n": int(np.isfinite(np.asarray(obs, float) + np.asarray(sim, float)).sum()),
    }
    if dates is not None:
        out["peak_timing_err"] = peak_timing_error(dates, obs, sim)
    return out


def evaluate_by_group(df: pd.DataFrame, obs_col: str, sim_col: str,
                      group: str = "code", date_col: str = "date") -> pd.DataFrame:
    """Per-group deterministic metrics -> tidy DataFrame (one row per group)."""
    rows = []
    for key, g in df.groupby(group):
        rec = {group: key}
        rec.update(evaluate(g[obs_col].values, g[sim_col].values,
                            g[date_col].values if date_col in g else None))
        rows.append(rec)
    return pd.DataFrame(rows)
