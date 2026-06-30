"""Project paths and global study configuration.

The repository root is resolved from the ``SBC_ROOT`` environment variable when
set (e.g. ``/content/drive/MyDrive/MDPI`` on Google Colab), otherwise it is
inferred as three levels above this file.  All other locations derive from it,
so the same code runs unchanged on Windows, Linux and Colab.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("SBC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Canonical filesystem layout."""

    root: Path

    # -- raw datasets -------------------------------------------------------
    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def ca_discharge(self) -> Path:
        return self.datasets / "ca_discharge"

    @property
    def glofas(self) -> Path:
        return self.datasets / "glofas_era5"

    @property
    def era5_land(self) -> Path:
        return self.datasets / "era5_land"

    @property
    def interim(self) -> Path:
        return self.datasets / "interim"

    @property
    def processed(self) -> Path:
        return self.datasets / "processed"

    # -- outputs ------------------------------------------------------------
    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def models_dir(self) -> Path:
        return self.results / "models"

    @property
    def shap_dir(self) -> Path:
        return self.results / "shap"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def ensure(self) -> "Paths":
        for p in (self.interim, self.processed, self.figures,
                  self.tables, self.models_dir, self.shap_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths(project_root())

# Canonical column names used throughout the modelling tables ---------------
COL_GAUGE = "code"          # gauge identifier (CA-discharge CODE, lower-cased)
COL_BASIN = "basin"         # basin / sub-region grouping for spatial CV
COL_DATE = "date"           # period representative date (datetime64)
COL_OBS = "q_obs"           # observed discharge [m3 s-1]
COL_SIM = "q_glofas"        # raw GloFAS-ERA5 discharge [m3 s-1]
COL_TARGET = "log_residual"  # ML target = log(q_obs+eps) - log(q_glofas+eps)
EPS = 1e-3                   # additive constant for the log transform [m3 s-1]

# Temporal scales handled by the framework ----------------------------------
SCALES = ("decadal", "daily")
