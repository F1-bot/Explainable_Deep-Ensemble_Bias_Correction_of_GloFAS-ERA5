"""Abstract corrector interface and a small registry.

Every model in :mod:`sbc.models` predicts the **log-space residual** that, added
to ``log(q_glofas + EPS)`` and exponentiated, yields the bias-corrected
discharge.  This keeps all models on a common target and lets the stacked
ensemble combine them through their out-of-fold residual predictions.

Probabilistic models additionally implement :meth:`predict_quantiles` and/or
:meth:`sample`, enabling CRPS evaluation and uncertainty bands.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..schemas import SIM_COL, back_transform

_REGISTRY: dict[str, type["BaseCorrector"]] = {}


def register(cls: type["BaseCorrector"]) -> type["BaseCorrector"]:
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str) -> type["BaseCorrector"]:
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


class BaseCorrector(ABC):
    """Base class for all bias-correction models."""

    name: str = "base"
    is_probabilistic: bool = False

    @abstractmethod
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "BaseCorrector":
        ...

    @abstractmethod
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Return the predicted log-space residual (point estimate)."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return bias-corrected discharge [m3 s-1]."""
        return back_transform(df[SIM_COL].values, self.predict_residual(df))

    # -- optional probabilistic API ----------------------------------------
    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predicted residual quantiles, shape (n_samples, n_quantiles)."""
        raise NotImplementedError

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Posterior residual samples, shape (n_samples, n)."""
        raise NotImplementedError

    def predict_discharge_quantiles(self, df, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        rq = self.predict_quantiles(df, quantiles)
        sim = df[SIM_COL].values[:, None]
        return back_transform(np.repeat(sim, rq.shape[1], axis=1), rq)
