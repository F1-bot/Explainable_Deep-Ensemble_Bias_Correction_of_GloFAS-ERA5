"""Bias-correction models.

Importing this package only pulls in the lightweight base interface.  Concrete
models (which may import torch / xgboost / lightgbm / catboost) live in
submodules and are imported on demand; :func:`load_all` registers them.
"""
from .base import BaseCorrector, available, get_model, register

__all__ = ["BaseCorrector", "register", "get_model", "available", "load_all"]


def load_all() -> list[str]:
    """Import every concrete model so it self-registers; return their names."""
    from importlib import import_module

    for mod in ("quantile_mapping", "boosting", "ea_lstm", "regime_prob_net", "ensemble",
                "robust", "sota_baselines", "probabilistic_baselines", "constraint_variants",
                "transfer_lstm", "snow_module", "generative_head", "cmal_head"):
        try:
            import_module(f"{__name__}.{mod}")
        except Exception:  # pragma: no cover - optional deps
            pass
    return available()
