"""Reproducible seeding across random / numpy / torch."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 1234, deterministic_torch: bool = True) -> int:
    """Seed all relevant RNGs and return the seed.

    Sets ``PYTHONHASHSEED`` and (when available) configures PyTorch for
    deterministic behaviour.  TensorFlow, if used, is seeded as well.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch optional at import time
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover
        pass

    return seed
