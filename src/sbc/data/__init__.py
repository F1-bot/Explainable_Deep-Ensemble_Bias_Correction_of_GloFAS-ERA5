"""Dataset access layer.

Submodules are imported lazily so that heavy optional dependencies
(geopandas, cdsapi, xarray) are only required by the code paths that use them.
"""
from . import domain  # noqa: F401

__all__ = ["domain"]
