"""Credential handling and client factory for the Copernicus CDS / EWDS APIs.

Credentials are read from ``configs/secrets.env`` (git-ignored) or, with higher
priority, from the process environment.  A single ECMWF Personal Access Token is
typically valid on both portals; only the endpoint URL differs.
"""
from __future__ import annotations

import os

from ..config import PATHS
from ..utils import get_logger

log = get_logger(__name__)


def load_secrets() -> dict[str, str]:
    env: dict[str, str] = {}
    p = PATHS.configs / "secrets.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    for k in ("CDS_URL", "CDS_KEY", "EWDS_URL", "EWDS_KEY"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def make_client(service: str = "cds"):
    """Return a ``cdsapi.Client`` for ``service`` in {``cds``, ``ewds``}."""
    import cdsapi

    s = load_secrets()
    prefix = "EWDS" if service.lower() == "ewds" else "CDS"
    url, key = s.get(f"{prefix}_URL"), s.get(f"{prefix}_KEY")
    if not key or "PASTE" in key:
        raise RuntimeError(
            f"Missing {prefix} credentials. Edit configs/secrets.env and set "
            f"{prefix}_KEY to your Copernicus Personal Access Token."
        )
    return cdsapi.Client(url=url, key=key, quiet=True, progress=False)


# errors that will never succeed on retry (request must be changed instead)
_FATAL = ("cost limits", "too large", "licence", "license", "invalid",
          "not valid", "forbidden")


def retrieve_with_retry(client, dataset: str, request: dict, target: str,
                        attempts: int = 4, base_sleep: int = 20):
    """``client.retrieve`` with backoff on transient (network/queue) errors.

    Fatal errors (cost-limit, malformed request, missing licence) are raised
    immediately so the caller can fix the request rather than loop.
    """
    import time

    last = None
    for i in range(attempts):
        try:
            client.retrieve(dataset, request, target)
            return target
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(tok in msg for tok in _FATAL):
                raise
            last = exc
            log.warning("retrieve attempt %d/%d failed (%s); retrying...",
                        i + 1, attempts, exc)
            time.sleep(base_sleep * (i + 1))
    raise last  # type: ignore[misc]
