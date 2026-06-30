"""Definition of the Central-Asian study domain.

The study targets the snow-/glacier-influenced transboundary headwater basins of
the **Syr Darya** system together with the **Chu** and **Talas** basins that feed
the Zhambyl agricultural region (grant BR24993128).  The hydrologically distinct
**Amu Darya** tributaries are held out as an independent transfer / prediction-in-
ungauged-regions (PUR) domain.

Basin labels follow the ``BASIN`` field of the CA-discharge data set
(Marti et al., 2023, *Scientific Data*).
"""
from __future__ import annotations

# -- core domain: Syr Darya system + Chu + Talas ----------------------------
CORE_BASINS: tuple[str, ...] = (
    "SYR_DARYA",   # Syr Darya main stem & Fergana tributaries
    "NARYN",       # Naryn headwaters (largest Syr Darya source)
    "CHU",         # Chu basin (Zhambyl, transboundary KGZ<->KAZ)
    "TALAS",       # Talas basin (Zhambyl, transboundary KGZ<->KAZ)
    "CHIRCHIK",    # Chirchik (Pskem/Chatkal) - snow & glacier fed
    "QASHQADARYA",  # Qashqadarya tributaries
    "AKHANGARAN",  # Akhangaran
)

# -- transfer domain: Amu Darya system (independent generalisation test) -----
TRANSFER_BASINS: tuple[str, ...] = (
    "PYANDZH",
    "VAKSH",
    "KOFARNIKHAN",
    "ZERAFSHAN",
    "SURKHANDARYA",
)

ALL_STUDY_BASINS: tuple[str, ...] = CORE_BASINS + TRANSFER_BASINS

# Download bounding box for GloFAS-ERA5 / ERA5-Land  [deg]; order N, W, S, E.
# Generous margin so every gauge's contributing river pixel is covered.
STUDY_BBOX = {"north": 44.0, "west": 65.0, "south": 37.0, "east": 78.0}

# Temporal overlap window with GloFAS-ERA5 (reanalysis starts 1979-01-01).
PERIOD = {"start": "1979-01-01", "end": "2020-12-31"}

# Minimum number of valid post-1979 years for a gauge to enter the study.
MIN_YEARS = {"decadal": 8.0, "daily": 8.0}


def domain_of(basin: str | None) -> str | None:
    """Return ``"core"``, ``"transfer"`` or ``None`` for a basin label."""
    if basin is None:
        return None
    b = str(basin).upper()
    if b in CORE_BASINS:
        return "core"
    if b in TRANSFER_BASINS:
        return "transfer"
    return None
