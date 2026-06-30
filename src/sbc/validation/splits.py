"""Leakage-safe cross-validation splitters keyed on the schema id columns.

Bias-correction skill is only meaningful if the evaluation protocol forbids the
model from "seeing" — directly or by proxy — any information about the records it
is scored on.  In a sparse, spatially auto-correlated gauge network three
distinct leakage channels must be closed, and each is probed by a different
splitter in this module:

* **Temporal leakage** — training on future periods of a gauge and testing on its
  past.  :func:`temporal_split` holds out, per gauge, the *latest* ``test_frac``
  of the record so the test window is always strictly posterior to training,
  emulating an operational forecast-in-time setting.

* **Spatial (gauge) leakage** — a gauge contributing rows to both train and test.
  Neighbouring gauges share catchment attributes, regional climate and even the
  same GloFAS river pixels, so per-row splitting grossly over-states skill.
  :func:`spatial_folds` performs leave-one-basin-out (LOBO): an entire basin is
  withheld, guaranteeing no basin — and therefore no gauge — straddles the split.

* **Domain leakage / extrapolation** — :func:`pur_split` trains on the *core*
  Syr-Darya/Chu/Talas basins and tests on the hydrologically distinct, fully
  held-out *transfer* (Amu Darya) domain.

Why PUR is the strongest generalisation test
---------------------------------------------
Prediction in Ungauged Regions (PUR) — sometimes called the "differential
split-sample" or spatial-extrapolation test — is the most demanding of the three
because it removes **every** form of local information simultaneously: the test
basins contribute neither past observations (unlike the temporal split) nor
in-region neighbours (unlike random K-fold), and they lie outside the spatial
support of the training data.  For sparse mountain gauge networks this mirrors
the real deployment target — correcting GloFAS at locations and in catchments
where *no* discharge record exists — so a corrector that survives PUR has
demonstrated transferable, physically-grounded skill rather than memorised local
quirks.  LOBO is an interpolative weakening of PUR (the held-out basin still sits
*amongst* training basins of the same domain); temporal holdout is weaker still
because the gauge itself is in the training set.

For stacking, :func:`group_oof_folds` assigns a group-disjoint
(``sklearn.model_selection.GroupKFold``-style) fold id to every row so that base
learners can emit out-of-fold residual predictions on which a meta-learner is
trained without leakage.

Conventions
-----------
All splitters are deterministic and operate purely on the schema id columns
(:data:`sbc.schemas.ID_COLS`).  Boolean masks are returned as ``numpy`` arrays of
length ``len(df)`` aligned to the *positional* row order of ``df`` (so
``df[mask]`` / ``df.iloc[np.flatnonzero(mask)]`` selects the intended rows even
when ``df`` carries a non-default index).  No splitter ever places the same
gauge or basin on both sides of a *spatial* split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import ID_COLS
from ..utils import get_logger

log = get_logger(__name__)

GAUGE_COL = "code"
BASIN_COL = "basin"
DOMAIN_COL = "domain"
DATE_COL = "date"


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _require(df: pd.DataFrame, cols: str | list[str]) -> None:
    """Raise if any required id column is absent."""
    cols = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"modelling table missing column(s) {missing}; "
            f"id columns are {ID_COLS}"
        )


def _bool(n: int, true_idx: np.ndarray) -> np.ndarray:
    """Length-``n`` boolean mask with ``True`` at the given positional indices."""
    m = np.zeros(n, dtype=bool)
    m[true_idx] = True
    return m


# --------------------------------------------------------------------------- #
#  Temporal holdout (per gauge)                                               #
# --------------------------------------------------------------------------- #
def temporal_split(df: pd.DataFrame, test_frac: float = 0.3
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per-gauge temporal holdout: earliest ``1 - test_frac`` train, latest test.

    For every gauge the records are ordered by ``date`` and the most recent
    ``test_frac`` fraction is assigned to the test set, the remainder to train.
    The test window is therefore always strictly posterior to the training
    window *within each gauge*, preventing look-ahead (future-to-past) leakage.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table with at least the ``code`` and ``date`` columns.
    test_frac : float, default 0.3
        Fraction (0, 1) of each gauge's most recent records held out for test.

    Returns
    -------
    (train_mask, test_mask) : tuple of numpy.ndarray
        Boolean masks of length ``len(df)``, positionally aligned to ``df``.
        Gauges with a single record contribute that record to ``train`` only.

    Notes
    -----
    The schema guarantees one row per (gauge, period), so dates are unique within
    a gauge and the chronological cut is unambiguous.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    _require(df, [GAUGE_COL, DATE_COL])

    tmp = df.reset_index(drop=True)
    grp = tmp.groupby(GAUGE_COL, sort=False)[DATE_COL]
    # 0-based chronological position of each row within its gauge
    pos = grp.rank(method="first").to_numpy() - 1.0
    size = grp.transform("size").to_numpy().astype(float)

    n_test = np.rint(test_frac * size).astype(int)
    has_two = size >= 2
    # guarantee a non-empty, non-total test set whenever the gauge has >= 2 rows
    n_test = np.where(has_two, np.clip(n_test, 1, size.astype(int) - 1), 0)
    n_train = size.astype(int) - n_test

    test_mask = pos >= n_train
    train_mask = ~test_mask
    return train_mask, test_mask


# --------------------------------------------------------------------------- #
#  Spatial leave-one-group-out (LOBO)                                         #
# --------------------------------------------------------------------------- #
def spatial_folds(df: pd.DataFrame, group: str = "basin"
                  ) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-basin-out folds for spatial cross-validation.

    Each fold withholds all rows of exactly one group (basin by default); the
    remaining groups form the training set.  Because a group is the test set in
    full, no basin — and hence no gauge — appears on both sides of any fold.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table containing the grouping column.
    group : str, default "basin"
        Spatial grouping column (e.g. ``"basin"`` or ``"code"``).

    Returns
    -------
    list of (fold_name, train_mask, test_mask)
        One entry per unique group, ordered by group label for determinism.
    """
    _require(df, group)
    values = df[group].to_numpy()
    n = len(df)
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for g in sorted(pd.unique(values).tolist()):
        test_mask = values == g
        train_mask = ~test_mask
        folds.append((f"lobo[{group}={g}]", train_mask, test_mask))
    if not folds:
        log.warning("spatial_folds: no groups found in column '%s'", group)
    else:
        log.info("spatial_folds: %d leave-one-%s-out folds over %d rows",
                 len(folds), group, n)
    return folds


# --------------------------------------------------------------------------- #
#  Prediction in Ungauged Regions (domain extrapolation)                      #
# --------------------------------------------------------------------------- #
def pur_split(df: pd.DataFrame, train_domain: str = "core",
              test_domain: str = "transfer") -> tuple[np.ndarray, np.ndarray]:
    """Prediction-in-Ungauged-Regions split: train on core, test on transfer.

    Trains on the core Syr-Darya / Chu / Talas basins and evaluates on the
    spatially and hydrologically distinct, fully held-out transfer (Amu Darya)
    domain.  This is the framework's strongest generalisation test: the test
    domain shares no gauges, no neighbouring training gauges and no past records
    with the training set, mirroring deployment to genuinely ungauged regions
    (see the module docstring).

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table with a ``domain`` column.
    train_domain, test_domain : str
        Domain labels for the two sides (default ``"core"`` / ``"transfer"``).

    Returns
    -------
    (train_mask, test_mask) : tuple of numpy.ndarray
        Boolean masks of length ``len(df)``, positionally aligned to ``df``.
    """
    _require(df, DOMAIN_COL)
    dom = df[DOMAIN_COL].to_numpy()
    train_mask = dom == train_domain
    test_mask = dom == test_domain
    if not train_mask.any():
        log.warning("pur_split: no rows in train_domain=%r", train_domain)
    if not test_mask.any():
        log.warning("pur_split: no rows in test_domain=%r - the transfer domain "
                    "is absent from this table", test_domain)
    return train_mask, test_mask


# --------------------------------------------------------------------------- #
#  Group out-of-fold assignment for stacking                                  #
# --------------------------------------------------------------------------- #
def _greedy_group_assign(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Seeded, group-disjoint fold assignment (GroupKFold fallback).

    Groups are shuffled deterministically then greedily packed into the
    currently lightest fold, balancing fold sizes while keeping every group
    intact.  Used only when the installed scikit-learn predates the
    ``shuffle``/``random_state`` arguments of ``GroupKFold``.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    inv = np.ravel(inv)
    sizes = np.bincount(inv)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    # process largest groups first (stable within the seeded permutation)
    order = perm[np.argsort(-sizes[perm], kind="stable")]
    load = np.zeros(k, dtype=int)
    gfold = np.empty(len(uniq), dtype=int)
    for gi in order:
        f = int(np.argmin(load))
        gfold[gi] = f
        load[f] += int(sizes[gi])
    return gfold[inv]


def group_oof_folds(df: pd.DataFrame, n_splits: int = 5, group: str = "basin",
                    seed: int = 0) -> np.ndarray:
    """Assign a group-disjoint out-of-fold id to every row (for stacking).

    Mirrors ``sklearn.model_selection.GroupKFold``: each group (basin by
    default) is confined to a single fold, so base learners trained on the
    complement of a fold can produce leakage-free out-of-fold residual
    predictions for that fold.  Concatenating these across folds yields the
    out-of-fold matrix on which a stacked meta-learner is fitted.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table containing the grouping column.
    n_splits : int, default 5
        Requested number of folds; capped at the number of distinct groups.
    group : str, default "basin"
        Grouping column kept intact across folds.
    seed : int, default 0
        Seed controlling the (deterministic) shuffling of groups across folds.

    Returns
    -------
    numpy.ndarray
        Integer fold id in ``[0, n_folds)`` per row, positionally aligned to
        ``df``.  No group spans more than one fold.
    """
    _require(df, group)
    groups = df[group].to_numpy()
    n = len(df)
    n_groups = int(pd.unique(groups).size)
    if n_groups == 0:
        return np.zeros(0, dtype=int)
    k = min(int(n_splits), n_groups)
    if k < 2:
        log.warning("group_oof_folds: only %d distinct '%s' value(s); "
                    "out-of-fold stacking needs >= 2 groups, returning fold 0",
                    n_groups, group)
        return np.zeros(n, dtype=int)
    if k < n_splits:
        log.info("group_oof_folds: capping n_splits %d -> %d (only %d groups)",
                 n_splits, k, n_groups)

    from sklearn.model_selection import GroupKFold

    fold = np.full(n, -1, dtype=int)
    try:
        gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splitter = gkf.split(np.zeros((n, 1)), groups=groups)
        for f, (_, test_idx) in enumerate(splitter):
            fold[test_idx] = f
    except TypeError:  # pragma: no cover - older scikit-learn without shuffle
        fold = _greedy_group_assign(groups, k, seed)

    if (fold < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("group_oof_folds: some rows left unassigned")
    return fold


# --------------------------------------------------------------------------- #
#  Canonical evaluation matrix                                                #
# --------------------------------------------------------------------------- #
def combined_spatiotemporal(df: pd.DataFrame, test_frac: float = 0.3
                            ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Assemble the canonical evaluation matrix as named (train, test) masks.

    Bundles the three leakage-safe protocols used throughout the paper:

    * ``"temporal_holdout"`` — per-gauge latest-``test_frac`` holdout
      (:func:`temporal_split`);
    * ``"lobo[basin=...]"`` — one leave-one-basin-out fold per basin
      (:func:`spatial_folds`);
    * ``"pur_core_to_transfer"`` — core-to-transfer extrapolation
      (:func:`pur_split`).

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling table.
    test_frac : float, default 0.3
        Forwarded to :func:`temporal_split`.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Mapping from split name to ``(train_mask, test_mask)``.  PUR is omitted
        only when the transfer domain is absent from ``df``.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["temporal_holdout"] = temporal_split(df, test_frac=test_frac)
    for name, tr, te in spatial_folds(df, group=BASIN_COL):
        out[name] = (tr, te)
    if DOMAIN_COL in df.columns and (df[DOMAIN_COL] == "transfer").any():
        out["pur_core_to_transfer"] = pur_split(df)
    else:
        log.info("combined_spatiotemporal: transfer domain absent - "
                 "skipping the PUR split")
    return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from sbc.synthetic import generate

    df = generate(n_basins=5, gauges_per_basin=(2, 3), years=3, seed=0)
    n = len(df)
    print(f"synthetic table: {n} rows, {df['code'].nunique()} gauges, "
          f"{df['basin'].nunique()} basins")

    # --- temporal holdout --------------------------------------------------
    tr, te = temporal_split(df, test_frac=0.3)
    assert tr.shape == (n,) and te.shape == (n,)
    assert not (tr & te).any() and (tr | te).all(), "temporal masks must partition"
    # leakage check: every test date is strictly after that gauge's last train date
    for code, g in df.assign(_tr=tr, _te=te).groupby("code"):
        if g["_te"].any() and g["_tr"].any():
            assert g.loc[g["_te"], "date"].min() > g.loc[g["_tr"], "date"].max(), \
                f"temporal leakage in gauge {code}"
    print(f"temporal_split        train={tr.sum():5d} test={te.sum():5d}  "
          f"(test frac={te.mean():.2f}, no future->past leakage)")

    # --- leave-one-basin-out ----------------------------------------------
    folds = spatial_folds(df, group="basin")
    for name, ftr, fte in folds:
        assert not (ftr & fte).any(), f"{name}: train/test overlap"
        assert (ftr | fte).all(), f"{name}: masks do not cover all rows"
        tr_basins = set(df.loc[ftr, "basin"])
        te_basins = set(df.loc[fte, "basin"])
        assert tr_basins.isdisjoint(te_basins), f"{name}: basin on both sides"
    print(f"spatial_folds (LOBO)  {len(folds)} folds; "
          f"example test sizes={[int(fte.sum()) for _, _, fte in folds][:5]}; "
          f"all train/test basin-disjoint")

    # --- PUR (core -> transfer) -------------------------------------------
    # The synthetic generator emits only core basins, so fabricate a minimal
    # transfer slice (relabel the last basin's domain) to exercise PUR with
    # non-empty sides without depending on the real assembled table.
    df_pur = df.copy()
    held = sorted(df_pur["basin"].unique())[-1]
    df_pur.loc[df_pur["basin"] == held, "domain"] = "transfer"
    ptr, pte = pur_split(df_pur)
    assert pte.any() and ptr.any(), "PUR self-test needs both domains populated"
    assert set(df_pur.loc[ptr, "domain"]).isdisjoint(set(df_pur.loc[pte, "domain"]))
    assert set(df_pur.loc[ptr, "basin"]).isdisjoint(set(df_pur.loc[pte, "basin"]))
    print(f"pur_split             train={ptr.sum():5d} (core) "
          f"test={pte.sum():5d} (transfer={held})")

    # --- group out-of-fold assignment -------------------------------------
    fid = group_oof_folds(df, n_splits=5, group="basin", seed=0)
    assert fid.shape == (n,) and fid.min() >= 0
    # each basin lives in exactly one fold (no group split across folds)
    per_basin_folds = df.assign(_f=fid).groupby("basin")["_f"].nunique()
    assert (per_basin_folds == 1).all(), "a basin was split across OOF folds"
    sizes = np.bincount(fid)
    print(f"group_oof_folds       {fid.max() + 1} folds, "
          f"row counts={sizes.tolist()}, every basin in exactly one fold")

    # --- canonical evaluation matrix --------------------------------------
    matrix = combined_spatiotemporal(df)
    for name, (mtr, mte) in matrix.items():
        if name.startswith(("lobo", "pur")):
            assert not (mtr & mte).any(), f"{name}: spatial train/test overlap"
    print(f"combined_spatiotemporal {len(matrix)} named splits: "
          f"{list(matrix)[:3]}... (+PUR={'pur_core_to_transfer' in matrix})")

    print("OK: all splits deterministic and leakage-safe.")
