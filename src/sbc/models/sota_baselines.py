"""State-of-the-art comparison baselines for the bias-correction benchmark.

A frequent and fair reviewer objection to any post-processing study is that the
proposed model is only ever shown to beat the *raw* product — here GloFAS-ERA5.
To close that gap this module implements two **published-method** baselines that
the flagship deep ensemble must out-perform on its own terms:

* :class:`GloFASLSTMCorrector` (``"glofas_lstm"``) — a *GloFAS-only* LSTM in the
  spirit of Hunt et al. (2022, *HESS* 26:5449, doi:10.5194/hess-26-5449-2022),
  who post-process model discharge with an LSTM that sees the simulated
  hydrograph (and static attributes) but **no** meteorological / snow forcing.
  We realise it by wrapping :class:`~sbc.models.ea_lstm.EALSTMCorrector` and
  masking its dynamic inputs down to the GloFAS-derived columns plus the static
  catchment attributes, so the comparison cleanly isolates *"GloFAS memory only"*
  from our full snow-aware model.  The masking is factored into a reusable
  :class:`FeatureMaskedCorrector` helper.

* :class:`DonorRegionalizationCorrector` (``"donor"``) — a SABER-style
  attribute-donor regionalization baseline (Hales et al., 2023, *Environ. Model.
  Softw.*; "Stream Analysis for Bias Estimation and Reduction") for the
  prediction-in-ungauged-regions (PUR) track.  Each *training* gauge contributes
  a seasonal multiplicative bias-correction factor ``mean(q_obs)/mean(q_glofas)``;
  an ungauged gauge inherits the distance-weighted mean of the seasonal factors
  of its ``k`` nearest training gauges in standardised static-attribute space.
  This is the regionalization reference our flagship's PUR skill is measured
  against (SABER reports a global median KGE ≈ 0.47).

Both baselines predict the framework's common **log-residual** target and are
fitted on the training split only (strictly leakage-safe).
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..config import EPS
from ..schemas import (
    OBS_COL,
    SIM_COL,
    feature_columns,
    static_feature_columns,
    validate,
)
from ..utils import get_logger
from .base import BaseCorrector, register
from .ea_lstm import EALSTMCorrector

log = get_logger(__name__)

# Substring/regex patterns identifying GloFAS-derived dynamic feature columns
# (the raw simulated discharge and its engineered lags / rolling moments / roc).
_GLOFAS_KEEP_PATTERNS: tuple[str, ...] = ("qglofas", "q_glofas")

# Guard rails to keep the multiplicative donor factor numerically sane.
_FACTOR_CLIP = (1e-3, 1e3)


def _matches(col: str, patterns: Sequence[str]) -> bool:
    """Return ``True`` if ``col`` matches any of ``patterns`` (case-insensitive)."""
    return any(re.search(p, col, flags=re.IGNORECASE) for p in patterns)


# --------------------------------------------------------------------------- #
#  Reusable feature-masking wrapper                                           #
# --------------------------------------------------------------------------- #
class FeatureMaskedCorrector(BaseCorrector):
    """Wrap any base corrector and restrict the feature columns it may see.

    The wrapper drops a fixed set of *feature* columns from the modelling table
    before delegating to a freshly built base corrector, so that ablations such
    as *"GloFAS memory only"* reduce to a one-line factory plus keep/drop
    patterns.  Identity, target and regime columns are never touched; only the
    numeric model-input columns discovered by :func:`sbc.schemas.feature_columns`
    are eligible for masking.

    Parameters
    ----------
    base_factory : Callable[[], BaseCorrector]
        Zero-argument factory returning a *fresh, unfitted* base corrector.
    keep : sequence of str, optional
        Regex patterns; in *whitelist* mode a dynamic feature column is retained
        only if it matches one of these (static columns are governed by
        ``keep_static``).  ``None`` disables the whitelist.
    drop : sequence of str, optional
        Regex patterns force-dropping any matching feature column (applied before
        the whitelist, but after the ``keep_static`` exemption).
    keep_static : bool, default True
        Always retain per-gauge-constant static attributes regardless of the
        keep/drop patterns.

    Notes
    -----
    The masked column set is resolved **once at fit time** from the training
    table and reused verbatim at predict time, so train and inference always see
    an identical feature space.  Predictions are returned positionally aligned to
    the rows of the table passed to :meth:`predict_residual`.
    """

    name = "feature_masked"
    is_probabilistic = False

    def __init__(self, base_factory: Callable[[], BaseCorrector],
                 keep: Sequence[str] | None = None,
                 drop: Sequence[str] | None = None,
                 keep_static: bool = True) -> None:
        if not callable(base_factory):
            raise TypeError("base_factory must be a zero-argument callable")
        self.base_factory = base_factory
        self.keep = list(keep) if keep else None
        self.drop = list(drop) if drop else None
        self.keep_static = bool(keep_static)

        # learned state
        self.base_: BaseCorrector | None = None
        self.dropped_cols_: list[str] = []
        self.kept_features_: list[str] = []

    # -- sklearn-style introspection (enables OOF cloning in the ensemble) ---
    def get_params(self) -> dict:
        return {"base_factory": self.base_factory, "keep": self.keep,
                "drop": self.drop, "keep_static": self.keep_static}

    # -- masking -------------------------------------------------------------
    def _resolve_dropped(self, df: pd.DataFrame) -> list[str]:
        """Decide which feature columns to drop given the keep/drop policy."""
        feats = feature_columns(df)
        static = set(static_feature_columns(df, feats)) if self.keep_static else set()
        dropped: list[str] = []
        for c in feats:
            if c in static:
                continue                                   # 1. statics exempt
            if self.drop and _matches(c, self.drop):
                dropped.append(c)                          # 2. force-drop
                continue
            if self.keep is not None:                      # 3. whitelist mode
                if not _matches(c, self.keep):
                    dropped.append(c)
        return dropped

    def _mask(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        return df.drop(columns=self.dropped_cols_, errors="ignore")

    # -- fit / predict -------------------------------------------------------
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "FeatureMaskedCorrector":
        """Resolve the mask on ``train`` and fit the base on the masked table."""
        train = validate(train)
        self.dropped_cols_ = self._resolve_dropped(train)
        masked = self._mask(train)
        self.kept_features_ = feature_columns(masked)
        base = self.base_factory()
        base.fit(masked, self._mask(valid))
        self.base_ = base
        log.info("%s: kept %d/%d feature columns (dropped %d) for base '%s'",
                 self.name, len(self.kept_features_),
                 len(self.kept_features_) + len(self.dropped_cols_),
                 len(self.dropped_cols_), getattr(base, "name", "?"))
        return self

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-residual, aligned 1:1 to the rows of ``df``."""
        if self.base_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        out = np.asarray(self.base_.predict_residual(self._mask(df)), float).ravel()
        if out.shape[0] != len(df):  # pragma: no cover - defensive
            raise ValueError("masked base prediction length mismatch")
        return out


# --------------------------------------------------------------------------- #
#  (A) GloFAS-only LSTM baseline (Hunt-et-al-2022 style)                       #
# --------------------------------------------------------------------------- #
@register
class GloFASLSTMCorrector(FeatureMaskedCorrector):
    """LSTM post-processor that sees **only** GloFAS-derived dynamics + statics.

    A feature-masked :class:`~sbc.models.ea_lstm.EALSTMCorrector` whose dynamic
    inputs are restricted to the raw GloFAS discharge and its engineered memory
    (lags, causal rolling mean/std, rate-of-change — every column matching
    :data:`_GLOFAS_KEEP_PATTERNS`), while the static catchment attributes are
    retained.  All meteorological and snow forcing is excluded, so contrasting
    this model with the full snow-aware ensemble quantifies exactly what the
    forcing buys over *"learn the GloFAS error from GloFAS's own memory"*.

    The constructor mirrors :class:`EALSTMCorrector`; the hyper-parameters are
    forwarded verbatim to the wrapped network.
    """

    name = "glofas_lstm"
    is_probabilistic = False

    def __init__(self, seq_length: int | None = None, hidden_size: int = 64,
                 dropout: float = 0.4, max_epochs: int = 100, batch_size: int = 256,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-6,
                 patience: int = 15, valid_fraction: float = 0.2,
                 seed: int = 1234, device: str | None = None) -> None:
        self.seq_length = seq_length
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.valid_fraction = float(valid_fraction)
        self.seed = int(seed)
        self.device = device

        def _factory() -> EALSTMCorrector:
            return EALSTMCorrector(
                seq_length=self.seq_length, hidden_size=self.hidden_size,
                dropout=self.dropout, max_epochs=self.max_epochs,
                batch_size=self.batch_size, learning_rate=self.learning_rate,
                weight_decay=self.weight_decay, patience=self.patience,
                valid_fraction=self.valid_fraction, seed=self.seed,
                device=self.device,
            )

        super().__init__(base_factory=_factory, keep=_GLOFAS_KEEP_PATTERNS,
                         drop=None, keep_static=True)

    # Expose the EA-LSTM hyper-parameters so the ensemble can clone us.
    def get_params(self) -> dict:
        return {"seq_length": self.seq_length, "hidden_size": self.hidden_size,
                "dropout": self.dropout, "max_epochs": self.max_epochs,
                "batch_size": self.batch_size, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "patience": self.patience,
                "valid_fraction": self.valid_fraction, "seed": self.seed,
                "device": self.device}


# --------------------------------------------------------------------------- #
#  (B) SABER-style attribute-donor regionalization baseline                   #
# --------------------------------------------------------------------------- #
def _bias_factor(obs_sum: float, sim_sum: float) -> float:
    """Volume-preserving multiplicative factor ``obs/sim``, guarded and clipped."""
    if not np.isfinite(obs_sum) or not np.isfinite(sim_sum) or sim_sum <= EPS:
        return 1.0
    return float(np.clip(obs_sum / sim_sum, *_FACTOR_CLIP))


@register
class DonorRegionalizationCorrector(BaseCorrector):
    """Seasonal multiplicative bias correction transferred by attribute donors.

    Fitting (training split only) stores, per training gauge, a seasonal
    multiplicative factor ``mean(q_obs) / mean(q_glofas)`` and the gauge's
    standardised static catchment attributes.  At prediction:

    * a **gauged** target (its ``code`` appears in training) uses its own
      seasonal factors;
    * an **ungauged** target borrows the *distance-weighted* mean of the seasonal
      factors of its ``k`` nearest training gauges in standardised
      static-attribute (Euclidean) space.

    Missing seasons cascade to a pooled-season factor and finally a single global
    factor, so every row receives a correction.  The query gauge's *attributes*
    are exogenous, and only *training* observations enter the factors, so the
    method is strictly leakage-safe — the regionalization analogue of SABER used
    as the PUR reference baseline.

    Parameters
    ----------
    k : int, default 5
        Number of nearest training donors blended for an ungauged target.
    """

    name = "donor"
    is_probabilistic = False

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)

        # learned state
        self.scale_: str = "decadal"
        self.static_cols_: list[str] = []
        self.attr_mean_: np.ndarray = np.zeros(0)
        self.attr_std_: np.ndarray = np.ones(0)
        self.donor_codes_: list[str] = []
        self.donor_attr_: np.ndarray = np.zeros((0, 0))
        self.gauge_factors_: dict[str, dict[int, float]] = {}
        self.pooled_factors_: dict[int, float] = {}
        self.global_factor_: float = 1.0
        self._fitted = False

    def get_params(self) -> dict:
        return {"k": self.k}

    # -- seasonal keying -----------------------------------------------------
    def _season_key(self, df: pd.DataFrame, scale: str | None = None) -> np.ndarray:
        """Per-row season id: decade-of-year (0..35) for decadal, else month."""
        scale = scale or self.scale_
        dates = pd.to_datetime(df["date"])
        if scale == "decadal":
            month = dates.dt.month.to_numpy()
            day = dates.dt.day.to_numpy()
            third = np.where(day <= 10, 0, np.where(day <= 20, 1, 2))
            return ((month - 1) * 3 + third).astype(int)
        return dates.dt.month.to_numpy().astype(int)

    # -- fit -----------------------------------------------------------------
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "DonorRegionalizationCorrector":
        train = validate(train)
        self.scale_ = (str(train["scale"].iloc[0])
                       if "scale" in train.columns and len(train) else "decadal")
        self.static_cols_ = static_feature_columns(train)

        # --- standardised per-gauge static attributes (donor coordinates) ----
        if self.static_cols_:
            per_gauge = train.groupby("code", sort=True)[self.static_cols_].mean()
            self.donor_codes_ = [str(c) for c in per_gauge.index]
            raw = per_gauge.to_numpy(float)
            self.attr_mean_ = np.nanmean(raw, axis=0)
            std = np.nanstd(raw, axis=0)
            self.attr_std_ = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
            self.donor_attr_ = np.nan_to_num(
                (raw - self.attr_mean_) / self.attr_std_)
        else:
            self.donor_codes_ = [str(c) for c in pd.unique(train["code"])]
            self.donor_attr_ = np.zeros((len(self.donor_codes_), 0))
            self.attr_mean_ = np.zeros(0)
            self.attr_std_ = np.ones(0)

        # --- seasonal multiplicative factors per gauge + pooled / global ----
        work = train[["code", "date", OBS_COL, SIM_COL]].copy()
        work["_season"] = self._season_key(work)
        work = work[np.isfinite(work[OBS_COL]) & np.isfinite(work[SIM_COL])]

        self.gauge_factors_ = {}
        for (code, season), cell in work.groupby(["code", "_season"], sort=False):
            f = _bias_factor(cell[OBS_COL].sum(), cell[SIM_COL].sum())
            self.gauge_factors_.setdefault(str(code), {})[int(season)] = f

        self.pooled_factors_ = {
            int(season): _bias_factor(cell[OBS_COL].sum(), cell[SIM_COL].sum())
            for season, cell in work.groupby("_season", sort=False)
        }
        self.global_factor_ = _bias_factor(work[OBS_COL].sum(), work[SIM_COL].sum())

        self._fitted = True
        log.info("donor fitted: %d donor gauges, %d static attrs, k=%d, global=%.3f",
                 len(self.donor_codes_), len(self.static_cols_), self.k,
                 self.global_factor_)
        return self

    # -- donor lookup --------------------------------------------------------
    def _query_attr(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised static-attribute vector for the (single-gauge) ``df``."""
        if not self.static_cols_:
            return np.zeros(0)
        vals = df[self.static_cols_].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        z = (vals.to_numpy(float) - self.attr_mean_) / self.attr_std_
        return np.nan_to_num(z)

    def _donor_factor_map(self, query_attr: np.ndarray) -> dict[int, float]:
        """Distance-weighted seasonal factors of the ``k`` nearest train donors."""
        if self.donor_attr_.shape[0] == 0 or self.donor_attr_.shape[1] == 0:
            return {}
        dist = np.sqrt(((self.donor_attr_ - query_attr[None, :]) ** 2).sum(axis=1))
        k = int(min(self.k, dist.size))
        nearest = np.argsort(dist, kind="stable")[:k]
        weights = 1.0 / (dist[nearest] + 1e-6)

        combined: dict[int, float] = {}
        seasons: set[int] = set()
        for j in nearest:
            seasons.update(self.gauge_factors_.get(self.donor_codes_[j], {}))
        for season in seasons:
            num = den = 0.0
            for j, w in zip(nearest, weights):
                f = self.gauge_factors_.get(self.donor_codes_[j], {}).get(season)
                if f is not None and np.isfinite(f):
                    num += w * f
                    den += w
            if den > 0:
                combined[season] = num / den
        return combined

    def _factor_for_season(self, fmap: dict[int, float], season: int) -> float:
        if season in fmap:
            return fmap[season]
        if season in self.pooled_factors_:
            return self.pooled_factors_[season]
        return self.global_factor_

    # -- predict -------------------------------------------------------------
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-residual from the (own or donor) seasonal factors."""
        if not self._fitted:
            raise RuntimeError("DonorRegionalizationCorrector.fit must be called first")

        sim = df[SIM_COL].to_numpy(float)
        codes = df["code"].astype(str).to_numpy()
        seasons = self._season_key(df)
        factors = np.ones(len(df), float)

        for code in pd.unique(codes):
            idx = np.flatnonzero(codes == code)
            if code in self.gauge_factors_:
                fmap = self.gauge_factors_[code]            # gauged: own factors
            else:                                            # ungauged: donors
                fmap = self._donor_factor_map(self._query_attr(df.iloc[idx]))
            factors[idx] = [self._factor_for_season(fmap, int(s))
                            for s in seasons[idx]]

        q_corr = np.clip(sim, 0.0, None) * factors
        return np.log(q_corr + EPS) - np.log(np.clip(sim, 0.0, None) + EPS)


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sbc.features.engineering import build_features
    from sbc.features.regimes import classify_regimes
    from sbc.synthetic import generate
    from sbc.validation.metrics import kge_prime
    from sbc.validation.splits import pur_split, temporal_split

    # Small synthetic decadal table with engineered features + regimes.
    df = generate(scale="decadal", years=8, n_basins=3,
                  gauges_per_basin=(2, 3), seed=0)
    df = build_features(df, "decadal")
    df = classify_regimes(df)
    df = validate(df)
    print(f"synthetic: {df['code'].nunique()} gauges / {df['basin'].nunique()} "
          f"basins / {len(df)} rows "
          f"(domains: {sorted(df['domain'].unique())})")

    def _kge(obs, sim):
        return kge_prime(np.asarray(obs, float), np.asarray(sim, float))["kge"]

    # -- (A) GloFAS-only LSTM on a temporal holdout -------------------------
    tr_m, te_m = temporal_split(df, test_frac=0.3)
    train, test = df[tr_m].reset_index(drop=True), df[te_m].reset_index(drop=True)

    glstm = GloFASLSTMCorrector(seq_length=6, hidden_size=16, max_epochs=3,
                                batch_size=128, patience=3, seed=0)
    glstm.fit(train)
    resid = glstm.predict_residual(test)
    q_glstm = glstm.predict(test)
    masked_only_glofas = all(_matches(c, _GLOFAS_KEEP_PATTERNS)
                             for c in glstm.kept_features_
                             if c not in static_feature_columns(train))
    assert len(resid) == len(test), "glofas_lstm residual not aligned to df rows"
    assert glstm.dropped_cols_, "glofas_lstm dropped no features (mask inactive)"
    assert masked_only_glofas, "glofas_lstm kept a non-GloFAS dynamic feature"
    print(f"[glofas_lstm] kept={len(glstm.kept_features_)} "
          f"dropped={len(glstm.dropped_cols_)} dyn_features="
          f"{[c for c in glstm.kept_features_ if 'qglofas' in c]}")

    kge_raw_t = _kge(test[OBS_COL], test[SIM_COL])
    kge_glstm = _kge(test[OBS_COL], q_glstm)
    print(f"[glofas_lstm] temporal KGE'  raw={kge_raw_t:+.3f} -> "
          f"glofas_lstm={kge_glstm:+.3f}")

    # -- (B) SABER-style donor regionalization on the PUR split -------------
    ptr_m, pte_m = pur_split(df)
    ptrain = df[ptr_m].reset_index(drop=True)
    ptest = df[pte_m].reset_index(drop=True)
    print(f"PUR: train={len(ptrain)} (core) test={len(ptest)} (transfer); "
          f"ungauged test gauges in train: "
          f"{set(ptest['code']).issubset(set(ptrain['code']))}")

    donor = DonorRegionalizationCorrector(k=5)
    donor.fit(ptrain)
    dresid = donor.predict_residual(ptest)
    q_donor = donor.predict(ptest)
    assert len(dresid) == len(ptest), "donor residual not aligned to df rows"
    # PUR target gauges are genuinely ungauged (none seen in training).
    assert not set(ptest["code"]) & set(donor.gauge_factors_), "donor PUR leakage"

    kge_raw_p = _kge(ptest[OBS_COL], ptest[SIM_COL])
    kge_donor = _kge(ptest[OBS_COL], q_donor)
    print(f"[donor]       PUR KGE'       raw={kge_raw_p:+.3f} -> "
          f"donor={kge_donor:+.3f}")

    ok = (np.isfinite(kge_glstm) and np.isfinite(kge_donor)
          and kge_donor >= kge_raw_p - 1e-9)
    print(f"SANITY: both baselines finite; donor not worse than raw on PUR -> {ok}")
    assert ok, "self-test sanity failed"
    print("OK: sota_baselines self-test passed.")
