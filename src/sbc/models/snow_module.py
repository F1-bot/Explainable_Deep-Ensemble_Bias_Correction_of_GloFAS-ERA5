"""Differentiable degree-day snow reservoir and a mass-aware residual corrector.

The flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet` encodes the snow
physics ("corrected discharge is non-decreasing in snowmelt and in available snow
water equivalent") as a *soft* input-gradient penalty.  A soft penalty only
nudges the fit; an adversarial audit of the real decadal run found it violated on
~79 % of melt samples and *net-negative* on KGE'.  This module replaces that
decorative penalty with a **real, mass-conserving mechanism**: a *differentiable*
degree-day snow reservoir whose hydrological parameters (degree-day factor, the
snow/melt temperature thresholds, the refreeze coefficient) are **learned by
gradient descent end-to-end with the bias-correction head** -- the differentiable
parameter-learning ("dPL") lineage of Song, Tsai, Shen and co-workers
(differentiable hydrology; e.g. Feng et al. 2022, *WRR*; Song et al. 2024).

Two public objects are exported.

``DifferentiableSnowReservoir`` (a ``torch.nn.Module``)
    Maps a window of the dynamic forcing -- total precipitation ``tp`` and air
    temperature ``t2m_mean`` -- to a physically-consistent snow-water-equivalent
    (SWE) *state* and a snowmelt *flux*.  At every step

    ``snowfall = tp * sigmoid((T_snow - T)/tau)``        (smooth rain/snow split)
    ``swe      += snowfall``                             (accumulation)
    ``melt      = min(DDF * relu(T - T_melt), swe)``     (degree-day, storage-capped)
    ``swe      -= melt``                                 (mass conservation)

    with an HBV-style liquid store and refreeze term ``min(CFR*DDF*relu(T_melt-T),
    liquid)`` returning water to the pack.  Total water is conserved to machine
    precision (``d(swe+liquid) = tp - outflow``).  Because ``relu`` and ``min``
    are monotone, the per-step melt flux is **non-decreasing in that step's
    temperature (at a given storage) and in available storage, and the SWE state
    is non-decreasing in accumulated snowfall, by construction** -- exactly the
    physics the soft penalty failed to
    guarantee, now enforced with a near-zero violation rate *by design* rather
    than by a tunable weight.

``SnowPhysicsCorrector`` (``BaseCorrector``, registry name ``"diffsnow"``)
    A small corrector whose front-end is the reservoir.  Its learned SWE/melt
    states are concatenated with the **GloFAS-memory** features (the
    ``f_log_qglofas*`` family) and mapped to the log-residual by a *partially
    monotone* head: the two physical snow channels flow through a
    non-negative-weight monotone sub-network (so the corrected discharge is
    provably non-decreasing in melt and in SWE), while the GloFAS-memory channel
    feeds a free MLP.  The whole thing -- reservoir parameters included -- is
    trained end-to-end.  The learned ``DDF`` / thresholds are exposed for
    physical interpretation (:meth:`SnowPhysicsCorrector.snow_parameters`) and a
    melt-monotonicity probe (:meth:`SnowPhysicsCorrector.melt_monotonicity_report`)
    confirms the ~0 % violation rate.

All heavy imports (``torch``) are deferred into methods; matplotlib is not used.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..config import COL_DATE, EPS
from ..schemas import OBS_COL, SIM_COL, TARGET_COL, make_target, validate
from ..utils import get_logger
from .base import BaseCorrector, register

log = get_logger(__name__)

#: default reservoir look-back window (periods) per temporal scale
_DEFAULT_SEQ_LEN: dict[str, int] = {"decadal": 12, "daily": 90}
#: raw forcing columns consumed by the reservoir front-end
SNOW_FORCING: str = "tp"
TEMP_FORCING: str = "t2m_mean"
#: prefix identifying the GloFAS-memory feature family (see features.engineering)
GLOFAS_MEMORY_PREFIX: str = "f_log_qglofas"
#: hard clip on the predicted log-residual (keeps ``exp`` well-behaved)
_RESIDUAL_CLIP: float = 15.0

__all__ = [
    "DifferentiableSnowReservoir",
    "SnowPhysicsCorrector",
    "build_diff_snow_reservoir",
]


# --------------------------------------------------------------------------- #
#  Differentiable degree-day snow reservoir (torch deferred)                   #
# --------------------------------------------------------------------------- #
def build_diff_snow_reservoir() -> "DifferentiableSnowReservoir":
    """Construct a default :class:`DifferentiableSnowReservoir` (torch deferred).

    Returns
    -------
    DifferentiableSnowReservoir
        A reservoir with the default physically-plausible parameter
        initialisation (``DDF`` ~ 4 mm/degC/period, ``T_snow`` ~ 1 degC,
        ``T_melt`` ~ 0 degC).
    """
    return DifferentiableSnowReservoir()


class _ReservoirMeta(type):
    """Metaclass that lazily turns the class into a ``torch.nn.Module`` subclass.

    Importing this module must not import ``torch`` (house style: heavy imports
    inside functions).  ``DifferentiableSnowReservoir`` is therefore declared as a
    plain object here and *rebuilt* as a genuine ``nn.Module`` subclass the first
    time it is instantiated, so callers get a real, trainable module while the
    import path stays light.
    """

    def __call__(cls, *args, **kwargs):  # noqa: D401 - see class docstring
        return _reservoir_class()(*args, **kwargs)


class DifferentiableSnowReservoir(metaclass=_ReservoirMeta):
    """Mass-conserving differentiable degree-day snow reservoir.

    A ``torch.nn.Module`` (materialised on first instantiation) that integrates a
    window of precipitation and temperature into a snow-water-equivalent state and
    a snowmelt flux with **learnable** hydrological parameters.

    Parameters
    ----------
    init_ddf : float, default 4.0
        Initial degree-day melt factor [mm degC-1 period-1]; constrained positive
        via ``softplus`` and learned.
    init_t_snow : float, default 1.0
        Initial rain/snow partition temperature [degC] (precip falls as snow below
        it); learned, unconstrained.
    init_t_melt : float, default 0.0
        Initial melt-onset temperature [degC] (degree-day melt above it); learned.
    init_refreeze : float, default 0.05
        Initial refreeze coefficient (fraction of the degree-day factor applied to
        sub-melt-threshold cold); constrained to ``(0, 1)`` via ``sigmoid``.
    init_liquid_cap : float, default 0.1
        Initial liquid-water holding capacity as a fraction of SWE; ``sigmoid``.
    init_smoothing : float, default 0.75
        Initial temperature smoothing [degC] of the rain/snow ``sigmoid`` split;
        ``softplus`` with a 0.25 floor (keeps the split differentiable, never a
        hard step).

    Notes
    -----
    The forward pass returns the full per-step SWE and melt sequences so the
    caller can read the current-period state (the last step) or inspect the warm-up.
    Monotonicity holds because every operation is monotone in the relevant input:
    ``relu(T - T_melt)`` and the degree-day product are non-decreasing in ``T``;
    ``min(., swe)`` is non-decreasing in both arguments; ``swe += snowfall`` is
    non-decreasing in snowfall.
    """

    # Real signature mirrored on the materialised subclass for documentation /
    # introspection; the metaclass forwards construction to that subclass.
    def __init__(
        self,
        init_ddf: float = 4.0,
        init_t_snow: float = 1.0,
        init_t_melt: float = 0.0,
        init_refreeze: float = 0.05,
        init_liquid_cap: float = 0.1,
        init_smoothing: float = 0.75,
    ) -> None:  # pragma: no cover - replaced by the materialised subclass
        raise RuntimeError("DifferentiableSnowReservoir is materialised lazily; "
                           "instantiate it normally (it returns an nn.Module).")


#: cache for the materialised reservoir ``nn.Module`` class (built once)
_RESERVOIR_CLS: type | None = None


def _reservoir_class() -> type:
    """Build (once) and return the concrete ``nn.Module`` reservoir class.

    Defined inside a function so importing this module never imports ``torch``;
    the class is cached in a module global so the public metaclass shim and the
    internal network share a single, identical implementation.
    """
    global _RESERVOIR_CLS
    if _RESERVOIR_CLS is not None:
        return _RESERVOIR_CLS
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _inv_softplus(y: float) -> float:
        return float(math.log(math.expm1(max(y, 1e-6))))

    def _logit(p: float) -> float:
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return float(math.log(p / (1.0 - p)))

    class _DifferentiableSnowReservoir(nn.Module):
        __doc__ = DifferentiableSnowReservoir.__doc__

        TAU_FLOOR: float = 0.25

        def __init__(
            self,
            init_ddf: float = 4.0,
            init_t_snow: float = 1.0,
            init_t_melt: float = 0.0,
            init_refreeze: float = 0.05,
            init_liquid_cap: float = 0.1,
            init_smoothing: float = 0.75,
        ) -> None:
            super().__init__()
            self.raw_ddf = nn.Parameter(torch.tensor(_inv_softplus(init_ddf)))
            self.t_snow = nn.Parameter(torch.tensor(float(init_t_snow)))
            self.t_melt = nn.Parameter(torch.tensor(float(init_t_melt)))
            self.raw_refreeze = nn.Parameter(torch.tensor(_logit(init_refreeze)))
            self.raw_liquid_cap = nn.Parameter(torch.tensor(_logit(init_liquid_cap)))
            self.raw_tau = nn.Parameter(
                torch.tensor(_inv_softplus(max(init_smoothing - self.TAU_FLOOR, 1e-3)))
            )

        # -- physical-parameter accessors (positive / bounded transforms) ------ #
        @property
        def ddf(self) -> "torch.Tensor":
            """Degree-day melt factor [mm degC-1 period-1] (> 0)."""
            return F.softplus(self.raw_ddf)

        @property
        def refreeze_coef(self) -> "torch.Tensor":
            """Refreeze coefficient in ``(0, 1)``."""
            return torch.sigmoid(self.raw_refreeze)

        @property
        def liquid_cap(self) -> "torch.Tensor":
            """Liquid-water holding capacity as a fraction of SWE, in ``(0, 1)``."""
            return torch.sigmoid(self.raw_liquid_cap)

        @property
        def tau(self) -> "torch.Tensor":
            """Rain/snow partition temperature smoothing [degC] (>= floor)."""
            return F.softplus(self.raw_tau) + self.TAU_FLOOR

        def physical_parameters(self) -> dict[str, float]:
            """Return the learned parameters in interpretable physical units."""
            with torch.no_grad():
                return {
                    "ddf_mm_per_degC_per_period": float(self.ddf),
                    "t_snow_degC": float(self.t_snow),
                    "t_melt_degC": float(self.t_melt),
                    "refreeze_coef": float(self.refreeze_coef),
                    "liquid_holding_frac": float(self.liquid_cap),
                    "temp_smoothing_degC": float(self.tau),
                }

        # -- the differentiable water balance --------------------------------- #
        def forward(self, tp: "torch.Tensor", temp: "torch.Tensor"
                    ) -> tuple["torch.Tensor", "torch.Tensor"]:
            """Integrate a forcing window into SWE and melt sequences.

            Parameters
            ----------
            tp : torch.Tensor
                Precipitation window [mm], shape ``(batch, seq_len)``.
            temp : torch.Tensor
                Air-temperature window [degC], shape ``(batch, seq_len)``.

            Returns
            -------
            swe_seq, melt_seq : torch.Tensor
                Per-step snow-water-equivalent state and snowmelt flux, each of
                shape ``(batch, seq_len)``; both are non-negative.
            """
            batch, length = tp.shape
            ddf = self.ddf
            t_snow = self.t_snow
            t_melt = self.t_melt
            cfr = self.refreeze_coef
            cwh = self.liquid_cap
            tau = self.tau

            swe = tp.new_zeros(batch)
            liquid = tp.new_zeros(batch)
            swe_seq: list["torch.Tensor"] = []
            melt_seq: list["torch.Tensor"] = []
            for t in range(length):
                temp_t = temp[:, t]
                precip = torch.clamp(tp[:, t], min=0.0)

                # smooth rain/snow partition (sigmoid -> differentiable threshold)
                snow_frac = torch.sigmoid((t_snow - temp_t) / tau)
                snowfall = precip * snow_frac
                rainfall = precip - snowfall

                # accumulation
                swe = swe + snowfall

                # degree-day melt, capped by available storage (mass-conserving)
                melt_pot = ddf * F.relu(temp_t - t_melt)
                melt = torch.minimum(melt_pot, swe)
                swe = swe - melt

                # refreeze: cold returns retained liquid water to the pack
                refreeze_pot = cfr * ddf * F.relu(t_melt - temp_t)
                refreeze = torch.minimum(refreeze_pot, liquid)
                liquid = liquid - refreeze
                swe = swe + refreeze

                # liquid routing: melt + rain enter the liquid store, excess over
                # the holding capacity leaves as snowpack outflow
                liquid = liquid + melt + rainfall
                outflow = F.relu(liquid - cwh * swe)
                liquid = liquid - outflow

                swe_seq.append(swe)
                melt_seq.append(melt)

            return torch.stack(swe_seq, dim=1), torch.stack(melt_seq, dim=1)

    _DifferentiableSnowReservoir.__name__ = "DifferentiableSnowReservoir"
    _DifferentiableSnowReservoir.__qualname__ = "DifferentiableSnowReservoir"
    _RESERVOIR_CLS = _DifferentiableSnowReservoir
    return _RESERVOIR_CLS


# --------------------------------------------------------------------------- #
#  Full network: reservoir front-end + partially-monotone residual head        #
# --------------------------------------------------------------------------- #
def _build_net(d_mem: int, hidden: int, mono_hidden: int, seed: int):
    """Build the reservoir + partially-monotone head ``nn.Module`` (torch here).

    The head reads ``[log1p(swe), log1p(melt)]`` through a non-negative-weight,
    ReLU monotone branch (so the residual is provably non-decreasing in both the
    snowmelt flux and the SWE state) and the standardised GloFAS-memory vector
    through a free MLP branch; the two branches are summed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    reservoir_cls = _reservoir_class()

    class _DiffSnowNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reservoir = reservoir_cls()
            g = torch.Generator().manual_seed(int(seed))
            # partially-monotone snow head: softplus(weights) >= 0 -> monotone
            self.mono_w1 = nn.Parameter(torch.randn(2, mono_hidden, generator=g) * 0.1)
            self.mono_b1 = nn.Parameter(torch.zeros(mono_hidden))
            self.mono_w2 = nn.Parameter(torch.randn(mono_hidden, 1, generator=g) * 0.1)
            self.mono_b2 = nn.Parameter(torch.zeros(1))
            # free GloFAS-memory branch
            if d_mem > 0:
                self.free = nn.Sequential(
                    nn.Linear(d_mem, hidden), nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
            else:
                self.free = None
                self.free_bias = nn.Parameter(torch.zeros(1))

        def reservoir_states(self, tp: "torch.Tensor", temp: "torch.Tensor"
                             ) -> tuple["torch.Tensor", "torch.Tensor"]:
            """Current-period (last-step) SWE state and melt flux, shape ``(B,)``."""
            swe_seq, melt_seq = self.reservoir(tp, temp)
            return swe_seq[:, -1], melt_seq[:, -1]

        def head(self, swe: "torch.Tensor", melt: "torch.Tensor",
                 mem: "torch.Tensor") -> "torch.Tensor":
            """Map physical snow states + GloFAS memory to the standardised residual."""
            s2 = torch.stack([torch.log1p(swe), torch.log1p(melt)], dim=1)  # (B, 2)
            w1 = F.softplus(self.mono_w1)            # >= 0 -> non-decreasing in s2
            w2 = F.softplus(self.mono_w2)            # >= 0 -> non-decreasing in hidden
            hmono = F.relu(s2 @ w1 + self.mono_b1)   # ReLU is non-decreasing
            mono_out = (hmono @ w2 + self.mono_b2).squeeze(-1)
            if self.free is not None:
                free_out = self.free(mem).squeeze(-1)
            else:
                free_out = self.free_bias.expand(swe.shape[0])
            return mono_out + free_out

        def forward(self, tp: "torch.Tensor", temp: "torch.Tensor",
                    mem: "torch.Tensor") -> "torch.Tensor":
            swe, melt = self.reservoir_states(tp, temp)
            return self.head(swe, melt, mem)

    return _DiffSnowNet()


# --------------------------------------------------------------------------- #
#  The corrector                                                              #
# --------------------------------------------------------------------------- #
@register
class SnowPhysicsCorrector(BaseCorrector):
    """Mass-aware bias corrector with a differentiable snow-reservoir front-end.

    The corrector replaces the flagship's *soft* SWE/temperature monotonicity
    penalty with a *structural* one: a :class:`DifferentiableSnowReservoir`
    converts the ``(tp, t2m_mean)`` window into a physically-consistent SWE state
    and melt flux, which a partially-monotone head combines with the GloFAS-memory
    features to predict the log-residual.  Because the snow channels pass through
    a non-negative-weight head, the corrected discharge is **provably
    non-decreasing in the snowmelt flux and in SWE** (~0 % melt-monotonicity
    violations), and because the reservoir -- not the raw, noisy ERA5 ``smlt`` /
    ``swe`` columns -- supplies those states, the corrector is by construction
    invariant to (hence non-violating in) the raw forcings the soft flagship
    mishandles.

    Parameters
    ----------
    seq_len : int, optional
        Reservoir look-back window (periods).  ``None`` selects 12 (decadal) or 90
        (daily) from the training table's ``scale`` column.
    hidden : int, default 32
        Width of the free GloFAS-memory MLP branch.
    mono_hidden : int, default 16
        Width of the monotone snow-head hidden layer.
    epochs, batch_size, lr, weight_decay, patience : training schedule.
    huber_beta : float, default 1.0
        Transition point of the Huber (smooth-L1) loss on the standardised
        residual; robust to the heavy-tailed log-residual outliers of small flows.
    valid_fraction : float, default 0.2
        Tail fraction (by date) held out from ``train`` for early stopping when no
        explicit validation table is supplied.
    snow_col, temp_col : str
        Forcing columns feeding the reservoir (default ``"tp"`` / ``"t2m_mean"``).
    memory_prefix : str, default ``"f_log_qglofas"``
        Prefix selecting the GloFAS-memory feature family; if none are present a
        single ``log(q_glofas + EPS)`` channel is synthesised.
    residual_clip : float, default 15.0
        Outer (absolute) safety clip on the predicted log-residual.
    band_k : float, default 4.0
        Width, in robust MADs about the training-target median, of the
        data-driven band the predicted log-residual is clipped to.  The
        multiplicative log-residual is heavy-tailed (a single over-corrected
        high-flow row otherwise wrecks KGE'); clipping to ``median +/- band_k*MAD``
        of the training residual bounds the correction factor to the range the
        data actually support (the same robustification the hard-monotone variant
        applies to its head).
    use_gpu : bool or None, default None
        ``None`` auto-selects CUDA when available.
    seed : int, default 1234
        Reproducibility seed.
    verbose : bool, default True
        Emit per-decile training logs.

    Attributes
    ----------
    net_ : torch.nn.Module
        The fitted reservoir + head network.
    mem_cols_ : list of str
        GloFAS-memory columns actually used (``["__log_qglofas__"]`` when synthesised).
    """

    name = "diffsnow"
    is_probabilistic = False

    def __init__(
        self,
        seq_len: int | None = None,
        hidden: int = 32,
        mono_hidden: int = 16,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 5e-3,
        weight_decay: float = 1e-5,
        patience: int = 12,
        huber_beta: float = 1.0,
        valid_fraction: float = 0.2,
        snow_col: str = SNOW_FORCING,
        temp_col: str = TEMP_FORCING,
        memory_prefix: str = GLOFAS_MEMORY_PREFIX,
        residual_clip: float = _RESIDUAL_CLIP,
        band_k: float = 4.0,
        use_gpu: bool | None = None,
        seed: int = 1234,
        verbose: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.hidden = int(hidden)
        self.mono_hidden = int(mono_hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.huber_beta = float(huber_beta)
        self.valid_fraction = float(valid_fraction)
        self.snow_col = str(snow_col)
        self.temp_col = str(temp_col)
        self.memory_prefix = str(memory_prefix)
        self.residual_clip = float(residual_clip)
        self.band_k = float(band_k)
        self.use_gpu = use_gpu
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # learned / fitted state
        self.seq_len_: int = 0
        self.mem_cols_: list[str] = []
        self._synth_mem: bool = False
        self.mem_mean_: np.ndarray | None = None
        self.mem_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.resid_lo_: float = -self.residual_clip
        self.resid_hi_: float = self.residual_clip
        self.net_ = None
        self.device = "cpu"
        self.history_: dict[str, list[float]] = {"train": [], "valid": []}
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  Column / window plumbing                                          #
    # ------------------------------------------------------------------ #
    def _resolve_seq_len(self, df: pd.DataFrame) -> int:
        if self.seq_len is not None:
            return int(self.seq_len)
        scale = str(df["scale"].iloc[0]) if "scale" in df.columns and len(df) else "decadal"
        return _DEFAULT_SEQ_LEN.get(scale, 12)

    def _find_mem_cols(self, df: pd.DataFrame) -> tuple[list[str], bool]:
        cols = [c for c in df.columns if c.startswith(self.memory_prefix)]
        if cols:
            return cols, False
        return ["__log_qglofas__"], True

    def _mem_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Raw (unstandardised) GloFAS-memory matrix, shape ``(n, d_mem)``."""
        if self._synth_mem:
            q = df[SIM_COL].to_numpy(float)
            return np.log(np.clip(q, 0.0, None) + EPS)[:, None]
        cols = []
        for c in self.mem_cols_:
            x = df[c].to_numpy(float) if c in df.columns else np.full(len(df), np.nan)
            cols.append(x)
        return np.column_stack(cols) if cols else np.zeros((len(df), 0), float)

    def _mem_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised GloFAS-memory matrix (NaN -> per-column mean == 0)."""
        raw = self._mem_raw(df)
        if raw.shape[1] == 0:
            return raw.astype(np.float32)
        z = (raw - self.mem_mean_) / self.mem_std_
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def _window_index(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row ``(seq_len,)`` window of preceding positional indices (causal)."""
        n = len(df)
        dates = df[COL_DATE].to_numpy()
        length = self.seq_len_
        win = np.empty((n, length), dtype=np.int64)
        for _, pos in df.groupby("code", sort=False).indices.items():
            pos = np.asarray(pos)
            pos = pos[np.argsort(dates[pos], kind="stable")]
            for t in range(pos.shape[0]):
                idx = pos[max(0, t - length + 1): t + 1]
                if idx.shape[0] < length:  # left-pad by repeating the earliest row
                    idx = np.concatenate([np.full(length - idx.shape[0], idx[0]), idx])
                win[pos[t]] = idx
        return win

    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(tp_window, temp_window, mem)`` aligned to ``df`` row order."""
        dfx = df.reset_index(drop=True)
        n = len(dfx)
        tp = (dfx[self.snow_col].to_numpy(float) if self.snow_col in dfx.columns
              else np.zeros(n))
        temp = (dfx[self.temp_col].to_numpy(float) if self.temp_col in dfx.columns
                else np.zeros(n))
        tp = np.where(np.isfinite(tp), tp, 0.0)
        temp = np.where(np.isfinite(temp), temp, 0.0)
        win = self._window_index(dfx)
        x_tp = tp[win].astype(np.float32)
        x_temp = temp[win].astype(np.float32)
        mem = self._mem_matrix(dfx)
        return x_tp, x_temp, mem

    def _time_split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.valid_fraction <= 0 or len(train) < 10:
            return train, train.iloc[0:0]
        cutoff = train[COL_DATE].quantile(1.0 - self.valid_fraction)
        tr = train[train[COL_DATE] <= cutoff]
        va = train[train[COL_DATE] > cutoff]
        if len(va) < 1 or len(tr) < 1:
            return train, train.iloc[0:0]
        return tr, va

    # ------------------------------------------------------------------ #
    #  Fit                                                               #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "SnowPhysicsCorrector":
        """Fit the reservoir parameters and residual head jointly, end-to-end."""
        import torch
        import torch.nn.functional as F

        train = validate(train).reset_index(drop=True)
        if self.snow_col not in train.columns or self.temp_col not in train.columns:
            raise ValueError(
                f"SnowPhysicsCorrector requires forcing columns "
                f"'{self.snow_col}' and '{self.temp_col}'"
            )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.use_gpu = torch.cuda.is_available() if self.use_gpu is None else self.use_gpu
        self.device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

        self.seq_len_ = self._resolve_seq_len(train)
        self.mem_cols_, self._synth_mem = self._find_mem_cols(train)

        if valid is not None and len(valid):
            tr_df, va_df = train, validate(valid).reset_index(drop=True)
        else:
            tr_df, va_df = self._time_split(train)
            tr_df = tr_df.reset_index(drop=True)
            va_df = va_df.reset_index(drop=True)

        # standardisation statistics (training split only)
        mem_raw = self._mem_raw(tr_df)
        if mem_raw.shape[1]:
            self.mem_mean_ = np.nanmean(mem_raw, axis=0)
            self.mem_std_ = np.clip(np.nanstd(mem_raw, axis=0), 1e-6, None)
        else:
            self.mem_mean_ = np.zeros(0)
            self.mem_std_ = np.ones(0)
        y_tr = tr_df[TARGET_COL].to_numpy(float)
        yf = y_tr[np.isfinite(y_tr)]
        self.y_mean_ = float(np.mean(yf)) if yf.size else 0.0
        self.y_std_ = float(np.clip(np.std(yf), 1e-6, None)) if yf.size else 1.0
        # robust data-driven band for the (heavy-tailed) multiplicative residual
        if yf.size:
            med = float(np.median(yf))
            mad = 1.4826 * float(np.median(np.abs(yf - med))) + 1e-3
            self.resid_lo_ = max(med - self.band_k * mad, -self.residual_clip)
            self.resid_hi_ = min(med + self.band_k * mad, self.residual_clip)
        else:
            self.resid_lo_, self.resid_hi_ = -self.residual_clip, self.residual_clip

        d_mem = len(self.mem_cols_)
        self.net_ = _build_net(d_mem, self.hidden, self.mono_hidden, self.seed).to(self.device)

        def _prep(df):
            x_tp, x_temp, mem = self._design(df)
            y = (df[TARGET_COL].to_numpy(float) - self.y_mean_) / self.y_std_
            y = np.where(np.isfinite(y), y, 0.0)
            return (
                torch.as_tensor(x_tp, dtype=torch.float32, device=self.device),
                torch.as_tensor(x_temp, dtype=torch.float32, device=self.device),
                torch.as_tensor(mem, dtype=torch.float32, device=self.device),
                torch.as_tensor(y, dtype=torch.float32, device=self.device),
            )

        Xtp, Xt, Xm, yt = _prep(tr_df)
        n = Xtp.shape[0]
        has_valid = len(va_df) > 0
        if has_valid:
            Vtp, Vt, Vm, vy = _prep(va_df)

        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)

        best_state = copy.deepcopy(self.net_.state_dict())
        best_metric = np.inf
        bad = 0
        bs = max(1, min(self.batch_size, n))

        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen)
            ep_losses = []
            for s in range(0, n, bs):
                bi = perm[s: s + bs].to(self.device)
                pred = self.net_(Xtp[bi], Xt[bi], Xm[bi])
                loss = F.smooth_l1_loss(pred, yt[bi], beta=self.huber_beta)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                opt.step()
                ep_losses.append(float(loss.detach()))

            train_metric = float(np.mean(ep_losses)) if ep_losses else np.nan
            if has_valid:
                self.net_.eval()
                with torch.no_grad():
                    vpred = self.net_(Vtp, Vt, Vm)
                    valid_metric = float(F.smooth_l1_loss(vpred, vy, beta=self.huber_beta))
            else:
                valid_metric = train_metric
            self.history_["train"].append(train_metric)
            self.history_["valid"].append(valid_metric)

            if valid_metric < best_metric - 1e-6:
                best_metric, bad = valid_metric, 0
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                bad += 1
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0
                                 or epoch == self.epochs - 1):
                p = self.net_.reservoir.physical_parameters()
                log.info("epoch %3d | train=%.4f | valid=%.4f | DDF=%.2f T_snow=%.2f "
                         "T_melt=%.2f", epoch, train_metric, valid_metric,
                         p["ddf_mm_per_degC_per_period"], p["t_snow_degC"],
                         p["t_melt_degC"])
            if bad >= self.patience:
                if self.verbose:
                    log.info("early stop at epoch %d (best valid=%.4f)", epoch, best_metric)
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self._fitted = True
        if self.verbose:
            p = self.snow_parameters()
            log.info("diffsnow fitted: L=%d d_mem=%d | learned DDF=%.2f mm/degC, "
                     "T_snow=%.2f, T_melt=%.2f, refreeze=%.3f",
                     self.seq_len_, d_mem, p["ddf_mm_per_degC_per_period"],
                     p["t_snow_degC"], p["t_melt_degC"], p["refreeze_coef"])
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                         #
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted or self.net_ is None:
            raise RuntimeError("SnowPhysicsCorrector is not fitted; call fit() first")

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-space residual aligned 1:1 to ``df`` rows."""
        self._check_fitted()
        import torch

        x_tp, x_temp, mem = self._design(df)
        n = x_tp.shape[0]
        out = np.zeros(n, float)
        bs = max(self.batch_size, 1024)
        self.net_.eval()
        with torch.no_grad():
            for s in range(0, n, bs):
                tp = torch.as_tensor(x_tp[s: s + bs], dtype=torch.float32, device=self.device)
                tt = torch.as_tensor(x_temp[s: s + bs], dtype=torch.float32, device=self.device)
                mm = torch.as_tensor(mem[s: s + bs], dtype=torch.float32, device=self.device)
                out[s: s + bs] = self.net_(tp, tt, mm).cpu().numpy()
        resid = out * self.y_std_ + self.y_mean_
        # clip to the robust training band (monotone -> preserves the snow
        # monotonicity guarantee) then to the outer absolute safety clip
        return np.clip(resid, self.resid_lo_, self.resid_hi_)

    # ------------------------------------------------------------------ #
    #  Physical interpretability & monotonicity                          #
    # ------------------------------------------------------------------ #
    def snow_parameters(self) -> dict[str, float]:
        """Learned reservoir parameters in physical units (interpretability)."""
        self._check_fitted()
        return self.net_.reservoir.physical_parameters()

    def reservoir_states(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Current-period physical SWE state and melt flux per row.

        Returns
        -------
        dict
            ``"swe"`` and ``"melt"`` arrays (length ``len(df)``), the
            mass-conserving snow-water-equivalent storage and snowmelt flux the
            reservoir derives from ``(tp, t2m_mean)``.
        """
        self._check_fitted()
        import torch

        x_tp, x_temp, _ = self._design(df)
        n = x_tp.shape[0]
        swe = np.zeros(n, float)
        melt = np.zeros(n, float)
        bs = max(self.batch_size, 1024)
        self.net_.eval()
        with torch.no_grad():
            for s in range(0, n, bs):
                tp = torch.as_tensor(x_tp[s: s + bs], dtype=torch.float32, device=self.device)
                tt = torch.as_tensor(x_temp[s: s + bs], dtype=torch.float32, device=self.device)
                sw, ml = self.net_.reservoir_states(tp, tt)
                swe[s: s + bs] = sw.cpu().numpy()
                melt[s: s + bs] = ml.cpu().numpy()
        return {"swe": swe, "melt": melt}

    def melt_monotonicity_report(
        self, df: pd.DataFrame, *, rel_delta: float = 0.25, tol: float = 1e-6
    ) -> dict[str, float]:
        """Empirical melt/SWE-monotonicity violation rate of the corrected residual.

        The reservoir's physical melt flux (and SWE state) is bumped *upward* by a
        small positive step and the predicted log-residual is recomputed from the
        head, holding the GloFAS-memory channel fixed.  A *violation* is a row
        whose residual **decreases** under the increase, contradicting the
        physical requirement that corrected discharge be non-decreasing in
        snowmelt / SWE.  Because the snow channels flow through the non-negative
        weight monotone head, the rate is ``~0`` by construction.

        Parameters
        ----------
        df : pandas.DataFrame
            Evaluation table.
        rel_delta : float, default 0.25
            Perturbation as a fraction of each channel's (finite) standard
            deviation.
        tol : float, default 1e-6
            Tolerance below which a decrease is not counted.

        Returns
        -------
        dict
            ``viol_melt``, ``viol_swe`` (per-channel rates), ``viol_rate``
            (violation in *either* channel), the perturbation sizes and ``n``.
        """
        self._check_fitted()
        import torch

        x_tp, x_temp, mem = self._design(df)
        n = x_tp.shape[0]
        out: dict[str, float] = {"n": int(n)}
        if n == 0:
            out["viol_rate"] = 0.0
            return out

        self.net_.eval()
        with torch.no_grad():
            tp = torch.as_tensor(x_tp, dtype=torch.float32, device=self.device)
            tt = torch.as_tensor(x_temp, dtype=torch.float32, device=self.device)
            mm = torch.as_tensor(mem, dtype=torch.float32, device=self.device)
            swe, melt = self.net_.reservoir_states(tp, tt)
            base = self.net_.head(swe, melt, mm).cpu().numpy()

            swe_np = swe.cpu().numpy()
            melt_np = melt.cpu().numpy()
            d_melt = max(rel_delta * float(np.std(melt_np)), 1e-6)
            d_swe = max(rel_delta * float(np.std(swe_np)), 1e-6)

            bumped_melt = self.net_.head(
                swe, melt + d_melt, mm).cpu().numpy()
            bumped_swe = self.net_.head(
                swe + d_swe, melt, mm).cpu().numpy()

        v_melt = bumped_melt < base - tol
        v_swe = bumped_swe < base - tol
        out["viol_melt"] = float(np.mean(v_melt))
        out["viol_swe"] = float(np.mean(v_swe))
        out["viol_rate"] = float(np.mean(v_melt | v_swe))
        out["delta_melt"] = float(d_melt)
        out["delta_swe"] = float(d_swe)
        return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..synthetic import generate
    from ..validation.metrics import kge_prime
    from ..validation.splits import temporal_split

    # --- small synthetic table -> features -> regimes ----------------------- #
    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(df, scale="decadal"))
    df = validate(df)
    df[TARGET_COL] = make_target(df[OBS_COL].values, df[SIM_COL].values)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    print(f"[snow_module] gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    # --- fit a tiny diffsnow ------------------------------------------------ #
    model = SnowPhysicsCorrector(
        seq_len=10, hidden=24, mono_hidden=8, epochs=60, batch_size=256,
        patience=20, lr=3e-3, seed=0, verbose=False).fit(train, test)

    q_corr = model.predict(test)
    kge_raw = kge_prime(test[OBS_COL].values, test[SIM_COL].values)["kge"]
    kge_cor = kge_prime(test[OBS_COL].values, q_corr)["kge"]
    resid = model.predict_residual(test)
    states = model.reservoir_states(test)
    params = model.snow_parameters()

    print(f"[snow_module] KGE' raw={kge_raw:+.3f} -> corrected={kge_cor:+.3f} | "
          f"resid_aligned={len(resid) == len(test)}")
    print(f"[snow_module] learned snow params: DDF={params['ddf_mm_per_degC_per_period']:.2f} "
          f"mm/degC/period | T_snow={params['t_snow_degC']:+.2f} degC | "
          f"T_melt={params['t_melt_degC']:+.2f} degC | refreeze={params['refreeze_coef']:.3f} | "
          f"liquid_cap={params['liquid_holding_frac']:.3f}")
    print(f"[snow_module] reservoir states: SWE mean={np.nanmean(states['swe']):.1f} mm, "
          f"melt mean={np.nanmean(states['melt']):.2f} mm, melt>=0={bool(np.all(states['melt'] >= -1e-6))}")

    # --- melt-monotonicity: diffsnow (~0%) vs the SOFT flagship ------------- #
    diff_mono = model.melt_monotonicity_report(test)
    print(f"[snow_module] diffsnow melt-monotonicity viol_rate={diff_mono['viol_rate']:.4f} "
          f"(melt={diff_mono['viol_melt']:.4f}, swe={diff_mono['viol_swe']:.4f}) -- ~0 by design")

    soft_viol = float("nan")
    try:
        from .constraint_variants import monotonicity_violation_rate
        from .regime_prob_net import RegimeProbNet

        soft = RegimeProbNet(
            K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
            epochs=3, batch_size=512, patience=3, lambda_gate=0.5, lambda_phys=0.1,
            physics=True, seed=0, verbose=False).fit(train, test)
        v_soft = monotonicity_violation_rate(soft, test)          # raw smlt / swe
        v_diff = monotonicity_violation_rate(model, test)         # invariant -> ~0
        soft_viol = v_soft["viol_rate"]
        print(f"[snow_module] SOFT flagship (smlt/swe) viol_rate={v_soft['viol_rate']:.4f} "
              f"-> diffsnow (smlt/swe) viol_rate={v_diff['viol_rate']:.4f}")
    except Exception as exc:  # pragma: no cover - flagship optional in self-test
        print(f"[snow_module] (soft-flagship comparison skipped: {exc})")

    # --- assertions --------------------------------------------------------- #
    assert resid.shape == (len(test),), "residual not aligned to rows"
    assert np.all(np.isfinite(resid)), "non-finite residual"
    assert np.all(states["melt"] >= -1e-6) and np.all(states["swe"] >= -1e-6), \
        "reservoir produced negative SWE/melt (mass-conservation broken)"
    assert diff_mono["viol_rate"] < 1e-3, "melt monotonicity not enforced (~0 expected)"
    assert params["ddf_mm_per_degC_per_period"] > 0, "DDF must be positive"
    print("[snow_module] SELF-TEST OK")
