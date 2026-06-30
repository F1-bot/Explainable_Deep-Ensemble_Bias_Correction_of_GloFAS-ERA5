"""Reviewer-demanded constraint and likelihood variants of the flagship.

The flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet` makes two
methodological choices that a careful reviewer will immediately ask to see
isolated against the obvious alternative:

1. **Soft vs hard monotonicity.**  RegimeProbNet encodes the physics that the
   corrected discharge must be non-decreasing in snowmelt (``smlt``) and snow
   water equivalent (``swe``) as a *soft* input-gradient penalty
   ``relu(-d mu / d x)`` on melt-regime samples (``lambda_phys``).  A soft
   penalty nudges, but never guarantees, monotonicity: a finite-difference probe
   of the fitted network still finds rows where ``mu`` *decreases* when ``swe``
   or ``smlt`` is increased.  :class:`HardMonoProbNetCorrector` (registry name
   ``"probnet_hardmono"``) supplies the *hard*-constraint counterpart: it wraps a
   trained flagship in a partially-monotone reparametrisation whose dependence on
   the snow forcings flows through a **non-negative-weight monotone head**, so the
   predicted residual is provably non-decreasing in ``swe`` / ``smlt`` and its
   empirical monotonicity-violation rate is ``~0``.  The two models share an API,
   so the SOFT-vs-HARD contrast reduces to one table row.

2. **Gaussian vs heavy-tailed mixture likelihood.**  RegimeProbNet models the
   log-residual with a *Gaussian* mixture-of-experts.  Snowmelt-freshet and
   rain-on-snow residuals are right-skewed and heavy-tailed, which a Gaussian
   under-weights in the tails.  :class:`AsymLaplaceMixtureCorrector` (registry
   name ``"probnet_alaplace"``) mirrors the flagship's regime-gated EA-LSTM
   architecture but replaces each Gaussian expert with a two-piece
   (asymmetric-Laplace) exponential density, trained by mixture negative
   log-likelihood.  Exponential tails capture freshet peaks the Gaussian clips,
   often sharpening CRPS on the melt regime.

Both are ordinary :class:`~sbc.models.base.BaseCorrector` subclasses exposing the
full probabilistic API (:meth:`predict_quantiles`, :meth:`predict_variance`,
:meth:`sample`), so they drop straight into the existing CRPS / calibration /
ablation machinery.  The module also exports :func:`monotonicity_violation_rate`,
a model-agnostic finite-difference probe used to quantify the SOFT-vs-HARD gap.

All heavy imports (``torch``) are deferred into methods; matplotlib is not used.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..schemas import TARGET_COL, validate
from ..utils import get_logger
from .base import BaseCorrector, register
from .regime_prob_net import (
    MELT_REGIMES,
    REGIME_TO_IDX,
    RegimeProbNet,
    _classify_regimes,
)

log = get_logger(__name__)

#: raw forcing columns the monotonicity constraint applies to (``d mu / d x>=0``);
#: mirrors the flagship's soft penalty, which acts on the same ``smlt`` / ``swe``.
SNOW_CONSTRAINED: tuple[str, ...] = ("smlt", "swe")

__all__ = [
    "SNOW_CONSTRAINED",
    "monotonicity_violation_rate",
    "HardMonoProbNetCorrector",
    "AsymLaplaceMixtureCorrector",
]


# --------------------------------------------------------------------------- #
#  Model-agnostic monotonicity probe                                          #
# --------------------------------------------------------------------------- #
def monotonicity_violation_rate(
    model: BaseCorrector,
    df: pd.DataFrame,
    features: tuple[str, ...] = SNOW_CONSTRAINED,
    *,
    rel_delta: float = 0.25,
    abs_delta: float | None = None,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Empirical rate at which a corrector violates snow monotonicity.

    For every constrained feature the column is perturbed *upward* by a small
    positive step and the predicted log-residual is recomputed.  A row is a
    *violation* when the residual **decreases** under that increase (i.e.
    ``mu(x + delta) < mu(x) - tol``), contradicting the physical requirement
    ``d mu / d x >= 0``.  The probe only calls :meth:`predict_residual`, so it
    works for any :class:`~sbc.models.base.BaseCorrector` -- the soft flagship,
    the hard variant, the boosters, etc.

    Parameters
    ----------
    model : BaseCorrector
        A *fitted* corrector.
    df : pandas.DataFrame
        Evaluation table (a modelling table; only the feature columns are read).
    features : tuple of str, default :data:`SNOW_CONSTRAINED`
        Columns over which monotonicity is required; absent columns are skipped.
    rel_delta : float, default 0.25
        Perturbation size as a fraction of each feature's (finite) standard
        deviation; used when ``abs_delta`` is ``None``.
    abs_delta : float, optional
        Absolute perturbation in the feature's own units (overrides ``rel_delta``).
    tol : float, default 1e-6
        Numerical tolerance below which a decrease is not counted as a violation.

    Returns
    -------
    dict
        ``viol_rate`` (fraction of rows violating monotonicity in *any*
        constrained feature), per-feature ``viol_{name}`` rates, the perturbation
        sizes ``delta_{name}``, and the row count ``n``.
    """
    present = [f for f in features if f in df.columns]
    n = int(len(df))
    out: dict[str, float] = {"n": n}
    if n == 0 or not present:
        out["viol_rate"] = 0.0
        return out

    base = np.asarray(model.predict_residual(df), float).ravel()
    viol_any = np.zeros(n, dtype=bool)
    for f in present:
        x = df[f].to_numpy(float)
        if abs_delta is not None:
            delta = float(abs_delta)
        else:
            xf = x[np.isfinite(x)]
            scale = float(np.std(xf)) if xf.size else 1.0
            delta = max(rel_delta * scale, 1e-6)
        bumped_df = df.copy()
        bumped_df[f] = x + delta
        bumped = np.asarray(model.predict_residual(bumped_df), float).ravel()
        ok = np.isfinite(base) & np.isfinite(bumped)
        viol = ok & (bumped < base - tol)
        out[f"viol_{f}"] = float(viol.sum()) / float(max(ok.sum(), 1))
        out[f"delta_{f}"] = delta
        viol_any |= viol
    out["viol_rate"] = float(np.mean(viol_any))
    return out


# --------------------------------------------------------------------------- #
#  (1) Hard-monotonic wrapper around the flagship                             #
# --------------------------------------------------------------------------- #
def _build_mono_head(d_in: int, hidden: int):
    """Construct a small monotone-increasing MLP head (torch deferred).

    The head is non-decreasing in **every** input coordinate by construction:
    both weight matrices are passed through ``softplus`` (so they are
    non-negative) and the hidden activation (ReLU) is monotone non-decreasing;
    the biases are free and do not affect monotonicity.  Composed with an
    affine, positively-scaled input standardisation, the network is a guaranteed
    monotone function of the *raw* snow features.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _MonoHead(nn.Module):
        def __init__(self, d_in: int, hidden: int) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.raw_w1 = nn.Parameter(torch.randn(d_in, hidden, generator=g) * 0.1)
            self.b1 = nn.Parameter(torch.zeros(hidden))
            self.raw_w2 = nn.Parameter(torch.randn(hidden, 1, generator=g) * 0.1)
            self.b2 = nn.Parameter(torch.zeros(1))

        def forward(self, s: "torch.Tensor") -> "torch.Tensor":
            w1 = F.softplus(self.raw_w1)            # >= 0  -> monotone in s
            w2 = F.softplus(self.raw_w2)            # >= 0  -> monotone in hidden
            h = F.relu(s @ w1 + self.b1)            # ReLU is non-decreasing
            return (h @ w2 + self.b2).squeeze(-1)

    return _MonoHead(int(d_in), int(hidden))


@register
class HardMonoProbNetCorrector(BaseCorrector):
    """Hard-monotone (partially-monotone) wrapper of :class:`RegimeProbNet`.

    The predicted log-residual is reparametrised as

    ``mu_hard(R, S) = mu_base(R)  +  h(S)``

    where ``S = (smlt, swe)`` are the monotone-constrained *raw* snow forcings and
    ``R`` is everything else.  The base flagship ``mu_base`` is trained on the
    table with the raw ``smlt`` / ``swe`` columns **removed** (so it never depends
    on them -- engineered snow-*memory* features such as ``f_swe_*`` are kept,
    exactly as the flagship's soft penalty also leaves them unconstrained), and
    ``h`` is a non-negative-weight monotone head (:func:`_build_mono_head`).
    Because the only path from the raw ``S`` to the output is the monotone head,
    ``d mu_hard / d smlt`` and ``d mu_hard / d swe`` are provably ``>= 0`` and the
    empirical violation rate is ``~0`` -- the *hard* counterpart to the flagship's
    *soft* gradient penalty on the same two forcings.  Dropping the columns
    (rather than freezing them at a baseline) keeps the backbone in-distribution,
    so the hard constraint costs little accuracy.

    The full predictive distribution is the base distribution *location-shifted*
    by ``h(S)``; the shift preserves the shape (hence the variance) and the
    monotone ordering of every quantile.

    Parameters
    ----------
    base : RegimeProbNet, optional
        An (unfitted) flagship *template* to clone the configuration from; it is
        always (re)fitted internally on the snow-dropped table so the
        monotonicity guarantee holds.  When ``None`` a fresh flagship is built
        from ``base_kwargs`` (with the *soft* physics penalty switched off, since
        the raw snow dependence is supplied by the hard head).
    constrained_features : tuple of str, default :data:`SNOW_CONSTRAINED`
        Raw columns the hard monotonicity constraint applies to.
    head_hidden : int, default 16
        Width of the monotone head's hidden layer.
    head_epochs : int, default 250
        Full-batch Adam epochs for the head.
    head_lr : float, default 5e-2
        Head learning rate.
    head_weight_decay : float, default 0.0
        L2 regularisation on the head.
    residual_clip : float, default 15.0
        Symmetric clip on the predicted log-residual (keeps ``exp`` well-behaved).
    seed : int, default 1234
        Reproducibility seed (also the default base seed).
    verbose : bool, default True
        Forwarded to a freshly built base flagship.
    **base_kwargs :
        Extra :class:`RegimeProbNet` keywords used only when a base is built
        internally (e.g. ``epochs``, ``hidden``, ``seq_len``).

    Attributes
    ----------
    base_ : RegimeProbNet
        The fitted base flagship.
    constrained_ : list of str
        Constrained columns actually present in the training table.
    """

    name = "probnet_hardmono"
    is_probabilistic = True

    def __init__(
        self,
        base: RegimeProbNet | None = None,
        constrained_features: tuple[str, ...] = SNOW_CONSTRAINED,
        *,
        head_hidden: int = 16,
        head_epochs: int = 250,
        head_lr: float = 5e-2,
        head_weight_decay: float = 0.0,
        residual_clip: float = 15.0,
        seed: int = 1234,
        verbose: bool = True,
        **base_kwargs,
    ) -> None:
        self.base = base
        self.constrained_features = tuple(constrained_features)
        self.head_hidden = int(head_hidden)
        self.head_epochs = int(head_epochs)
        self.head_lr = float(head_lr)
        self.head_weight_decay = float(head_weight_decay)
        self.residual_clip = float(residual_clip)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.base_kwargs = dict(base_kwargs)

        # learned state
        self.base_: RegimeProbNet | None = None
        self.constrained_: list[str] = []
        self.s0_: dict[str, float] = {}
        self._snow_mean: np.ndarray | None = None
        self._snow_std: np.ndarray | None = None
        self._snow_zlo: np.ndarray | None = None
        self._snow_zhi: np.ndarray | None = None
        self._head_lo: float = -np.inf
        self._head_hi: float = np.inf
        self._head = None
        self._fitted = False

    # -- sklearn-style introspection (so ensembles can clone it) ------------- #
    def get_params(self) -> dict:
        return {
            "base": self.base,
            "constrained_features": self.constrained_features,
            "head_hidden": self.head_hidden,
            "head_epochs": self.head_epochs,
            "head_lr": self.head_lr,
            "head_weight_decay": self.head_weight_decay,
            "residual_clip": self.residual_clip,
            "seed": self.seed,
            "verbose": self.verbose,
            **self.base_kwargs,
        }

    # -- base construction --------------------------------------------------- #
    def _base_defaults(self) -> dict:
        # Snow is frozen in the base path, so a soft penalty on it is moot;
        # default it OFF and let the hard head own the snow dependence.
        return dict(physics=False, lambda_phys=0.0, seed=self.seed, verbose=self.verbose)

    def _make_base(self) -> RegimeProbNet:
        if self.base is not None:
            return self.base
        return RegimeProbNet(**{**self._base_defaults(), **self.base_kwargs})

    # -- snow-feature surgery ------------------------------------------------ #
    def _drop_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the raw constrained columns so the backbone never sees them."""
        present = [f for f in self.constrained_ if f in df.columns]
        return df.drop(columns=present) if present else df

    def _snow_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised raw-snow design matrix, shape ``(n, d_in)``."""
        cols = []
        for j, f in enumerate(self.constrained_):
            x = df[f].to_numpy(float) if f in df.columns else np.full(len(df), self.s0_[f])
            z = (x - self._snow_mean[j]) / self._snow_std[j]
            z = np.where(np.isfinite(z), z, 0.0)
            if self._snow_zlo is not None:  # clamp to the train range: caps tail
                z = np.clip(z, self._snow_zlo[j], self._snow_zhi[j])  # extrapolation,
            cols.append(z)                                            # stays monotone
        return np.column_stack(cols).astype(np.float32)

    def _base_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Base log-residual; ``df`` is passed as-is (the base ignores raw snow)."""
        return np.asarray(self.base_.predict_residual(df), float).ravel()

    def _head_predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.constrained_ or self._head is None:
            return np.zeros(len(df), float)
        import torch

        s = torch.as_tensor(self._snow_matrix(df), dtype=torch.float32)
        self._head.eval()
        with torch.no_grad():
            out = self._head(s).cpu().numpy().astype(float).ravel()
        return np.clip(out, self._head_lo, self._head_hi)  # band-limit (monotone-safe)

    # -- fitting ------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "HardMonoProbNetCorrector":
        """Fit the snow-free base flagship, then fit the hard monotone snow head."""
        import torch

        train = validate(train).reset_index(drop=True)
        if valid is not None:
            valid = validate(valid).reset_index(drop=True)

        self.constrained_ = [f for f in self.constrained_features if f in train.columns]

        # fit the backbone on the table with the raw constrained columns removed,
        # so it has no dependence on them (the monotonicity guarantee).
        base = self._make_base()
        base.fit(self._drop_raw(train),
                 self._drop_raw(valid) if valid is not None else None)
        self.base_ = base

        if not self.constrained_:
            log.warning("probnet_hardmono: no constrained snow columns present; "
                        "the hard head is a no-op and predictions equal the base.")
            self._fitted = True
            return self

        # snow standardisation (and a median fallback for absent columns) from train
        snow = train[self.constrained_].to_numpy(float)
        self.s0_ = {f: float(np.nanmedian(train[f].to_numpy(float)))
                    for f in self.constrained_}
        self._snow_mean = np.nanmean(snow, axis=0)
        self._snow_std = np.clip(np.nanstd(snow, axis=0), 1e-6, None)
        ztrain = (snow - self._snow_mean) / self._snow_std
        ztrain = np.where(np.isfinite(ztrain), ztrain, 0.0)
        self._snow_zlo = np.nanmin(ztrain, axis=0)
        self._snow_zhi = np.nanmax(ztrain, axis=0)

        # leftover residual the monotone snow head must explain
        base_resid = self._base_residual(train)
        y = train[TARGET_COL].to_numpy(float)
        target = y - base_resid
        ok = np.isfinite(target)
        if int(ok.sum()) < 2:
            raise ValueError("probnet_hardmono: not enough finite rows to fit head")

        # band-limit the snow contribution to a *robust* spread of the leftover
        # residual (median +/- 3 MAD) so the heavy-tailed log-residual outliers
        # of small-discharge rows cannot drive ``exp`` to blow up.  The head's
        # training target is winsorised to the same band for the same reason.
        t = target[ok]
        med = float(np.median(t))
        scale = 1.4826 * float(np.median(np.abs(t - med))) + 1e-3
        self._head_lo, self._head_hi = med - 3.0 * scale, med + 3.0 * scale
        t = np.clip(t, self._head_lo, self._head_hi)

        s = self._snow_matrix(train)[ok]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        head = _build_mono_head(len(self.constrained_), self.head_hidden)
        opt = torch.optim.Adam(head.parameters(), lr=self.head_lr,
                               weight_decay=self.head_weight_decay)
        st = torch.as_tensor(s, dtype=torch.float32)
        tt = torch.as_tensor(t, dtype=torch.float32)
        head.train()
        for epoch in range(self.head_epochs):
            opt.zero_grad()
            pred = head(st)
            # Huber: robust to the heavy-tailed log-residual outliers
            loss = torch.nn.functional.smooth_l1_loss(pred, tt, beta=1.0)
            loss.backward()
            opt.step()
        head.eval()
        self._head = head
        self._fitted = True
        if self.verbose:
            log.info("probnet_hardmono fitted: base=%s, hard-monotone over %s "
                     "(head_mse=%.4f)", getattr(base, "name", "base"),
                     self.constrained_, float(loss.detach()))
        return self

    # -- inference (location-shifted base distribution) ---------------------- #
    def _check_fitted(self) -> None:
        if not self._fitted or self.base_ is None:
            raise RuntimeError("HardMonoProbNetCorrector is not fitted; call fit() first")

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Hard-monotone predicted log-residual ``mu_base(R) + h(S)``."""
        self._check_fitted()
        out = self._base_residual(df) + self._head_predict(df)
        return np.clip(out, -self.residual_clip, self.residual_clip)

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)
                          ) -> np.ndarray:
        """Residual quantiles: base quantiles location-shifted by ``h(S)``."""
        self._check_fitted()
        q = np.asarray(self.base_.predict_quantiles(df, quantiles), float)
        return q + self._head_predict(df)[:, None]

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Predictive variance (unchanged by the deterministic location shift)."""
        self._check_fitted()
        return np.asarray(self.base_.predict_variance(df), float).ravel()

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Posterior residual samples: base draws location-shifted by ``h(S)``."""
        self._check_fitted()
        draws = np.asarray(self.base_.sample(df, n=n, seed=seed), float)
        return draws + self._head_predict(df)[:, None]

    def gate_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Delegate the learned soft-regime weights to the wrapped base."""
        self._check_fitted()
        return np.asarray(self.base_.gate_weights(df), float)

    def monotonicity_report(self, df: pd.DataFrame, **kwargs) -> dict[str, float]:
        """Convenience: :func:`monotonicity_violation_rate` for ``self``."""
        return monotonicity_violation_rate(
            self, df, features=tuple(self.constrained_ or self.constrained_features),
            **kwargs)


# --------------------------------------------------------------------------- #
#  (2) Asymmetric-Laplace mixture variant of the flagship                     #
# --------------------------------------------------------------------------- #
#  Each expert is a two-piece exponential (asymmetric Laplace) density with
#  location m, left scale b_lo, right scale b_hi:
#      f(x) = 1/(b_lo+b_hi) * exp((x-m)/b_lo)    for x <  m
#      f(x) = 1/(b_lo+b_hi) * exp(-(x-m)/b_hi)   for x >= m
#  giving component mean   m + (b_hi - b_lo),
#  component variance      2(b_lo^3 + b_hi^3)/(b_lo+b_hi) - (b_hi - b_lo)^2,
#  and the closed-form CDF / inverse-CDF used below.
# --------------------------------------------------------------------------- #
def _ald_component_mean(m, b_lo, b_hi) -> np.ndarray:
    return m + (b_hi - b_lo)


def _ald_component_var(b_lo, b_hi) -> np.ndarray:
    s = np.clip(b_lo + b_hi, 1e-9, None)
    return 2.0 * (b_lo ** 3 + b_hi ** 3) / s - (b_hi - b_lo) ** 2


def _ald_mixture_quantiles(w, m, b_lo, b_hi, probs, n_iter: int = 60) -> np.ndarray:
    """Quantiles of a two-piece-exponential mixture by vectorised bisection."""
    w = np.asarray(w, float)
    m = np.asarray(m, float)
    b_lo = np.clip(np.asarray(b_lo, float), 1e-9, None)
    b_hi = np.clip(np.asarray(b_hi, float), 1e-9, None)
    probs = np.atleast_1d(np.asarray(probs, float))
    psplit = b_lo / (b_lo + b_hi)

    lo = (m - 50.0 * b_lo).min(axis=1)
    hi = (m + 50.0 * b_hi).max(axis=1)
    out = np.empty((m.shape[0], probs.size), float)
    for j, p in enumerate(probs):
        a, b = lo.copy(), hi.copy()
        for _ in range(n_iter):
            mid = 0.5 * (a + b)
            xm = mid[:, None]
            left = psplit * np.exp(np.clip((xm - m) / b_lo, -700.0, 0.0))
            right = 1.0 - (1.0 - psplit) * np.exp(np.clip(-(xm - m) / b_hi, -700.0, 0.0))
            cdf = (w * np.where(xm < m, left, right)).sum(axis=1)
            below = cdf < p
            a = np.where(below, mid, a)
            b = np.where(below, b, mid)
        out[:, j] = 0.5 * (a + b)
    return out


@register
class AsymLaplaceMixtureCorrector(RegimeProbNet):
    """Regime-gated EA-LSTM with an asymmetric-Laplace mixture head.

    Architecturally identical to :class:`RegimeProbNet` -- an entity-aware LSTM
    backbone feeding a regime-gated mixture-of-experts -- but each expert emits a
    *two-piece exponential* (asymmetric-Laplace) density over the log-residual
    instead of a Gaussian.  The model is trained by mixture negative
    log-likelihood (the closed-form Gaussian CRPS has no equally clean
    asymmetric-Laplace analogue), retaining the flagship's soft gate supervision
    and optional SWE/snowmelt monotonicity penalty.  The exponential tails are a
    better match for the right-skewed snowmelt-freshet and rain-on-snow
    residuals, frequently improving tail calibration and CRPS over the Gaussian
    mixture.

    The constructor mirrors :class:`RegimeProbNet` (so it is a drop-in
    replacement in the ablation / ensemble harnesses); ``sigma_floor`` is renamed
    ``scale_floor`` and the primary loss is fixed to ``"nll"``.

    Parameters
    ----------
    scale_floor : float, default 1e-3
        Lower bound added to each expert scale ``b_lo`` / ``b_hi``.
    K, hidden, seq_len, expert_hidden, gate_hidden, epochs, batch_size, lr,
    weight_decay, patience, lambda_gate, lambda_phys, lambda_mse, physics,
    use_gpu, seed, verbose :
        See :class:`~sbc.models.regime_prob_net.RegimeProbNet`.
    """

    name = "probnet_alaplace"
    is_probabilistic = True

    def __init__(
        self,
        K: int = 5,
        hidden: int = 64,
        seq_len: int = 6,
        expert_hidden: int = 32,
        gate_hidden: int = 32,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 12,
        lambda_gate: float = 0.5,
        lambda_phys: float = 0.1,
        lambda_mse: float = 0.0,
        scale_floor: float = 1e-3,
        physics: bool = True,
        use_gpu: bool | None = None,
        seed: int = 1234,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            K=K, hidden=hidden, seq_len=seq_len, expert_hidden=expert_hidden,
            gate_hidden=gate_hidden, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, lambda_gate=lambda_gate,
            lambda_phys=lambda_phys, lambda_mse=lambda_mse, loss="nll",
            sigma_floor=scale_floor, physics=physics, use_gpu=use_gpu, seed=seed,
            verbose=verbose,
        )
        self.scale_floor = float(scale_floor)

    # ------------------------------------------------------------------ #
    #  Network: EA-LSTM backbone + regime-gated asymmetric-Laplace MoE     #
    # ------------------------------------------------------------------ #
    def _build_net(self, d_dyn: int, d_stat: int):  # type: ignore[override]
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        scale_floor = self.scale_floor

        class _EALSTM(nn.Module):
            """Entity-aware LSTM (Kratzert et al., 2019): static -> input gate."""

            def __init__(self, d_dyn: int, d_stat: int, hidden: int):
                super().__init__()
                self.hidden = hidden
                if d_stat > 0:
                    self.input_gate = nn.Linear(d_stat, hidden)
                    self.static_bias = None
                else:
                    self.input_gate = None
                    self.static_bias = nn.Parameter(torch.zeros(hidden))
                self.fgo = nn.Linear(d_dyn + hidden, 3 * hidden)

            def forward(self, x_dyn, x_stat):
                b, length, _ = x_dyn.shape
                if self.input_gate is not None:
                    i = torch.sigmoid(self.input_gate(x_stat))
                else:
                    i = torch.sigmoid(self.static_bias).unsqueeze(0).expand(b, -1)
                h = x_dyn.new_zeros(b, self.hidden)
                c = x_dyn.new_zeros(b, self.hidden)
                for t in range(length):
                    z = self.fgo(torch.cat([h, x_dyn[:, t, :]], dim=1))
                    f, g, o = z.chunk(3, dim=1)
                    c = torch.sigmoid(f) * c + i * torch.tanh(g)
                    h = torch.sigmoid(o) * torch.tanh(c)
                return h

        class _RegimeALDMoE(nn.Module):
            """Regime-gated asymmetric-Laplace mixture over the log-residual."""

            def __init__(self, d_dyn, d_stat, hidden, K, expert_hidden, gate_hidden):
                super().__init__()
                self.encoder = _EALSTM(d_dyn, d_stat, hidden)
                self.gate = nn.Sequential(
                    nn.Linear(hidden + d_dyn, gate_hidden), nn.ReLU(),
                    nn.Linear(gate_hidden, K),
                )
                self.body = nn.Sequential(nn.Linear(hidden, expert_hidden), nn.ReLU())
                self.head_m = nn.Linear(expert_hidden, K)
                self.head_log_blo = nn.Linear(expert_hidden, K)
                self.head_log_bhi = nn.Linear(expert_hidden, K)

            def forward(self, x_dyn, x_stat):
                h = self.encoder(x_dyn, x_stat)
                gate_logits = self.gate(torch.cat([h, x_dyn[:, -1, :]], dim=1))
                w = torch.softmax(gate_logits, dim=1)
                z = self.body(h)
                m = self.head_m(z)
                b_lo = F.softplus(self.head_log_blo(z)) + scale_floor
                b_hi = F.softplus(self.head_log_bhi(z)) + scale_floor
                return w, gate_logits, m, b_lo, b_hi

        return _RegimeALDMoE(d_dyn, d_stat, self.hidden, self.K,
                             self.expert_hidden, self.gate_hidden)

    # ------------------------------------------------------------------ #
    #  Fit (asymmetric-Laplace mixture NLL)                               #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "AsymLaplaceMixtureCorrector":
        import torch
        import torch.nn.functional as F

        train = validate(train).reset_index(drop=True)
        if valid is not None:
            valid = validate(valid).reset_index(drop=True)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.use_gpu = torch.cuda.is_available() if self.use_gpu is None else self.use_gpu
        self.device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

        self._fit_scaler(train)
        d_dyn, d_stat = len(self.dyn_cols), len(self.stat_cols)
        self.net = self._build_net(d_dyn, d_stat).to(self.device)

        def ald_nll(y, w, m, b_lo, b_hi):
            """Negative log-likelihood of a two-piece-exponential mixture."""
            y_ = y[:, None]
            log_norm = torch.log(b_lo + b_hi)
            log_comp = -log_norm - F.relu(y_ - m) / b_hi - F.relu(m - y_) / b_lo
            return -torch.logsumexp(torch.log(w + 1e-12) + log_comp, dim=1)

        def _prepare(df):
            x_dyn, x_stat = self._design_matrices(df)
            y = (df[TARGET_COL].to_numpy(float) - self.y_mean) / self.y_std
            labels = _classify_regimes(df)
            ridx = np.array([REGIME_TO_IDX.get(s, -1) for s in labels], dtype=np.int64)
            ridx[ridx >= self.K] = -1
            melt = np.array([s in MELT_REGIMES for s in labels], dtype=bool)
            return (
                torch.as_tensor(x_dyn, dtype=torch.float32, device=self.device),
                torch.as_tensor(x_stat, dtype=torch.float32, device=self.device),
                torch.as_tensor(y, dtype=torch.float32, device=self.device),
                torch.as_tensor(ridx, dtype=torch.long, device=self.device),
                torch.as_tensor(melt, dtype=torch.bool, device=self.device),
            )

        Xd, Xs, yt, rt, mt = _prepare(train)
        n = Xd.shape[0]
        if valid is not None and len(valid):
            Vd, Vs, vy, _, _ = _prepare(valid)
        else:
            Vd = None

        phys_on = self.physics and self.lambda_phys > 0 and (
            self.smlt_idx is not None or self.swe_idx is not None
        )

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)

        best_state = copy.deepcopy(self.net.state_dict())
        best_metric = np.inf
        bad = 0

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            ep_losses = []
            for s in range(0, n, self.batch_size):
                bi = perm[s: s + self.batch_size].to(self.device)
                xb, sb = Xd[bi], Xs[bi]
                yb, rb, mb = yt[bi], rt[bi], mt[bi]
                if phys_on:
                    xb = xb.detach().clone().requires_grad_(True)

                w, logits, m, b_lo, b_hi = self.net(xb, sb)
                loss = ald_nll(yb, w, m, b_lo, b_hi).mean()

                if self.lambda_mse > 0:
                    mean = (w * (m + b_hi - b_lo)).sum(1)
                    loss = loss + self.lambda_mse * (mean - yb).pow(2).mean()

                if self.lambda_gate > 0:
                    sup = rb >= 0
                    if bool(sup.any()):
                        loss = loss + self.lambda_gate * F.cross_entropy(logits[sup], rb[sup])

                if phys_on and bool(mb.any()):
                    mu_mix = (w * (m + b_hi - b_lo)).sum(1)
                    grad = torch.autograd.grad(mu_mix.sum(), xb, create_graph=True)[0]
                    pen = xb.new_zeros(())
                    if self.smlt_idx is not None:
                        pen = pen + F.relu(-grad[:, -1, self.smlt_idx][mb]).mean()
                    if self.swe_idx is not None:
                        pen = pen + F.relu(-grad[:, -1, self.swe_idx][mb]).mean()
                    loss = loss + self.lambda_phys * pen

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                ep_losses.append(float(loss.detach()))

            train_metric = float(np.mean(ep_losses)) if ep_losses else np.nan
            if Vd is not None:
                self.net.eval()
                with torch.no_grad():
                    w, _, m, b_lo, b_hi = self.net(Vd, Vs)
                    valid_metric = float(ald_nll(vy, w, m, b_lo, b_hi).mean())
            else:
                valid_metric = train_metric
            self.history_["train"].append(train_metric)
            self.history_["valid"].append(valid_metric)

            if valid_metric < best_metric - 1e-5:
                best_metric, bad = valid_metric, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                log.info("epoch %3d | train=%.4f | valid_nll=%.4f | best=%.4f",
                         epoch, train_metric, valid_metric, best_metric)
            if bad >= self.patience:
                if self.verbose:
                    log.info("early stop at epoch %d (best valid_nll=%.4f)", epoch, best_metric)
                break

        self.net.load_state_dict(best_state)
        self.net.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    def _forward(self, df: pd.DataFrame):  # type: ignore[override]
        """Return ``(w, m, b_lo, b_hi)`` in real log-residual units."""
        if not self._fitted:
            raise RuntimeError("AsymLaplaceMixtureCorrector must be fitted before inference")
        import torch

        x_dyn, x_stat = self._design_matrices(df)
        self.net.eval()
        ws, ms, blos, bhis = [], [], [], []
        bs = max(self.batch_size, 1024)
        with torch.no_grad():
            for s in range(0, x_dyn.shape[0], bs):
                xb = torch.as_tensor(x_dyn[s: s + bs], dtype=torch.float32, device=self.device)
                sb = torch.as_tensor(x_stat[s: s + bs], dtype=torch.float32, device=self.device)
                w, _, m, b_lo, b_hi = self.net(xb, sb)
                ws.append(w.cpu().numpy())
                ms.append(m.cpu().numpy())
                blos.append(b_lo.cpu().numpy())
                bhis.append(b_hi.cpu().numpy())
        w = np.concatenate(ws, axis=0)
        m = np.concatenate(ms, axis=0) * self.y_std + self.y_mean
        b_lo = np.clip(np.concatenate(blos, axis=0) * self.y_std, 1e-9, None)
        b_hi = np.clip(np.concatenate(bhis, axis=0) * self.y_std, 1e-9, None)
        return w, m, b_lo, b_hi

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Mixture predictive mean ``sum_k w_k (m_k + b_hi_k - b_lo_k)``."""
        w, m, b_lo, b_hi = self._forward(df)
        return (w * _ald_component_mean(m, b_lo, b_hi)).sum(axis=1)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Total predictive variance (within + between expert), shape ``(n,)``."""
        w, m, b_lo, b_hi = self._forward(df)
        comp_mean = _ald_component_mean(m, b_lo, b_hi)
        comp_var = _ald_component_var(b_lo, b_hi)
        mean = (w * comp_mean).sum(axis=1, keepdims=True)
        return (w * (comp_var + comp_mean ** 2)).sum(axis=1) - mean[:, 0] ** 2

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Residual quantiles from the asymmetric-Laplace mixture, shape ``(n, q)``."""
        w, m, b_lo, b_hi = self._forward(df)
        return _ald_mixture_quantiles(w, m, b_lo, b_hi, quantiles)

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Exact inverse-CDF posterior residual samples, shape ``(n_rows, n)``."""
        w, m, b_lo, b_hi = self._forward(df)
        rng = np.random.default_rng(seed)
        nr, K = m.shape
        cum = np.cumsum(w, axis=1)
        u = rng.random((nr, n))
        comp = (u[:, :, None] > cum[:, None, :]).sum(axis=2)
        comp = np.clip(comp, 0, K - 1)
        rows = np.arange(nr)[:, None]
        m_sel = m[rows, comp]
        blo_sel = np.clip(b_lo[rows, comp], 1e-9, None)
        bhi_sel = np.clip(b_hi[rows, comp], 1e-9, None)
        psplit = blo_sel / (blo_sel + bhi_sel)
        u2 = rng.random((nr, n))
        left = m_sel + blo_sel * np.log(np.clip(u2 / psplit, 1e-12, None))
        right = m_sel - bhi_sel * np.log(np.clip((1.0 - u2) / (1.0 - psplit), 1e-12, None))
        return np.where(u2 <= psplit, left, right)

    def gate_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Learned soft regime / expert weights, shape ``(n, K)``."""
        w, _, _, _ = self._forward(df)
        return w


# --------------------------------------------------------------------------- #
#  Self-test: soft-vs-hard violation rate & Gaussian-vs-ALaplace CRPS          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..schemas import OBS_COL, SIM_COL, make_target
    from ..synthetic import generate
    from ..validation.metrics import crps_ensemble, kge_prime
    from ..validation.splits import temporal_split
    from .base import available

    # --- small synthetic table -> features -> regimes ----------------------- #
    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(df, scale="decadal"))
    df = validate(df)
    df[TARGET_COL] = make_target(df[OBS_COL].values, df[SIM_COL].values)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    print(f"[constraint_variants] registered: "
          f"{[m for m in available() if m in ('probnet_hardmono', 'probnet_alaplace')]}")
    print(f"[constraint_variants] gauges={df['code'].nunique()} "
          f"train={len(train)} test={len(test)}")

    tiny = dict(K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                epochs=3, batch_size=512, patience=3, seed=0, verbose=False)
    y = test[TARGET_COL].to_numpy(float)

    # --- (1) SOFT flagship vs HARD-monotone wrapper ------------------------- #
    soft = RegimeProbNet(lambda_gate=0.5, lambda_phys=0.1, physics=True, **tiny).fit(train, test)
    hard = HardMonoProbNetCorrector(
        head_hidden=16, head_epochs=200, lambda_gate=0.5, **tiny).fit(train, test)

    v_soft = monotonicity_violation_rate(soft, test)
    v_hard = monotonicity_violation_rate(hard, test)
    kge_soft = kge_prime(test[OBS_COL].values, soft.predict(test))["kge"]
    kge_hard = kge_prime(test[OBS_COL].values, hard.predict(test))["kge"]
    print(f"[constraint_variants] MONOTONICITY  soft viol_rate={v_soft['viol_rate']:.4f} "
          f"(smlt={v_soft.get('viol_smlt', float('nan')):.4f}, "
          f"swe={v_soft.get('viol_swe', float('nan')):.4f}) "
          f"-> HARD viol_rate={v_hard['viol_rate']:.4f}")
    print(f"[constraint_variants]               KGE' soft={kge_soft:+.3f} | hard={kge_hard:+.3f}")

    # HARD quantiles must stay monotone after the location shift
    hq = hard.predict_quantiles(test, (0.05, 0.25, 0.5, 0.75, 0.95))
    hq_mono = bool(np.all(np.diff(hq, axis=1) >= -1e-9))

    # --- (2) Gaussian flagship vs asymmetric-Laplace mixture CRPS ----------- #
    ala = AsymLaplaceMixtureCorrector(lambda_gate=0.5, lambda_phys=0.1, physics=True,
                                      scale_floor=1e-3, **tiny).fit(train, test)
    n_draw = 100
    g_draws = soft.sample(test, n=n_draw, seed=1)
    a_draws = ala.sample(test, n=n_draw, seed=1)
    crps_g = crps_ensemble(y, g_draws)
    crps_a = crps_ensemble(y, a_draws)
    aq = ala.predict_quantiles(test, (0.05, 0.25, 0.5, 0.75, 0.95))
    aq_mono = bool(np.all(np.diff(aq, axis=1) >= -1e-9))
    avar = ala.predict_variance(test)
    kge_ala = kge_prime(test[OBS_COL].values, ala.predict(test))["kge"]
    print(f"[constraint_variants] LIKELIHOOD    CRPS(resid) Gaussian={crps_g:.4f} | "
          f"asym-Laplace={crps_a:.4f} (lower is sharper) | KGE' alaplace={kge_ala:+.3f}")
    print(f"[constraint_variants]               ala quantiles{aq.shape} monotone={aq_mono} | "
          f"var>=0={bool(np.all(avar >= 0))} | gate{ala.gate_weights(test).shape}")

    # --- assertions --------------------------------------------------------- #
    assert v_hard["viol_rate"] <= v_soft["viol_rate"] + 1e-9, "hard must not violate more than soft"
    assert v_hard["viol_rate"] < 1e-3, "hard monotonicity not enforced (~0 expected)"
    assert hq.shape == (len(test), 5) and hq_mono, "hard quantiles not monotone"
    assert aq.shape == (len(test), 5) and aq_mono, "alaplace quantiles not monotone"
    assert np.all(np.isfinite(avar)) and np.all(avar >= 0), "bad alaplace variance"
    assert a_draws.shape == (len(test), n_draw), "bad alaplace sample shape"
    assert np.isfinite(crps_g) and np.isfinite(crps_a), "non-finite CRPS"
    print("[constraint_variants] SELF-TEST OK")
