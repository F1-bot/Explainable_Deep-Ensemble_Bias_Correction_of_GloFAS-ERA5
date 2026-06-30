"""Regime-gated *generative* probabilistic residual corrector (implicit quantiles).

This module answers the diffusion-UQ frontier (DRUM, 2025; HydroDiffusion, 2025)
with the smallest genuinely *generative* uncertainty model that is tractable for a
sparse decadal/daily Central-Asian gauge network: a regime-conditioned **implicit
quantile network** (IQN; Dabney et al., 2018, *ICML*) over the GloFAS log-residual.

Why generative -- and why not a Gaussian mixture
-------------------------------------------------
The flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet` is a *parametric*
Gaussian mixture: every predictive shape is a sum of Gaussians, so its tails are
fixed by construction.  An IQN instead learns the **quantile function**
``Q(tau | x)`` directly and draws samples by the inverse-CDF transform
``y = Q(U, x), U ~ Uniform(0, 1)`` -- a one-step generative model that places *no*
shape assumption on the predictive law.  Trained on the quantile (pinball) loss,
whose average over ``tau ~ Uniform(0, 1)`` is exactly the CRPS (Koenker, 2005;
Gneiting & Raftery, 2007), it optimises the same strictly-proper score the
diffusion baselines report.

The genuinely-new bit: regime-adaptive predictive *shape*
---------------------------------------------------------
The quantile function is parameterised as a **non-negative combination of fixed
monotone basis functions of ``tau``**

    ``Q(tau | h) = loc(h)  +  sum_m  a_m(h) * B_m(tau)``,   ``a_m(h) >= 0``,

where the basis ``{B_m}`` is a probit term ``Phi^{-1}(tau)`` (a Gaussian
backbone) plus a bank of monotone piecewise-linear ramps (an I-spline-like
monotone flow over ``tau``; cf. Durkan et al., 2019; Gasthaus et al., 2019).
Because every ``B_m`` is non-decreasing in ``tau`` and every coefficient
``a_m(h) >= 0``, ``Q`` is **monotone in ``tau`` by construction** -- the predicted
quantiles can never cross.  Crucially the coefficients ``a_m(h)`` are an explicit
function of the *regime conditioning* ``h`` (the rule-based regime one-hot of
:mod:`sbc.features.regimes`, or -- when a fitted flagship is supplied -- its soft
gate ``RegimeProbNet.gate_weights``).  The predictive **shape** therefore adapts
per regime: a heavy-tailed, right-skewed melt-freshet law versus a tight,
near-symmetric recession law emerge from the data rather than from a fixed
Gaussian family.  This regime-conditioned generative shape is the contribution
over the Gaussian-mixture flagship.

API
---
``predict_residual`` returns the predictive median ``Q(0.5)``; ``predict_quantiles``
evaluates the (crossing-free) implicit quantile function at the requested levels;
``sample`` draws inverse-CDF samples; ``predict_variance`` integrates the quantile
function.  All heavy imports (``torch``, ``scipy``) are deferred into methods, the
backbone is a small MLP (an LSTM is intentionally *not* required at this data
size), matplotlib is never imported, and the model is GPU-optional.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..schemas import (
    OBS_COL,
    SIM_COL,
    TARGET_COL,
    feature_columns,
    make_target,
    validate,
)
from ..utils import get_logger
from .base import BaseCorrector, register
from .regime_prob_net import REGIME_NAMES, REGIME_TO_IDX, _classify_regimes

log = get_logger(__name__)

#: numerical guard keeping ``tau`` strictly inside the open unit interval
_TAU_EPS: float = 1e-4
#: symmetric clip on the probit basis ``Phi^{-1}(tau)`` (tames the open-interval tails)
_PROBIT_CLIP: float = 5.0

__all__ = ["GenerativeResidualCorrector", "DEFAULT_QUANTILE_LEVELS"]

#: default evaluation grid (mirrors the probabilistic baselines / calibration tools)
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = tuple(np.round(np.linspace(0.05, 0.95, 19), 4))


# --------------------------------------------------------------------------- #
#  Monotone basis of the quantile variable ``tau`` (pure helpers)             #
# --------------------------------------------------------------------------- #
def _inv_softplus(y: float) -> float:
    """Inverse of ``softplus``: return ``x`` with ``log(1 + exp(x)) = y``."""
    y = float(y)
    return float(np.log(np.expm1(y))) if y < 20.0 else y


def _basis_np(tau: np.ndarray, knots: np.ndarray,
              clip: float = _PROBIT_CLIP) -> np.ndarray:
    """Monotone basis matrix ``B(tau)`` (NumPy), shape ``tau.shape + (M,)``.

    The columns are, in order: the probit term ``Phi^{-1}(tau)``, the linear term
    ``tau``, the right ramps ``relu(tau - c_k)`` and the left ramps
    ``min(tau - c_k, 0)`` over the ``knots`` ``c_k``.  Every column is
    non-decreasing in ``tau``; combined with non-negative coefficients this yields
    a crossing-free quantile function.

    Parameters
    ----------
    tau : numpy.ndarray
        Probability levels in ``(0, 1)`` (any shape).
    knots : numpy.ndarray, shape (K,)
        Interior ramp knots.
    clip : float
        Symmetric clip applied to the probit term.

    Returns
    -------
    numpy.ndarray
        Basis values with a trailing axis of size ``M = 2 + 2 * K``.
    """
    from scipy.special import erfinv

    tau = np.clip(np.asarray(tau, float), _TAU_EPS, 1.0 - _TAU_EPS)
    probit = np.clip(erfinv(2.0 * tau - 1.0) * np.sqrt(2.0), -clip, clip)
    lin = tau
    d = tau[..., None] - np.asarray(knots, float)
    right = np.maximum(d, 0.0)
    left = np.minimum(d, 0.0)
    return np.concatenate([probit[..., None], lin[..., None], right, left], axis=-1)


# --------------------------------------------------------------------------- #
#  The model                                                                  #
# --------------------------------------------------------------------------- #
@register
class GenerativeResidualCorrector(BaseCorrector):
    """Regime-gated generative residual corrector via an implicit quantile network.

    A small MLP encodes the (standardised) features together with a *regime
    conditioning* block into a location ``loc`` and a vector of non-negative basis
    coefficients ``a``; the predictive quantile function is the monotone
    combination ``Q(tau) = loc + sum_m a_m B_m(tau)`` (see the module docstring).
    The network is trained by the quantile (pinball) loss at randomly sampled
    ``tau`` -- the implicit-quantile-network scheme of Dabney et al. (2018), whose
    ``tau``-average is the CRPS.

    Parameters
    ----------
    hidden : int, default 64
        Width of the two-layer MLP trunk.
    n_knots : int, default 9
        Number of interior ramp knots in the monotone ``tau`` basis (the basis
        dimension is ``M = 2 + 2 * n_knots``).
    n_tau : int, default 16
        Number of ``tau`` samples drawn per row each training step (the Monte-Carlo
        estimator of the CRPS / averaged pinball loss).
    regime_conditioning : {"onehot", "gate"}, default "onehot"
        ``"onehot"`` feeds the rule-based regime one-hot of
        :func:`sbc.features.regimes`; ``"gate"`` uses ``flagship.gate_weights`` (a
        soft regime gate) and requires ``flagship`` to be a *fitted*
        :class:`~sbc.models.regime_prob_net.RegimeProbNet`.  ``"gate"`` silently
        falls back to ``"onehot"`` when no fitted flagship is supplied.
    flagship : RegimeProbNet, optional
        A fitted flagship whose soft gate supplies the conditioning when
        ``regime_conditioning="gate"``.
    epochs, batch_size, lr, weight_decay, patience : training schedule.
    huber_beta : float, default 0.0
        When ``> 0`` the pinball loss is Huberised below ``huber_beta`` (smooth
        near the kink); ``0`` uses the exact pinball loss.
    use_gpu : bool or None, default None
        ``None`` auto-selects CUDA when available.
    seed : int, default 1234
        Reproducibility seed.
    verbose : bool, default True
        Emit per-epoch training logs.

    Attributes
    ----------
    feat_cols_ : list of str
        Feature columns discovered with :func:`sbc.schemas.feature_columns`.
    cond_names_ : list of str
        Labels of the regime-conditioning block actually used.
    knots_ : numpy.ndarray
        Interior ramp knots of the monotone basis.
    """

    name = "gen_resid"
    is_probabilistic = True

    def __init__(
        self,
        hidden: int = 64,
        n_knots: int = 9,
        n_tau: int = 16,
        regime_conditioning: str = "onehot",
        flagship: "BaseCorrector | None" = None,
        epochs: int = 300,
        batch_size: int = 512,
        lr: float = 3e-3,
        weight_decay: float = 1e-5,
        patience: int = 30,
        huber_beta: float = 0.0,
        use_gpu: bool | None = None,
        seed: int = 1234,
        verbose: bool = True,
    ) -> None:
        self.hidden = int(hidden)
        self.n_knots = int(n_knots)
        self.n_tau = int(n_tau)
        self.regime_conditioning = str(regime_conditioning).lower()
        self.flagship = flagship
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.huber_beta = float(huber_beta)
        self.use_gpu = use_gpu
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # fitted state
        self.feat_cols_: list[str] = []
        self.cond_names_: list[str] = []
        self.knots_: np.ndarray = np.linspace(0.1, 0.9, self.n_knots)
        self.M_: int = 2 + 2 * self.n_knots
        self.feat_mean_ = self.feat_std_ = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self._cond_mode = self.regime_conditioning
        self.net = None
        self.device = "cpu"
        self.history_: dict[str, list[float]] = {"train": [], "valid": []}
        self._fitted = False

    # -- sklearn-style introspection (so ensembles can clone it) ------------- #
    def get_params(self) -> dict:
        return {
            "hidden": self.hidden, "n_knots": self.n_knots, "n_tau": self.n_tau,
            "regime_conditioning": self.regime_conditioning, "flagship": self.flagship,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
            "weight_decay": self.weight_decay, "patience": self.patience,
            "huber_beta": self.huber_beta, "use_gpu": self.use_gpu,
            "seed": self.seed, "verbose": self.verbose,
        }

    # ------------------------------------------------------------------ #
    #  Conditioning & design matrix                                       #
    # ------------------------------------------------------------------ #
    def _regime_conditioning(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Return the regime conditioning block ``(n, d_cond)`` and its labels."""
        if self._cond_mode == "gate" and self.flagship is not None \
                and getattr(self.flagship, "_fitted", False):
            gate = np.asarray(self.flagship.gate_weights(df), float)
            names = [f"gate_{k}" for k in range(gate.shape[1])]
            return np.where(np.isfinite(gate), gate, 0.0).astype(np.float32), names
        # rule-based one-hot over the canonical regimes
        labels = _classify_regimes(df)
        idx = np.array([REGIME_TO_IDX.get(s, REGIME_TO_IDX["recession"]) for s in labels],
                       dtype=int)
        onehot = np.zeros((len(df), len(REGIME_NAMES)), dtype=np.float32)
        onehot[np.arange(len(df)), idx] = 1.0
        return onehot, [f"regime_{r}" for r in REGIME_NAMES]

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised ``[features | regime-conditioning]`` matrix ``(n, d_in)``."""
        feats = df[self.feat_cols_].to_numpy(float)
        z = (feats - self.feat_mean_) / self.feat_std_
        z = np.where(np.isfinite(z), z, 0.0)
        cond, _ = self._regime_conditioning(df)
        return np.concatenate([z.astype(np.float32), cond], axis=1)

    # ------------------------------------------------------------------ #
    #  Network                                                            #
    # ------------------------------------------------------------------ #
    def _build_net(self, d_in: int):
        import torch
        import torch.nn as nn

        M = self.M_

        class _IQNHead(nn.Module):
            """MLP trunk -> (loc, non-negative basis coefficients)."""

            def __init__(self, d_in: int, hidden: int, M: int) -> None:
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(d_in, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                self.head_loc = nn.Linear(hidden, 1)
                self.head_loga = nn.Linear(hidden, M)
                # Initialise near the standardised marginal: a Gaussian backbone
                # (probit coefficient ~ 1) with small ramp coefficients, so even a
                # lightly-trained model is already ~calibrated.
                with torch.no_grad():
                    self.head_loc.weight.mul_(0.1)
                    self.head_loc.bias.zero_()
                    self.head_loga.weight.mul_(0.05)
                    b = torch.full((M,), _inv_softplus(0.05))
                    b[0] = _inv_softplus(1.0)          # probit term -> standard normal
                    self.head_loga.bias.copy_(b)

            def forward(self, x):
                z = self.trunk(x)
                loc = self.head_loc(z).squeeze(-1)
                a = torch.nn.functional.softplus(self.head_loga(z))  # >= 0 -> monotone
                return loc, a

        return _IQNHead(int(d_in), self.hidden, M)

    @staticmethod
    def _basis_torch(tau, knots_t, clip: float = _PROBIT_CLIP):
        """Monotone basis ``B(tau)`` (torch), trailing axis ``M = 2 + 2 K``."""
        import torch

        tau = tau.clamp(_TAU_EPS, 1.0 - _TAU_EPS)
        probit = (torch.erfinv(2.0 * tau - 1.0) * (2.0 ** 0.5)).clamp(-clip, clip)
        lin = tau
        d = tau.unsqueeze(-1) - knots_t
        right = torch.relu(d)
        left = -torch.relu(-d)
        return torch.cat([probit.unsqueeze(-1), lin.unsqueeze(-1), right, left], dim=-1)

    # ------------------------------------------------------------------ #
    #  Fit                                                                #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "GenerativeResidualCorrector":
        import torch

        train = validate(train).reset_index(drop=True)
        if valid is not None:
            valid = validate(valid).reset_index(drop=True)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.use_gpu = torch.cuda.is_available() if self.use_gpu is None else self.use_gpu
        self.device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

        self._cond_mode = self.regime_conditioning
        if self._cond_mode == "gate" and not (
            self.flagship is not None and getattr(self.flagship, "_fitted", False)
        ):
            log.warning("gen_resid: regime_conditioning='gate' but no fitted flagship "
                        "supplied; falling back to rule-based one-hot conditioning.")
            self._cond_mode = "onehot"

        # feature discovery & scaling (never hard-coded; from schemas)
        self.feat_cols_ = feature_columns(train)
        if not self.feat_cols_:
            raise ValueError("gen_resid needs at least one numeric feature column")
        feats = train[self.feat_cols_].to_numpy(float)
        self.feat_mean_ = np.nanmean(feats, axis=0)
        self.feat_std_ = np.clip(np.nanstd(feats, axis=0), 1e-6, None)

        y_raw = (train[TARGET_COL].to_numpy(float) if TARGET_COL in train.columns
                 else make_target(train[OBS_COL].to_numpy(float),
                                  train[SIM_COL].to_numpy(float)))
        yf = y_raw[np.isfinite(y_raw)]
        self.y_mean_ = float(np.mean(yf)) if yf.size else 0.0
        self.y_std_ = float(np.clip(np.std(yf), 1e-6, None)) if yf.size else 1.0

        _, self.cond_names_ = self._regime_conditioning(train)

        X = self._design(train)
        y = (y_raw - self.y_mean_) / self.y_std_
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        d_in = X.shape[1]

        self.net = self._build_net(d_in).to(self.device)
        knots_t = torch.as_tensor(self.knots_, dtype=torch.float32, device=self.device)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        n = Xt.shape[0]

        if valid is not None and len(valid):
            Xv = self._design(valid)
            yv_raw = valid[TARGET_COL].to_numpy(float)
            yv = (yv_raw - self.y_mean_) / self.y_std_
            okv = np.isfinite(yv)
            Vt = torch.as_tensor(Xv[okv], dtype=torch.float32, device=self.device)
            vy = torch.as_tensor(yv[okv], dtype=torch.float32, device=self.device)
        else:
            Vt = None

        def _pinball(diff, tau):
            base = torch.maximum(tau * diff, (tau - 1.0) * diff)
            if self.huber_beta > 0.0:
                b = self.huber_beta
                ad = diff.abs()
                huber = torch.where(ad <= b, 0.5 * diff * diff / b, ad - 0.5 * b)
                return torch.abs(tau - (diff < 0).float()) * huber
            return base

        def _loss(net, xb, yb, gen):
            loc, a = net(xb)
            tau = torch.rand(xb.shape[0], self.n_tau, generator=gen, device=self.device)
            tau = tau.clamp(_TAU_EPS, 1.0 - _TAU_EPS)
            B = self._basis_torch(tau, knots_t)                 # (b, n_tau, M)
            q = loc.unsqueeze(1) + (a.unsqueeze(1) * B).sum(-1)  # (b, n_tau)
            diff = yb.unsqueeze(1) - q
            return _pinball(diff, tau).mean()

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        perm_gen = torch.Generator().manual_seed(self.seed)

        best_state = copy.deepcopy(self.net.state_dict())
        best_metric = np.inf
        bad = 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=perm_gen).to(self.device)
            ep = []
            for s in range(0, n, self.batch_size):
                bi = perm[s: s + self.batch_size]
                loss = _loss(self.net, Xt[bi], yt[bi], gen)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                ep.append(float(loss.detach()))
            train_metric = float(np.mean(ep)) if ep else np.nan

            if Vt is not None:
                self.net.eval()
                with torch.no_grad():
                    valid_metric = float(_loss(self.net, Vt, vy, gen))
            else:
                valid_metric = train_metric
            self.history_["train"].append(train_metric)
            self.history_["valid"].append(valid_metric)

            if valid_metric < best_metric - 1e-6:
                best_metric, bad = valid_metric, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0
                                 or epoch == self.epochs - 1):
                log.info("epoch %3d | train_pinball=%.4f | valid_pinball=%.4f | best=%.4f",
                         epoch, train_metric, valid_metric, best_metric)
            if bad >= self.patience:
                if self.verbose:
                    log.info("early stop at epoch %d (best valid_pinball=%.4f)",
                             epoch, best_metric)
                break

        self.net.load_state_dict(best_state)
        self.net.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    def _loc_a(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(loc, a)`` in the *standardised* residual space, NumPy arrays."""
        if not self._fitted:
            raise RuntimeError("GenerativeResidualCorrector must be fitted before inference")
        import torch

        X = self._design(df)
        self.net.eval()
        locs, as_ = [], []
        bs = max(self.batch_size, 1024)
        with torch.no_grad():
            for s in range(0, X.shape[0], bs):
                xb = torch.as_tensor(X[s: s + bs], dtype=torch.float32, device=self.device)
                loc, a = self.net(xb)
                locs.append(loc.cpu().numpy())
                as_.append(a.cpu().numpy())
        return np.concatenate(locs, axis=0), np.concatenate(as_, axis=0)

    def _quantiles_real(self, loc: np.ndarray, a: np.ndarray,
                        tau: np.ndarray) -> np.ndarray:
        """Evaluate ``Q`` (real log-residual units) at probabilities ``tau`` ``(T,)``.

        Returns an array of shape ``(n, T)`` aligned to ``loc`` / ``a`` rows.
        """
        B = _basis_np(np.asarray(tau, float), self.knots_)   # (T, M)
        q_std = loc[:, None] + a @ B.T                        # (n, T)
        return self.y_mean_ + self.y_std_ * q_std

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predictive **median** log-residual ``Q(0.5)``, shape ``(n,)``."""
        loc, a = self._loc_a(df)
        return self._quantiles_real(loc, a, np.array([0.5]))[:, 0]

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)
                          ) -> np.ndarray:
        """Crossing-free residual quantiles, shape ``(n, len(quantiles))``.

        The implicit quantile function is monotone in ``tau`` by construction; a
        defensive ``maximum.accumulate`` removes any residual floating-point
        non-monotonicity before the columns are returned in the requested order.
        """
        q = np.atleast_1d(np.asarray(quantiles, float))
        order = np.argsort(q)
        loc, a = self._loc_a(df)
        sorted_q = self._quantiles_real(loc, a, q[order])
        sorted_q = np.maximum.accumulate(sorted_q, axis=1)
        out = np.empty_like(sorted_q)
        out[:, order] = sorted_q
        return out

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Inverse-CDF generative draws ``Q(U)``, shape ``(n_rows, n)``."""
        loc, a = self._loc_a(df)
        rng = np.random.default_rng(seed)
        tau = rng.uniform(_TAU_EPS, 1.0 - _TAU_EPS, size=(loc.shape[0], n))
        B = _basis_np(tau, self.knots_)                       # (n_rows, n, M)
        q_std = loc[:, None] + np.einsum("rm,rnm->rn", a, B)
        return self.y_mean_ + self.y_std_ * q_std

    def predict_variance(self, df: pd.DataFrame, n_grid: int = 99) -> np.ndarray:
        """Predictive variance by integrating the quantile function, shape ``(n,)``.

        A dense uniform ``tau`` grid turns ``Q`` into equally-weighted inverse-CDF
        samples whose variance estimates the predictive variance.
        """
        loc, a = self._loc_a(df)
        tau = np.linspace(_TAU_EPS, 1.0 - _TAU_EPS, int(n_grid))
        q = self._quantiles_real(loc, a, tau)                 # (n, n_grid)
        return q.var(axis=1)

    def gate_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Regime-conditioning block used by the head, shape ``(n, d_cond)``.

        For API parity with :meth:`RegimeProbNet.gate_weights`: returns the soft
        flagship gate when ``regime_conditioning='gate'``, else the rule-based
        regime one-hot.  Column labels are in :pyattr:`cond_names_`.
        """
        cond, _ = self._regime_conditioning(df)
        return cond


# --------------------------------------------------------------------------- #
#  Self-test: CRPS vs the Gaussian flagship; monotone & ~calibrated quantiles  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..schemas import OBS_COL as _OBS, SIM_COL as _SIM
    from ..synthetic import generate
    from ..validation.calibration import coverage
    from ..validation.metrics import crps_ensemble, kge_prime
    from ..validation.splits import temporal_split
    from .base import available
    from .regime_prob_net import RegimeProbNet

    # --- small synthetic table -> features -> regimes ----------------------- #
    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(df, scale="decadal"))
    df = validate(df)
    df[TARGET_COL] = make_target(df[_OBS].values, df[_SIM].values)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    y = test[TARGET_COL].to_numpy(float)
    print(f"[generative_head] registered gen_resid={'gen_resid' in available()} | "
          f"gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    # --- Gaussian-mixture flagship (tiny, 3 epochs) ------------------------- #
    flag = RegimeProbNet(K=3, hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                         epochs=3, batch_size=512, patience=3, lambda_gate=0.5,
                         lambda_phys=0.0, seed=0, verbose=False).fit(train, test)
    crps_flag = crps_ensemble(y, flag.sample(test, n=100, seed=1))

    # --- generative IQN corrector (one-hot conditioning) -------------------- #
    gen = GenerativeResidualCorrector(hidden=48, n_knots=9, n_tau=16, epochs=250,
                                      batch_size=512, lr=3e-3, patience=40,
                                      seed=0, verbose=False).fit(train, test)
    crps_gen = crps_ensemble(y, gen.sample(test, n=100, seed=1))

    # --- generative IQN conditioned on the flagship soft gate --------------- #
    gen_g = GenerativeResidualCorrector(hidden=48, n_knots=9, n_tau=16, epochs=120,
                                        regime_conditioning="gate", flagship=flag,
                                        batch_size=512, lr=3e-3, patience=40,
                                        seed=0, verbose=False).fit(train, test)
    crps_gen_g = crps_ensemble(y, gen_g.sample(test, n=100, seed=1))

    # --- quantiles: monotone (no crossing) & ~calibrated 90% coverage ------- #
    levels = np.round(np.linspace(0.05, 0.95, 19), 4)
    q = gen.predict_quantiles(test, tuple(levels))
    monotone = bool(np.all(np.diff(q, axis=1) >= -1e-9))
    cov90 = coverage(y, q[:, 0], q[:, -1])           # [q05, q95] -> nominal 0.90
    var = gen.predict_variance(test)
    kge_raw = kge_prime(test[_OBS].values, test[_SIM].values)["kge"]
    kge_gen = kge_prime(test[_OBS].values, gen.predict(test))["kge"]
    gw = gen.gate_weights(test)

    print(f"[generative_head] CRPS(log-resid)  gen(onehot)={crps_gen:.4f} | "
          f"gen(gate)={crps_gen_g:.4f} | Gaussian flagship={crps_flag:.4f} "
          f"(lower is sharper)")
    print(f"[generative_head] quantiles{q.shape} monotone={monotone} | "
          f"cov90={cov90:.3f} (target ~0.90) | var>=0={bool(np.all(var >= 0))}")
    print(f"[generative_head] KGE' raw={kge_raw:+.3f} -> gen={kge_gen:+.3f} | "
          f"cond{gw.shape}={gen.cond_names_[:3]}... rowsum~{gw.sum(1).mean():.2f}")

    # --- assertions --------------------------------------------------------- #
    assert "gen_resid" in available(), "gen_resid not registered"
    assert q.shape == (len(test), 19) and monotone, "quantiles cross / wrong shape"
    assert np.all(np.isfinite(var)) and np.all(var >= 0), "bad predictive variance"
    assert gw.shape[0] == len(test) and np.all(np.isfinite(gw)), "bad conditioning block"
    assert np.isfinite(crps_gen) and np.isfinite(crps_flag), "non-finite CRPS"
    assert 0.70 <= cov90 <= 0.99, f"90% interval coverage {cov90:.3f} not ~calibrated"
    print("[generative_head] SELF-TEST OK")
