"""CMAL probabilistic head -- the streamflow SOTA UQ of Klotz et al. (2022).

:class:`CMALCorrector` implements the *Countable Mixture of Asymmetric
Laplacians* (CMAL) of Klotz et al. (2022, *HESS* 26, 1673-1693,
doi:10.5194/hess-26-1673-2022), which their large-sample CAMELS benchmark found
to be the best-reliability / best-sharpness deep-learning uncertainty estimator
for rainfall-runoff (out-scoring a Gaussian mixture, an uncountable mixture of
asymmetric Laplacians, and Monte-Carlo dropout).  Here it is repurposed as a
probabilistic *bias-correction* head over the GloFAS log-residual
``log(q_obs+EPS) - log(q_glofas+EPS)``, giving the framework both an upgraded
calibrated-UQ model and the SOTA CMAL baseline that reviewers demand.

Method (Klotz et al., 2022, Appendix B2)
----------------------------------------
An entity-aware LSTM (EA-LSTM; Kratzert et al., 2019) backbone -- shared with the
flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet` -- maps the dynamic
forcing window and the static catchment attributes to the parameters of a
mixture of ``K`` asymmetric-Laplacian distributions (ALDs) over the log-residual.
Each component ``k`` has a location ``m_k``, a scale ``s_k > 0`` and an asymmetry
``tau_k in (0, 1)``, and the components are combined with convex weights
``pi_k`` (softmax).  The ALD density (Klotz Eq. B3; the quantile-regression
parametrisation of Yu and Moyeed, 2001) is

.. math::

    \\mathrm{ALD}(q\\mid m, s, \\tau) = \\frac{\\tau(1-\\tau)}{s}\\,
        \\exp\\!\\big[-\\rho_\\tau\\!\\big(\\tfrac{q-m}{s}\\big)\\big],
        \\qquad \\rho_\\tau(u) = u\\,(\\tau - \\mathbb{1}[u < 0]),

i.e. a two-piece exponential that is sharper at its centre and heavier-tailed
than a Gaussian and, via ``tau``, intrinsically *skewed* -- a better match for
the right-skewed snowmelt-freshet and rain-on-snow residuals than the flagship's
Gaussian mixture.  Training minimises the exact mixture negative log-likelihood
(Klotz Eq. B5)

.. math::

    \\mathcal{L} = -\\log \\sum_{k=1}^{K} \\pi_k\\,\\mathrm{ALD}(q\\mid m_k, s_k, \\tau_k)

evaluated with a numerically stable ``log-sum-exp``.  Activations follow Klotz:
softmax for ``pi``, sigmoid for ``tau``, softplus (plus a small floor) for ``s``.

Because each ALD has a closed-form CDF, the mixture CDF
``F(q) = sum_k pi_k F_k(q)`` is monotone, so the predictive quantiles
(:meth:`predict_quantiles`) and the predictive median used as the point
correction (:meth:`predict_residual`) are obtained by vectorised bisection and
are guaranteed monotone; :meth:`sample` draws exactly via per-component
inverse-CDF.

Regime-conditioned variant
--------------------------
``regime_gated=True`` reproduces the flagship's coupling: the mixture-weight gate
additionally sees the contemporaneous regime forcings and is softly supervised
towards the rule-based regime label (``lambda_gate``), so the CMAL components
acquire the same physical, regime-aligned interpretation.  The default
(``regime_gated=False``) is the literature-faithful Klotz CMAL whose gate depends
only on the LSTM state.

This is distinct from :class:`~sbc.models.constraint_variants.AsymLaplaceMixtureCorrector`
(``"probnet_alaplace"``), which uses a *two-scale* ``(b_lo, b_hi)`` asymmetric
Laplace; CMAL uses the Klotz/Yu-Moyeed ``(m, s, tau)`` form (the published SOTA).

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

__all__ = ["CMALCorrector", "ald_logpdf", "ald_cdf"]

#: numerical guards
_TAU_EPS = 1e-6      # keep asymmetry strictly inside (0, 1)
_S_EPS = 1e-9        # keep scale strictly positive
_EXP_CLIP = 700.0    # guard against overflow in exp()


# --------------------------------------------------------------------------- #
#  Pure-NumPy asymmetric-Laplacian primitives (Klotz et al., 2022, Eq. B3)     #
#  Used for inference (median / quantiles / variance / sampling) and reporting.#
# --------------------------------------------------------------------------- #
def _two_piece_scales(s, tau) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent two-piece-exponential scales ``(b_lo, b_hi)`` of an ALD.

    The Klotz ALD with scale ``s`` and asymmetry ``tau`` is exactly the
    two-piece exponential with left scale ``b_lo = s / (1 - tau)`` and right
    scale ``b_hi = s / tau`` (so its left/right tails decay at rates
    ``(1-tau)/s`` and ``tau/s``).  This identity gives closed-form moments and a
    clean CDF / inverse-CDF.
    """
    tau = np.clip(np.asarray(tau, float), _TAU_EPS, 1.0 - _TAU_EPS)
    s = np.clip(np.asarray(s, float), _S_EPS, None)
    return s / (1.0 - tau), s / tau


def ald_logpdf(q, m, s, tau) -> np.ndarray:
    """Log-density of a single asymmetric Laplacian (Klotz Eq. B3).

    Parameters
    ----------
    q, m, s, tau : array_like
        Evaluation point, location, scale (``> 0``) and asymmetry
        (``in (0, 1)``); broadcast against one another.

    Returns
    -------
    numpy.ndarray
        ``log ALD(q | m, s, tau)``.
    """
    tau = np.clip(np.asarray(tau, float), _TAU_EPS, 1.0 - _TAU_EPS)
    s = np.clip(np.asarray(s, float), _S_EPS, None)
    u = (np.asarray(q, float) - np.asarray(m, float)) / s
    rho = tau * np.maximum(u, 0.0) + (1.0 - tau) * np.maximum(-u, 0.0)
    return np.log(tau) + np.log1p(-tau) - np.log(s) - rho


def ald_cdf(q, m, s, tau) -> np.ndarray:
    """Closed-form CDF of a single asymmetric Laplacian.

    ``F(m) = tau`` (the location is the ``tau``-quantile); the function is
    continuous and strictly increasing, so a convex mixture of such CDFs is
    monotone and admits well-defined quantiles by bisection.
    """
    tau = np.clip(np.asarray(tau, float), _TAU_EPS, 1.0 - _TAU_EPS)
    s = np.clip(np.asarray(s, float), _S_EPS, None)
    z = (np.asarray(q, float) - np.asarray(m, float)) / s
    left = tau * np.exp(np.clip((1.0 - tau) * z, -_EXP_CLIP, 0.0))
    right = 1.0 - (1.0 - tau) * np.exp(np.clip(-tau * z, -_EXP_CLIP, 0.0))
    return np.where(np.asarray(q, float) < np.asarray(m, float), left, right)


def _ald_icdf(p, m, s, tau) -> np.ndarray:
    """Inverse CDF (quantile function) of a single asymmetric Laplacian."""
    tau = np.clip(np.asarray(tau, float), _TAU_EPS, 1.0 - _TAU_EPS)
    s = np.clip(np.asarray(s, float), _S_EPS, None)
    p = np.clip(np.asarray(p, float), 1e-12, 1.0 - 1e-12)
    m = np.asarray(m, float)
    lower = m + (s / (1.0 - tau)) * np.log(np.clip(p / tau, 1e-12, None))
    upper = m - (s / tau) * np.log(np.clip((1.0 - p) / (1.0 - tau), 1e-12, None))
    return np.where(p < tau, lower, upper)


def _ald_component_mean(m, s, tau) -> np.ndarray:
    """Component mean ``m + (b_hi - b_lo)``."""
    b_lo, b_hi = _two_piece_scales(s, tau)
    return np.asarray(m, float) + (b_hi - b_lo)


def _ald_component_var(s, tau) -> np.ndarray:
    """Component variance ``2 (b_lo^3 + b_hi^3)/(b_lo+b_hi) - (b_hi-b_lo)^2``."""
    b_lo, b_hi = _two_piece_scales(s, tau)
    sp = np.clip(b_lo + b_hi, _S_EPS, None)
    return 2.0 * (b_lo ** 3 + b_hi ** 3) / sp - (b_hi - b_lo) ** 2


def cmal_mixture_quantiles(w, m, s, tau, probs, n_iter: int = 60) -> np.ndarray:
    """Quantiles of a CMAL mixture by vectorised bisection of its CDF.

    The mixture CDF ``F(q) = sum_k w_k F_k(q)`` is monotone increasing, so the
    bisection returns quantiles that are monotone non-decreasing in ``probs`` by
    construction.

    Parameters
    ----------
    w, m, s, tau : array_like, shape (n, K)
        Mixture weights, component locations, scales and asymmetries.
    probs : array_like
        Cumulative probabilities at which to evaluate the quantiles.
    n_iter : int, default 60
        Bisection iterations (each halves the bracket; 60 gives ~1e-18 of the
        initial width).

    Returns
    -------
    numpy.ndarray, shape (n, len(probs))
        Predictive quantiles.
    """
    w = np.asarray(w, float)
    m = np.asarray(m, float)
    s = np.clip(np.asarray(s, float), _S_EPS, None)
    tau = np.clip(np.asarray(tau, float), _TAU_EPS, 1.0 - _TAU_EPS)
    probs = np.atleast_1d(np.asarray(probs, float))
    b_lo, b_hi = _two_piece_scales(s, tau)

    lo = (m - 60.0 * b_lo).min(axis=1)
    hi = (m + 60.0 * b_hi).max(axis=1)
    out = np.empty((m.shape[0], probs.size), float)
    for j, p in enumerate(probs):
        a, b = lo.copy(), hi.copy()
        for _ in range(n_iter):
            mid = 0.5 * (a + b)
            cdf = (w * ald_cdf(mid[:, None], m, s, tau)).sum(axis=1)
            below = cdf < p
            a = np.where(below, mid, a)
            b = np.where(below, b, mid)
        out[:, j] = 0.5 * (a + b)
    return out


# --------------------------------------------------------------------------- #
#  The model                                                                  #
# --------------------------------------------------------------------------- #
@register
class CMALCorrector(RegimeProbNet):
    """EA-LSTM corrector with a Countable-Mixture-of-Asymmetric-Laplacians head.

    A drop-in :class:`~sbc.models.base.BaseCorrector` exposing the full
    probabilistic API (:meth:`predict_residual`, :meth:`predict_quantiles`,
    :meth:`predict_variance`, :meth:`sample`).  Architecturally it reuses the
    flagship's entity-aware LSTM backbone and feature pipeline, but replaces the
    Gaussian mixture-of-experts head with the CMAL head of Klotz et al. (2022):
    ``K`` asymmetric-Laplacian components ``(m_k, s_k, tau_k)`` mixed by softmax
    weights ``pi_k`` and trained on the exact mixture negative log-likelihood.

    Parameters
    ----------
    K : int, default 3
        Number of asymmetric-Laplacian mixture components (Klotz et al. selected
        ``3`` on CAMELS).  When ``regime_gated`` and ``K >= 5`` the first five
        components align to the canonical regimes, as in the flagship.
    regime_gated : bool, default False
        If ``True`` the mixture-weight gate additionally consumes the
        contemporaneous regime forcings and is softly supervised towards the
        rule-based regime label (weight ``lambda_gate``) -- the regime-conditioned
        CMAL variant that mirrors the flagship.  If ``False`` (default) the gate
        depends only on the LSTM hidden state and ``lambda_gate`` is ignored: the
        literature-faithful Klotz CMAL.
    scale_floor : float, default 1e-3
        Lower bound added to every component scale ``s_k`` after softplus.
    lambda_gate : float, default 0.5
        Weight of the regime cross-entropy on the gate (only when
        ``regime_gated``).
    physics : bool, default False
        Master switch for the optional SWE / snowmelt monotonicity penalty
        (``lambda_phys``); off by default to keep the baseline literature-faithful.
    hidden, seq_len, expert_hidden, gate_hidden, epochs, batch_size, lr,
    weight_decay, patience, lambda_phys, lambda_mse, use_gpu, seed, verbose :
        See :class:`~sbc.models.regime_prob_net.RegimeProbNet`.

    Notes
    -----
    The predictive distribution is the CMAL mixture over the log-residual; the
    point correction (:meth:`predict_residual`) is its (numeric) **median**, not
    the mean, which is the natural, robust central estimate for a skewed,
    heavy-tailed mixture.
    """

    name = "cmal"
    is_probabilistic = True

    def __init__(
        self,
        K: int = 3,
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
        regime_gated: bool = False,
        physics: bool = False,
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
        self.regime_gated = bool(regime_gated)

    # -- sklearn-style introspection (so ensembles can clone it) ------------- #
    def get_params(self) -> dict:
        return {
            "K": self.K, "hidden": self.hidden, "seq_len": self.seq_len,
            "expert_hidden": self.expert_hidden, "gate_hidden": self.gate_hidden,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
            "weight_decay": self.weight_decay, "patience": self.patience,
            "lambda_gate": self.lambda_gate, "lambda_phys": self.lambda_phys,
            "lambda_mse": self.lambda_mse, "scale_floor": self.scale_floor,
            "regime_gated": self.regime_gated, "physics": self.physics,
            "use_gpu": self.use_gpu, "seed": self.seed, "verbose": self.verbose,
        }

    # ------------------------------------------------------------------ #
    #  Network: EA-LSTM backbone + CMAL head                              #
    # ------------------------------------------------------------------ #
    def _build_net(self, d_dyn: int, d_stat: int):  # type: ignore[override]
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        scale_floor = self.scale_floor
        regime_gated = self.regime_gated
        K = self.K

        class _EALSTM(nn.Module):
            """Entity-aware LSTM (Kratzert et al., 2019): static -> input gate."""

            def __init__(self, d_dyn: int, d_stat: int, hidden: int):
                super().__init__()
                self.hidden = hidden
                if d_stat > 0:
                    self.input_gate = nn.Linear(d_stat, hidden)
                    self.static_bias = None
                else:  # single-gauge / no static attributes: learn a constant gate
                    self.input_gate = None
                    self.static_bias = nn.Parameter(torch.zeros(hidden))
                self.fgo = nn.Linear(d_dyn + hidden, 3 * hidden)

            def forward(self, x_dyn, x_stat):
                b, length, _ = x_dyn.shape
                if self.input_gate is not None:
                    i = torch.sigmoid(self.input_gate(x_stat))      # time-invariant
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

        class _CMALHead(nn.Module):
            """EA-LSTM backbone + countable asymmetric-Laplacian mixture head."""

            def __init__(self, d_dyn, d_stat, hidden, K, expert_hidden, gate_hidden):
                super().__init__()
                self.encoder = _EALSTM(d_dyn, d_stat, hidden)
                # gate: [hidden | regime forcings] when regime-conditioned, else [hidden]
                gate_in = hidden + d_dyn if regime_gated else hidden
                self.regime_gated = regime_gated
                self.gate = nn.Sequential(
                    nn.Linear(gate_in, gate_hidden), nn.ReLU(),
                    nn.Linear(gate_hidden, K),
                )
                self.body = nn.Sequential(nn.Linear(hidden, expert_hidden), nn.ReLU())
                self.head_m = nn.Linear(expert_hidden, K)            # location
                self.head_log_s = nn.Linear(expert_hidden, K)        # scale  (softplus)
                self.head_tau = nn.Linear(expert_hidden, K)          # asymmetry (sigmoid)
                # spread the component asymmetries across (0, 1) at init so the
                # mixture can represent skew immediately (quantile-regression spirit)
                with torch.no_grad():
                    taus0 = np.linspace(0.3, 0.7, K) if K > 1 else np.array([0.5])
                    self.head_tau.bias.copy_(
                        torch.as_tensor(np.log(taus0 / (1.0 - taus0)), dtype=torch.float32))
                    self.head_tau.weight.mul_(0.1)
                    # start components moderately sharp (softplus(-0.43) ~ 0.5 in
                    # standardised units) so the NLL converges quickly from a
                    # well-conditioned, not over-dispersed, mixture
                    self.head_log_s.bias.fill_(-0.43)
                    self.head_log_s.weight.mul_(0.1)

            def forward(self, x_dyn, x_stat):
                h = self.encoder(x_dyn, x_stat)
                gin = torch.cat([h, x_dyn[:, -1, :]], dim=1) if self.regime_gated else h
                gate_logits = self.gate(gin)
                w = torch.softmax(gate_logits, dim=1)
                z = self.body(h)
                m = self.head_m(z)
                s = F.softplus(self.head_log_s(z)) + scale_floor
                tau = torch.sigmoid(self.head_tau(z))
                tau = tau.clamp(_TAU_EPS, 1.0 - _TAU_EPS)
                return w, gate_logits, m, s, tau

        return _CMALHead(d_dyn, d_stat, self.hidden, K,
                         self.expert_hidden, self.gate_hidden)

    # ------------------------------------------------------------------ #
    #  Fit (CMAL mixture negative log-likelihood, Klotz Eq. B5)           #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "CMALCorrector":
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

        def cmal_nll(y, w, m, s, tau):
            """Mixture negative log-likelihood of a CMAL (Klotz Eq. B5).

            Stable via ``log-sum-exp`` over ``log pi_k + log ALD_k``.
            """
            u = (y[:, None] - m) / s
            rho = tau * F.relu(u) + (1.0 - tau) * F.relu(-u)         # check function
            log_comp = torch.log(tau) + torch.log1p(-tau) - torch.log(s) - rho
            return -torch.logsumexp(torch.log(w + 1e-12) + log_comp, dim=1)

        def _comp_mean(m, s, tau):  # ALD component mean (torch)
            b_lo = s / (1.0 - tau)
            b_hi = s / tau
            return m + (b_hi - b_lo)

        def _prepare(df):
            x_dyn, x_stat = self._design_matrices(df)
            y = (df[TARGET_COL].to_numpy(float) - self.y_mean) / self.y_std
            labels = _classify_regimes(df)
            ridx = np.array([REGIME_TO_IDX.get(s, -1) for s in labels], dtype=np.int64)
            ridx[ridx >= self.K] = -1  # cannot supervise components beyond K
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

        # gate supervision only makes sense for the regime-conditioned variant
        gate_on = self.regime_gated and self.lambda_gate > 0
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

                w, logits, m, scale, tau = self.net(xb, sb)
                loss = cmal_nll(yb, w, m, scale, tau).mean()

                if self.lambda_mse > 0:
                    mean = (w * _comp_mean(m, scale, tau)).sum(1)
                    loss = loss + self.lambda_mse * (mean - yb).pow(2).mean()

                if gate_on:
                    sup = rb >= 0
                    if bool(sup.any()):
                        loss = loss + self.lambda_gate * F.cross_entropy(logits[sup], rb[sup])

                if phys_on and bool(mb.any()):
                    mu_mix = (w * _comp_mean(m, scale, tau)).sum(1)
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
                    w, _, m, scale, tau = self.net(Vd, Vs)
                    valid_metric = float(cmal_nll(vy, w, m, scale, tau).mean())
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
        """Return ``(w, m, s, tau)`` in real log-residual units, per row.

        The ALD is a location-scale family in ``(m, s)`` with ``tau`` invariant,
        so de-standardising the network's output is ``m -> y_mean + y_std m``,
        ``s -> y_std s`` and ``tau`` unchanged.
        """
        if not self._fitted:
            raise RuntimeError("CMALCorrector must be fitted before inference")
        import torch

        x_dyn, x_stat = self._design_matrices(df)
        self.net.eval()
        ws, ms, ss, taus = [], [], [], []
        bs = max(self.batch_size, 1024)
        with torch.no_grad():
            for s in range(0, x_dyn.shape[0], bs):
                xb = torch.as_tensor(x_dyn[s: s + bs], dtype=torch.float32, device=self.device)
                sb = torch.as_tensor(x_stat[s: s + bs], dtype=torch.float32, device=self.device)
                w, _, m, scale, tau = self.net(xb, sb)
                ws.append(w.cpu().numpy())
                ms.append(m.cpu().numpy())
                ss.append(scale.cpu().numpy())
                taus.append(tau.cpu().numpy())
        w = np.concatenate(ws, axis=0)
        m = np.concatenate(ms, axis=0) * self.y_std + self.y_mean
        s = np.clip(np.concatenate(ss, axis=0) * self.y_std, _S_EPS, None)
        tau = np.clip(np.concatenate(taus, axis=0), _TAU_EPS, 1.0 - _TAU_EPS)
        return w, m, s, tau

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Mixture predictive **median** of the log-residual, shape ``(n,)``."""
        w, m, s, tau = self._forward(df)
        return cmal_mixture_quantiles(w, m, s, tau, [0.5])[:, 0]

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        """Mixture predictive mean ``sum_k w_k (m_k + b_hi_k - b_lo_k)``."""
        w, m, s, tau = self._forward(df)
        return (w * _ald_component_mean(m, s, tau)).sum(axis=1)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Total predictive variance (within + between component), shape ``(n,)``."""
        w, m, s, tau = self._forward(df)
        comp_mean = _ald_component_mean(m, s, tau)
        comp_var = _ald_component_var(s, tau)
        mean = (w * comp_mean).sum(axis=1, keepdims=True)
        return (w * (comp_var + comp_mean ** 2)).sum(axis=1) - mean[:, 0] ** 2

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Residual quantiles from the CMAL mixture (monotone), shape ``(n, q)``."""
        w, m, s, tau = self._forward(df)
        return cmal_mixture_quantiles(w, m, s, tau, quantiles)

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Exact inverse-CDF posterior residual samples, shape ``(n_rows, n)``."""
        w, m, s, tau = self._forward(df)
        rng = np.random.default_rng(seed)
        nr, K = m.shape
        cum = np.cumsum(w, axis=1)
        u = rng.random((nr, n))
        comp = (u[:, :, None] > cum[:, None, :]).sum(axis=2)
        comp = np.clip(comp, 0, K - 1)
        rows = np.arange(nr)[:, None]
        m_sel = m[rows, comp]
        s_sel = s[rows, comp]
        tau_sel = tau[rows, comp]
        u2 = rng.random((nr, n))
        return _ald_icdf(u2, m_sel, s_sel, tau_sel)

    def gate_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Learned soft mixture weights, shape ``(n, K)``."""
        w, _, _, _ = self._forward(df)
        return w

    def asymmetry(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row component asymmetries ``tau_k in (0, 1)``, shape ``(n, K)``."""
        _, _, _, tau = self._forward(df)
        return tau


# --------------------------------------------------------------------------- #
#  Self-test: CMAL vs the Gaussian flagship -- CRPS, 90% coverage, monotonicity
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from ..features.engineering import build_features
    from ..features.regimes import classify_regimes
    from ..schemas import OBS_COL, SIM_COL, make_target
    from ..synthetic import generate
    from ..validation.calibration import coverage
    from ..validation.metrics import crps_ensemble, kge_prime
    from ..validation.splits import temporal_split
    from .base import available
    from .regime_prob_net import RegimeProbNet, mixture_crps

    # --- small synthetic table -> features -> regimes ----------------------- #
    df = generate(scale="decadal", years=8, n_basins=3, gauges_per_basin=(2, 3), seed=7)
    df = classify_regimes(build_features(df, scale="decadal"))
    df = validate(df)
    df[TARGET_COL] = make_target(df[OBS_COL].values, df[SIM_COL].values)
    tr_mask, te_mask = temporal_split(df, test_frac=0.3)
    train, test = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    y = test[TARGET_COL].to_numpy(float)
    print(f"[cmal_head] registered 'cmal' = {'cmal' in available()} | "
          f"gauges={df['code'].nunique()} train={len(train)} test={len(test)}")

    # tiny nets; the NLL-trained CMAL gets a few more steps than the
    # CRPS-trained flagship to sharpen (Klotz trains on NLL, not on CRPS)
    tiny = dict(hidden=16, seq_len=4, expert_hidden=16, gate_hidden=16,
                epochs=5, batch_size=512, patience=5, seed=0, verbose=False)

    # --- fit CMAL (vanilla + regime-conditioned) and the Gaussian flagship --- #
    cmal = CMALCorrector(K=3, regime_gated=False, **tiny).fit(train, test)
    cmal_rg = CMALCorrector(K=5, regime_gated=True, lambda_gate=0.5, **tiny).fit(train, test)
    flag = RegimeProbNet(K=3, lambda_gate=0.0, lambda_phys=0.0, physics=False, **tiny).fit(train, test)

    qlev = (0.05, 0.5, 0.95)
    cq = cmal.predict_quantiles(test, qlev)
    rq = cmal_rg.predict_quantiles(test, qlev)
    gq = flag.predict_quantiles(test, qlev)

    cmal_mono = bool(np.all(np.diff(cq, axis=1) >= -1e-9))
    rg_mono = bool(np.all(np.diff(rq, axis=1) >= -1e-9))
    flag_mono = bool(np.all(np.diff(gq, axis=1) >= -1e-9))

    cov_cmal = coverage(y, cq[:, 0], cq[:, 2])
    cov_rg = coverage(y, rq[:, 0], rq[:, 2])
    cov_flag = coverage(y, gq[:, 0], gq[:, 2])

    # CRPS on an identical sampling estimator for a fair head-to-head -------- #
    n_draw = 200
    crps_cmal = crps_ensemble(y, cmal.sample(test, n=n_draw, seed=1))
    crps_rg = crps_ensemble(y, cmal_rg.sample(test, n=n_draw, seed=1))
    crps_flag = crps_ensemble(y, flag.sample(test, n=n_draw, seed=1))
    # closed-form Gaussian-mixture CRPS cross-check for the flagship
    w, mu, sigma = flag._forward(test)
    crps_flag_cf = float(mixture_crps(y, w, mu, sigma).mean())

    var = cmal.predict_variance(test)
    tau = cmal.asymmetry(test)
    gw = cmal_rg.gate_weights(test)
    kge_raw = kge_prime(test[OBS_COL].values, test[SIM_COL].values)["kge"]
    kge_cmal = kge_prime(test[OBS_COL].values, cmal.predict(test))["kge"]

    print(f"[cmal_head] CRPS(log-resid)  CMAL={crps_cmal:.4f} | "
          f"CMAL-regime={crps_rg:.4f} | Gaussian flagship={crps_flag:.4f} "
          f"(closed-form={crps_flag_cf:.4f})")
    print(f"[cmal_head] 90% coverage     CMAL={cov_cmal:.3f} | "
          f"CMAL-regime={cov_rg:.3f} | Gaussian flagship={cov_flag:.3f}  (target ~0.90)")
    print(f"[cmal_head] quantiles monotone CMAL={cmal_mono} regime={rg_mono} flagship={flag_mono} | "
          f"tau in (0,1)={bool(np.all((tau > 0) & (tau < 1)))} | var>=0={bool(np.all(var >= 0))}")
    print(f"[cmal_head] KGE' raw={kge_raw:+.3f} -> CMAL-corrected={kge_cmal:+.3f} | "
          f"regime-gate rowsum~{gw.sum(1).mean():.3f}")

    # --- assertions --------------------------------------------------------- #
    assert cq.shape == (len(test), 3) and cmal_mono, "CMAL quantiles not monotone-shaped"
    assert rq.shape == (len(test), 3) and rg_mono, "regime-CMAL quantiles not monotone"
    assert np.all(np.isfinite(var)) and np.all(var >= 0), "bad CMAL variance"
    assert np.all((tau > 0) & (tau < 1)), "asymmetry tau escaped (0, 1)"
    assert np.isfinite(crps_cmal) and np.isfinite(crps_flag), "non-finite CRPS"
    assert np.isclose(gw.sum(1), 1.0, atol=1e-4).all(), "mixture weights not normalised"
    assert 0.5 <= cov_cmal <= 1.0, f"CMAL 90% coverage out of plausible band: {cov_cmal:.3f}"
    print("[cmal_head] SELF-TEST OK")
