"""Regime-aware probabilistic physically-constrained deep residual corrector.

``RegimeProbNet`` is the flagship model of the *sbc* framework and the central
methodological contribution of the paper.  It fuses four ideas that are usually
studied in isolation into a single, end-to-end-trainable estimator of the
GloFAS log-residual ``log(q_obs+EPS) - log(q_glofas+EPS)``:

1. **Entity-aware LSTM backbone (EA-LSTM).**  Following Kratzert et al. (2019,
   *HESS* 23, doi:10.5194/hess-23-5089-2019), static catchment attributes drive a
   *time-invariant* input gate while the dynamic meteorological sequence flows
   through the forget / cell / output gates.  This lets one shared network learn
   catchment-specific behaviour from attributes -- the regional-modelling
   property that makes LSTMs competitive with calibrated conceptual models.

2. **Regime-gated Mixture-of-Experts head (MoE / MDN).**  A gating network
   (Jacobs et al., 1991; Shazeer et al., 2017) maps the LSTM hidden state plus
   the contemporaneous *regime features* to a softmax distribution over ``K``
   experts.  Each expert is a Gaussian density (Bishop, 1994, mixture-density
   networks) over the log-residual, ``N(mu_k, sigma_k^2)``.  The experts are
   aligned, by default, to the five hydrological regimes of
   ``sbc.features.regimes`` -- *accumulation, melt_freshet, rain_on_snow,
   glacier_melt, recession* -- so the mixture components acquire a physical
   interpretation and the gate weights become a learned, soft regime
   classification that can be inspected (``gate_weights``).

3. **Closed-form probabilistic training.**  The primary loss is the exact CRPS
   of the Gaussian mixture (Grimit et al., 2006; the single-Gaussian special
   case is Gneiting et al., 2005), a strictly proper score that is fully
   differentiable.  A negative-log-likelihood objective is also provided.  When
   a rule-based ``regime`` label is available the gate is *softly supervised*
   with a cross-entropy term (weight ``lambda_gate``); otherwise the gate is
   free.

4. **Soft physical constraints.**  During melt-driven regimes the corrected
   discharge must be non-decreasing in snowmelt (``smlt``) and in available snow
   water equivalent (``swe``).  Because ``back_transform`` is monotone in the
   predicted residual, this reduces to ``d mu / d smlt >= 0`` and
   ``d mu / d swe >= 0``, which we enforce with an input-gradient penalty
   ``relu(-d mu / d x)`` (computed via ``torch.autograd.grad``) on melt-regime
   samples (weight ``lambda_phys``, fully toggleable).

The mixture predictive mean is ``sum_k w_k mu_k``; the predictive variance is the
law-of-total-variance combination of within- and between-expert variance,
``sum_k w_k (sigma_k^2 + mu_k^2) - (sum_k w_k mu_k)^2``.

All heavy imports (``torch``) are deferred into methods so that importing the
package stays light; the model is GPU-optional via ``torch.cuda.is_available()``.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..config import COL_DATE, EPS
from ..schemas import (
    OBS_COL,
    REGIME_COL,
    SIM_COL,
    TARGET_COL,
    dynamic_feature_columns,
    make_target,
    static_feature_columns,
    validate,
)
from ..utils import get_logger
from .base import BaseCorrector, register

log = get_logger(__name__)

#: Canonical regime order; expert ``k`` is aligned to ``REGIME_NAMES[k]`` when
#: ``K >= len(REGIME_NAMES)``.  Mirrors ``sbc.features.regimes``.
REGIME_NAMES: tuple[str, ...] = (
    "accumulation", "melt_freshet", "rain_on_snow", "glacier_melt", "recession",
)
REGIME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(REGIME_NAMES)}
#: Regimes during which the monotonicity constraints are physically expected.
MELT_REGIMES: frozenset[str] = frozenset({"melt_freshet", "glacier_melt", "rain_on_snow"})


# --------------------------------------------------------------------------- #
#  Pure-NumPy helpers (no torch) -- used for reporting, quantiles, sampling.   #
# --------------------------------------------------------------------------- #
def mixture_crps(y, w, mu, sigma) -> np.ndarray:
    """Exact per-sample CRPS of a Gaussian mixture (Grimit et al., 2006).

    Parameters
    ----------
    y : array_like, shape (n,)
        Observations (here, log-residuals).
    w, mu, sigma : array_like, shape (n, K)
        Mixture weights, component means and component standard deviations.

    Returns
    -------
    np.ndarray, shape (n,)
        CRPS for each sample (lower is better).
    """
    from scipy.stats import norm

    y = np.asarray(y, float)
    w = np.asarray(w, float)
    mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-9, None)

    def _a(m, s):
        z = m / s
        return m * (2.0 * norm.cdf(z) - 1.0) + 2.0 * s * norm.pdf(z)

    t1 = (w * _a(y[:, None] - mu, sigma)).sum(axis=1)
    s_ij = np.sqrt(sigma[:, :, None] ** 2 + sigma[:, None, :] ** 2)
    a_ij = _a(mu[:, :, None] - mu[:, None, :], s_ij)
    w_ij = w[:, :, None] * w[:, None, :]
    t2 = 0.5 * (w_ij * a_ij).sum(axis=(1, 2))
    return t1 - t2


def _mixture_quantiles(w, mu, sigma, probs, n_iter: int = 60) -> np.ndarray:
    """Quantiles of a Gaussian mixture by vectorised bisection of its CDF.

    Returns an array of shape ``(n, len(probs))``.
    """
    from scipy.stats import norm

    w = np.asarray(w, float)
    mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-9, None)
    probs = np.atleast_1d(np.asarray(probs, float))

    lo = (mu - 10.0 * sigma).min(axis=1)
    hi = (mu + 10.0 * sigma).max(axis=1)
    out = np.empty((mu.shape[0], probs.size), float)
    for j, p in enumerate(probs):
        a, b = lo.copy(), hi.copy()
        for _ in range(n_iter):
            m = 0.5 * (a + b)
            cdf = (w * norm.cdf((m[:, None] - mu) / sigma)).sum(axis=1)
            right = cdf < p
            a = np.where(right, m, a)
            b = np.where(right, b, m)
        out[:, j] = 0.5 * (a + b)
    return out


def _fallback_regimes(df: pd.DataFrame) -> np.ndarray:
    """Minimal, scale-aware rule-based regime classifier.

    Used only when neither a ``regime`` column nor ``sbc.features.regimes`` is
    available (e.g. in the self-test).  It is intentionally simple; the real
    classifier lives in :mod:`sbc.features.regimes`.
    """
    n = len(df)

    def col(name):
        return df[name].to_numpy(float) if name in df.columns else np.zeros(n)

    smlt = col("smlt")
    swe = col("swe")
    sf = col("sf")
    tp = col("tp")
    t2m = col("t2m_mean")
    glac = col("glacier_frac")
    rain = np.clip(tp - sf, 0.0, None)

    def _pos_thr(x, frac, floor):
        pos = x[x > 0]
        return max(floor, frac * float(pos.mean())) if pos.size else floor

    smlt_thr = _pos_thr(smlt, 0.25, 0.1)
    swe_thr = _pos_thr(swe, 0.05, 1.0)
    rain_thr = _pos_thr(rain, 0.25, 0.5)

    labels = np.full(n, "recession", dtype=object)
    # accumulation: snowfall dominates melt while it is cold
    labels[(sf >= np.maximum(smlt, 1e-6)) & (t2m < 1.0) & ((sf > 0) | (swe > swe_thr))] = "accumulation"
    # melt freshet: appreciable melt with a snowpack present
    labels[(smlt > smlt_thr) & (swe > swe_thr)] = "melt_freshet"
    # rain-on-snow: liquid rain on an existing snowpack (overrides plain melt)
    labels[(rain > rain_thr) & (swe > swe_thr)] = "rain_on_snow"
    # glacier melt: warm, exposed ice once seasonal snow is gone
    labels[(glac > 0.05) & (t2m > 2.0) & (swe < swe_thr)] = "glacier_melt"
    return labels


def _classify_regimes(df: pd.DataFrame) -> np.ndarray:
    """Return a string regime label per row, normalised to ``REGIME_NAMES``.

    Priority: explicit ``regime`` column -> ``sbc.features.regimes`` (best
    effort, API-agnostic) -> internal fallback rule.
    """
    if REGIME_COL in df.columns:
        raw = df[REGIME_COL].astype(str).str.strip().str.lower().to_numpy()
        return np.array([r if r in REGIME_TO_IDX else "recession" for r in raw], dtype=object)

    try:  # opportunistic use of the dedicated module if a sibling agent ships it
        from ..features import regimes as _reg  # type: ignore

        for fn in ("classify", "assign_regimes", "label", "regime_of"):
            f = getattr(_reg, fn, None)
            if callable(f):
                out = f(df)
                vals = out.to_numpy() if isinstance(out, pd.Series) else np.asarray(out)
                vals = np.array([str(v).strip().lower() for v in vals], dtype=object)
                if vals.shape[0] == len(df):
                    return np.array([v if v in REGIME_TO_IDX else "recession" for v in vals],
                                    dtype=object)
    except Exception:  # pragma: no cover - module optional / experimental
        pass

    return _fallback_regimes(df)


# --------------------------------------------------------------------------- #
#  The model                                                                  #
# --------------------------------------------------------------------------- #
@register
class RegimeProbNet(BaseCorrector):
    """Regime-aware probabilistic physically-constrained EA-LSTM corrector.

    Parameters
    ----------
    K : int, default 5
        Number of mixture experts (aligned to :data:`REGIME_NAMES` when >= 5).
    hidden : int, default 64
        EA-LSTM hidden size.
    seq_len : int, default 6
        Length of the dynamic input window (in periods) fed to the LSTM.
    expert_hidden, gate_hidden : int
        Hidden widths of the expert body and gating MLP.
    epochs, batch_size, lr, weight_decay, patience : training schedule.
    lambda_gate : float, default 0.5
        Weight of the soft gate-supervision cross-entropy.
    lambda_phys : float, default 0.1
        Weight of the monotonicity (physics) penalty.
    lambda_mse : float, default 0.0
        Optional weight of an auxiliary MSE on the mixture mean.
    loss : {"crps", "nll"}, default "crps"
        Primary probabilistic objective.
    physics : bool, default True
        Master switch for the physical-constraint penalty.
    use_gpu : bool or None, default None
        ``None`` auto-selects CUDA when available.
    seed : int, default 1234
        Reproducibility seed.
    """

    name = "regimeprobnet"
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
        loss: str = "crps",
        sigma_floor: float = 1e-3,
        physics: bool = True,
        use_gpu: bool | None = None,
        seed: int = 1234,
        verbose: bool = True,
    ) -> None:
        self.K = int(K)
        self.hidden = int(hidden)
        self.seq_len = int(seq_len)
        self.expert_hidden = int(expert_hidden)
        self.gate_hidden = int(gate_hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.lambda_gate = float(lambda_gate)
        self.lambda_phys = float(lambda_phys)
        self.lambda_mse = float(lambda_mse)
        self.loss = str(loss).lower()
        self.sigma_floor = float(sigma_floor)
        self.physics = bool(physics)
        self.use_gpu = use_gpu
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # learned / fitted state
        self.dyn_cols: list[str] = []
        self.stat_cols: list[str] = []
        self.dyn_mean = self.dyn_std = None
        self.stat_mean = self.stat_std = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.smlt_idx: int | None = None
        self.swe_idx: int | None = None
        self.regime_names: list[str] = list(REGIME_NAMES)
        self.net = None
        self.device = "cpu"
        self.history_: dict[str, list[float]] = {"train": [], "valid": []}
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  Feature engineering / scaling / windowing                          #
    # ------------------------------------------------------------------ #
    def _fit_scaler(self, df: pd.DataFrame) -> None:
        self.dyn_cols = dynamic_feature_columns(df)
        self.stat_cols = static_feature_columns(df)
        if not self.dyn_cols:
            raise ValueError("RegimeProbNet needs at least one dynamic feature column")

        dyn = df[self.dyn_cols].to_numpy(float)
        self.dyn_mean = np.nanmean(dyn, axis=0)
        self.dyn_std = np.clip(np.nanstd(dyn, axis=0), 1e-6, None)

        if self.stat_cols:
            stat = df[self.stat_cols].to_numpy(float)
            self.stat_mean = np.nanmean(stat, axis=0)
            self.stat_std = np.clip(np.nanstd(stat, axis=0), 1e-6, None)
        else:
            self.stat_mean = np.zeros(0)
            self.stat_std = np.ones(0)

        y = make_target(df[OBS_COL].to_numpy(float), df[SIM_COL].to_numpy(float)) \
            if TARGET_COL not in df.columns else df[TARGET_COL].to_numpy(float)
        y = y[np.isfinite(y)]
        self.y_mean = float(np.mean(y)) if y.size else 0.0
        self.y_std = float(np.clip(np.std(y), 1e-6, None)) if y.size else 1.0

        self.smlt_idx = self.dyn_cols.index("smlt") if "smlt" in self.dyn_cols else None
        self.swe_idx = self.dyn_cols.index("swe") if "swe" in self.dyn_cols else None

    def _standardise(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        dyn = (df[self.dyn_cols].to_numpy(float) - self.dyn_mean) / self.dyn_std
        dyn = np.where(np.isfinite(dyn), dyn, 0.0)  # NaN -> per-feature mean (== 0)
        if self.stat_cols:
            stat = (df[self.stat_cols].to_numpy(float) - self.stat_mean) / self.stat_std
            stat = np.where(np.isfinite(stat), stat, 0.0)
        else:
            stat = np.zeros((len(df), 0), float)
        return dyn, stat

    def _window_index(self, df: pd.DataFrame) -> np.ndarray:
        """Build, per row, the (seq_len,) window of preceding positional indices."""
        n = len(df)
        dates = df[COL_DATE].to_numpy()
        L = self.seq_len
        win = np.empty((n, L), dtype=np.int64)
        for _, pos in df.groupby("code", sort=False).indices.items():
            pos = np.asarray(pos)
            pos = pos[np.argsort(dates[pos], kind="stable")]
            for t in range(pos.shape[0]):
                idx = pos[max(0, t - L + 1): t + 1]
                if idx.shape[0] < L:  # left-pad short histories by repeating the first row
                    idx = np.concatenate([np.full(L - idx.shape[0], idx[0]), idx])
                win[pos[t]] = idx
        return win

    def _design_matrices(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_dyn[n, L, d_dyn], X_stat[n, d_stat]) aligned to ``df`` rows."""
        dfx = df.reset_index(drop=True)
        dyn, stat = self._standardise(dfx)
        win = self._window_index(dfx)
        x_dyn = dyn[win].astype(np.float32)           # (n, L, d_dyn)
        x_stat = stat.astype(np.float32)              # (n, d_stat)
        return x_dyn, x_stat

    # ------------------------------------------------------------------ #
    #  Network construction (torch deferred to here)                      #
    # ------------------------------------------------------------------ #
    def _build_net(self, d_dyn: int, d_stat: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        sigma_floor = self.sigma_floor

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
                    i = torch.sigmoid(self.input_gate(x_stat))          # (B, H), time-invariant
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

        class _RegimeProbMoE(nn.Module):
            """Regime-gated Gaussian mixture-of-experts over the log-residual."""

            def __init__(self, d_dyn, d_stat, hidden, K, expert_hidden, gate_hidden):
                super().__init__()
                self.encoder = _EALSTM(d_dyn, d_stat, hidden)
                # gate sees [hidden state | contemporaneous regime features]
                self.gate = nn.Sequential(
                    nn.Linear(hidden + d_dyn, gate_hidden), nn.ReLU(),
                    nn.Linear(gate_hidden, K),
                )
                self.body = nn.Sequential(nn.Linear(hidden, expert_hidden), nn.ReLU())
                self.head_mu = nn.Linear(expert_hidden, K)
                self.head_logsig = nn.Linear(expert_hidden, K)

            def forward(self, x_dyn, x_stat):
                h = self.encoder(x_dyn, x_stat)
                gate_logits = self.gate(torch.cat([h, x_dyn[:, -1, :]], dim=1))
                w = torch.softmax(gate_logits, dim=1)
                z = self.body(h)
                mu = self.head_mu(z)
                sigma = F.softplus(self.head_logsig(z)) + sigma_floor
                return w, gate_logits, mu, sigma

        return _RegimeProbMoE(d_dyn, d_stat, self.hidden, self.K,
                              self.expert_hidden, self.gate_hidden)

    # ------------------------------------------------------------------ #
    #  Fit                                                                #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "RegimeProbNet":
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

        sqrt2 = float(np.sqrt(2.0))
        sqrt2pi = float(np.sqrt(2.0 * np.pi))
        inv_sqrt_pi = float(1.0 / np.sqrt(np.pi))

        def _phi(z):
            return torch.exp(-0.5 * z * z) / sqrt2pi

        def _Phi(z):
            return 0.5 * (1.0 + torch.erf(z / sqrt2))

        def _a(m, s):  # E|N(m, s^2)| building block of the mixture CRPS
            z = m / s
            return m * (2.0 * _Phi(z) - 1.0) + 2.0 * s * _phi(z)

        def crps_loss(y, w, mu, sigma):
            y_ = y[:, None]
            t1 = (w * _a(y_ - mu, sigma)).sum(dim=1)
            s_ij = torch.sqrt(sigma[:, :, None] ** 2 + sigma[:, None, :] ** 2 + 1e-12)
            a_ij = _a(mu[:, :, None] - mu[:, None, :], s_ij)
            w_ij = w[:, :, None] * w[:, None, :]
            t2 = 0.5 * (w_ij * a_ij).sum(dim=(1, 2))
            return t1 - t2

        def nll_loss(y, w, mu, sigma):
            y_ = y[:, None]
            log_comp = -0.5 * ((y_ - mu) / sigma) ** 2 - torch.log(sigma) - float(np.log(sqrt2pi))
            return -torch.logsumexp(torch.log(w + 1e-12) + log_comp, dim=1)

        primary = crps_loss if self.loss != "nll" else nll_loss

        def _prepare(df):
            x_dyn, x_stat = self._design_matrices(df)
            y = (df[TARGET_COL].to_numpy(float) - self.y_mean) / self.y_std
            labels = _classify_regimes(df)
            ridx = np.array([REGIME_TO_IDX.get(s, -1) for s in labels], dtype=np.int64)
            ridx[ridx >= self.K] = -1  # cannot supervise experts beyond K
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

                w, logits, mu, sigma = self.net(xb, sb)
                loss = primary(yb, w, mu, sigma).mean()

                if self.lambda_mse > 0:
                    loss = loss + self.lambda_mse * ((w * mu).sum(1) - yb).pow(2).mean()

                if self.lambda_gate > 0:
                    sup = rb >= 0
                    if bool(sup.any()):
                        loss = loss + self.lambda_gate * F.cross_entropy(logits[sup], rb[sup])

                if phys_on and bool(mb.any()):
                    mu_mix = (w * mu).sum(1)
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
                    w, _, mu, sigma = self.net(Vd, Vs)
                    valid_metric = float(crps_loss(vy, w, mu, sigma).mean())
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
                log.info("epoch %3d | train=%.4f | valid_crps=%.4f | best=%.4f",
                         epoch, train_metric, valid_metric, best_metric)
            if bad >= self.patience:
                if self.verbose:
                    log.info("early stop at epoch %d (best valid_crps=%.4f)", epoch, best_metric)
                break

        self.net.load_state_dict(best_state)
        self.net.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    def _forward(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (w, mu, sigma) in real log-residual units, aligned to rows."""
        if not self._fitted:
            raise RuntimeError("RegimeProbNet must be fitted before inference")
        import torch

        x_dyn, x_stat = self._design_matrices(df)
        self.net.eval()
        ws, mus, sigs = [], [], []
        bs = max(self.batch_size, 1024)
        with torch.no_grad():
            for s in range(0, x_dyn.shape[0], bs):
                xb = torch.as_tensor(x_dyn[s: s + bs], dtype=torch.float32, device=self.device)
                sb = torch.as_tensor(x_stat[s: s + bs], dtype=torch.float32, device=self.device)
                w, _, mu, sigma = self.net(xb, sb)
                ws.append(w.cpu().numpy())
                mus.append(mu.cpu().numpy())
                sigs.append(sigma.cpu().numpy())
        w = np.concatenate(ws, axis=0)
        mu = np.concatenate(mus, axis=0) * self.y_std + self.y_mean
        sigma = np.concatenate(sigs, axis=0) * self.y_std
        return w, mu, sigma

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Mixture predictive mean of the log-residual, shape (n,)."""
        w, mu, _ = self._forward(df)
        return (w * mu).sum(axis=1)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Total predictive variance (within + between expert), shape (n,)."""
        w, mu, sigma = self._forward(df)
        mean = (w * mu).sum(axis=1, keepdims=True)
        return (w * (sigma ** 2 + mu ** 2)).sum(axis=1) - mean[:, 0] ** 2

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Residual quantiles from the Gaussian mixture, shape (n, n_quantiles)."""
        w, mu, sigma = self._forward(df)
        return _mixture_quantiles(w, mu, sigma, quantiles)

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Draw posterior residual samples, shape (n_rows, n)."""
        w, mu, sigma = self._forward(df)
        rng = np.random.default_rng(seed)
        nr, K = mu.shape
        cum = np.cumsum(w, axis=1)
        u = rng.random((nr, n))
        comp = (u[:, :, None] > cum[:, None, :]).sum(axis=2)
        comp = np.clip(comp, 0, K - 1)
        rows = np.arange(nr)[:, None]
        mu_sel = mu[rows, comp]
        sig_sel = sigma[rows, comp]
        return mu_sel + sig_sel * rng.standard_normal((nr, n))

    def gate_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Learned soft regime / expert weights, shape (n, K).

        Column ``k`` corresponds to expert ``k``; for ``k < len(REGIME_NAMES)``
        the expert is aligned to ``REGIME_NAMES[k]``.
        """
        w, _, _ = self._forward(df)
        return w


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from sbc.synthetic import generate
    from sbc.validation.metrics import kge_prime

    df = generate(scale="decadal", years=12, n_basins=3,
                  gauges_per_basin=(2, 3), seed=7)
    df[TARGET_COL] = make_target(df[OBS_COL].values, df[SIM_COL].values)

    cut = df["date"].quantile(0.7)
    train = df[df["date"] <= cut].copy()
    test = df[df["date"] > cut].copy()
    print(f"[regime_prob_net] gauges={df['code'].nunique()} "
          f"train_rows={len(train)} test_rows={len(test)}")

    model = RegimeProbNet(K=3, hidden=32, seq_len=5, expert_hidden=24, gate_hidden=24,
                          epochs=20, batch_size=512, patience=8,
                          lambda_gate=0.5, lambda_phys=0.1, seed=0, verbose=True)
    model.fit(train, valid=test)

    q_corr = model.predict(test)
    kge_raw = kge_prime(test[OBS_COL].values, test[SIM_COL].values)["kge"]
    kge_cor = kge_prime(test[OBS_COL].values, q_corr)["kge"]

    w, mu, sigma = model._forward(test)
    crps = float(mixture_crps(test[TARGET_COL].values, w, mu, sigma).mean())

    qs = model.predict_quantiles(test, (0.05, 0.5, 0.95))
    monotone = bool(np.all(np.diff(qs, axis=1) >= -1e-6))
    draws = model.sample(test, n=50, seed=1)
    gw = model.gate_weights(test)

    print(f"[regime_prob_net] KGE' raw={kge_raw:+.3f} -> corrected={kge_cor:+.3f} | "
          f"mean CRPS(log-resid)={crps:.4f} | quantiles{qs.shape} monotone={monotone} | "
          f"samples{draws.shape} | gate{gw.shape} rowsum~{gw.sum(1).mean():.3f}")
    assert qs.shape == (len(test), 3) and monotone, "quantiles not calibrated-shaped"
    print("[regime_prob_net] SELF-TEST OK")
