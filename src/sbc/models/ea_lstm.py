"""Entity-Aware LSTM residual corrector — the deep-learning baseline.

This module implements the *Entity-Aware LSTM* (EA-LSTM) of Kratzert et al.
(2019, *Hydrol. Earth Syst. Sci.* 23:5089, doi:10.5194/hess-23-5089-2019) as a
single **regional** bias-correction model that is conditioned on each gauge by
its static catchment attributes.

The EA-LSTM departs from a vanilla LSTM in exactly one place: the *input gate*
is computed **once per sequence from the static features only** and is held
constant across time,

    i = sigmoid(W_s s + b_s)                              (entity-aware gate)

while the forget gate, cell candidate and output gate evolve with the dynamic
forcing as usual,

    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
    g_t = tanh   (W_g x_t + U_g h_{t-1} + b_g)
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
    c_t = f_t ⊙ c_{t-1} + i ⊙ g_t
    h_t = o_t ⊙ tanh(c_t).

Intuitively the static attributes learn *which* dynamic features matter for a
given catchment, letting one shared set of recurrent weights serve every gauge.
A linear head maps the final hidden state to the **log-space residual**
``log(q_obs+EPS) - log(q_glofas+EPS)``; :func:`sbc.schemas.back_transform` then
reconstructs the bias-corrected discharge.

Data handling follows the project conventions: dynamic features are turned into
sliding windows of length ``L`` (default 12 for decadal, 90 for daily), the
static vector is taken per gauge from :func:`sbc.schemas.static_feature_columns`,
and **all** standardisation statistics are estimated on the training split only.
Rows that lack a full history are handled by left edge-padding so every table
row receives a prediction; the recurrence still focuses on the most recent
steps, and a row with no informative history simply approaches a near-zero
residual (i.e. falls back to raw GloFAS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schemas import (
    SIM_COL,
    TARGET_COL,
    WEIGHT_COL,
    dynamic_feature_columns,
    static_feature_columns,
    validate,
)
from ..utils import get_logger
from .base import BaseCorrector, register

log = get_logger(__name__)

# Default sliding-window length per temporal scale (timesteps of look-back).
_DEFAULT_SEQ_LENGTH = {"decadal": 12, "daily": 90}
# Hard clip on the predicted log-residual to keep ``exp`` well-behaved.
_RESIDUAL_CLIP = 15.0

# Lazily-built torch ``nn.Module`` class (keeps torch out of the import path).
_NET_CLASS = None


def _net_class():
    """Build (once) and return the EA-LSTM ``nn.Module`` class.

    Defined inside a function so that importing :mod:`sbc.models.ea_lstm` does
    not import torch; the class is cached in a module global.
    """
    global _NET_CLASS
    if _NET_CLASS is not None:
        return _NET_CLASS

    import torch
    from torch import nn

    class _EALSTMNet(nn.Module):
        """One-layer Entity-Aware LSTM with a linear residual head."""

        def __init__(self, dynamic_size: int, static_size: int,
                     hidden_size: int = 64, dropout: float = 0.4) -> None:
            super().__init__()
            self.dynamic_size = int(dynamic_size)
            self.static_size = int(static_size)
            self.hidden_size = int(hidden_size)
            # Static-only input gate (constant over the sequence).
            self.input_gate = nn.Linear(self.static_size, self.hidden_size)
            # Dynamic + recurrent weights for [forget, candidate, output].
            self.w_xh = nn.Linear(self.dynamic_size, 3 * self.hidden_size, bias=True)
            self.w_hh = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(self.hidden_size, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.w_xh.weight)
            nn.init.orthogonal_(self.w_hh.weight)
            nn.init.zeros_(self.w_xh.bias)
            if self.static_size > 0:
                nn.init.xavier_uniform_(self.input_gate.weight)
            nn.init.zeros_(self.input_gate.bias)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            # Positive forget-gate bias encourages long-term memory early on.
            with torch.no_grad():
                self.w_xh.bias[: self.hidden_size].fill_(1.0)

        def forward(self, x_dyn: "torch.Tensor", x_stat: "torch.Tensor") -> "torch.Tensor":
            """Map (dynamic window, static vector) to a scalar residual.

            Parameters
            ----------
            x_dyn : torch.Tensor
                Dynamic features, shape ``(batch, seq_len, dynamic_size)``.
            x_stat : torch.Tensor
                Static features, shape ``(batch, static_size)``.

            Returns
            -------
            torch.Tensor
                Predicted (standardised) residual, shape ``(batch,)``.
            """
            batch, seq_len, _ = x_dyn.shape
            h = x_dyn.new_zeros(batch, self.hidden_size)
            c = x_dyn.new_zeros(batch, self.hidden_size)
            i = torch.sigmoid(self.input_gate(x_stat))  # (batch, hidden), constant
            for t in range(seq_len):
                gates = self.w_xh(x_dyn[:, t, :]) + self.w_hh(h)
                f, g, o = gates.chunk(3, dim=1)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c = f * c + i * g
                h = o * torch.tanh(c)
            return self.head(self.dropout(h)).squeeze(-1)

    _NET_CLASS = _EALSTMNet
    return _NET_CLASS


# --------------------------------------------------------------------------- #
#  Feature scaling                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class _Scaler:
    """Per-column standardiser fitted on the training split only."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))

    @classmethod
    def fit(cls, x: np.ndarray, floor: float = 1e-6) -> "_Scaler":
        x = np.asarray(x, float)
        if x.size == 0 or x.shape[0] == 0:
            return cls(np.zeros(x.shape[1] if x.ndim == 2 else 0),
                       np.ones(x.shape[1] if x.ndim == 2 else 0))
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(np.isfinite(std) & (std > floor), std, 1.0)
        return cls(np.nan_to_num(mean), std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.shape[-1] == 0:
            return x.astype(np.float32)
        out = (x - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _build_windows(df: pd.DataFrame, dyn_cols: list[str], stat_cols: list[str],
                   seq_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct per-row sliding windows aligned to ``df`` row positions.

    For every row a look-back window of ``seq_length`` dynamic-feature vectors
    ending at that row is built, left edge-padded where history is short so the
    output is one window per row.

    Returns
    -------
    x_dyn : np.ndarray
        ``(n_rows, seq_length, n_dynamic)`` raw (unscaled) dynamic windows.
    x_stat : np.ndarray
        ``(n_rows, n_static)`` raw (unscaled) static vectors.
    pos : np.ndarray
        Positional indices into ``df.reset_index(drop=True)`` for each window,
        so predictions can be scattered back to the original rows.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    df = df.reset_index(drop=True)
    n_dyn = len(dyn_cols)
    dyn_parts: list[np.ndarray] = []
    stat_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []

    for _, g in df.groupby("code", sort=False):
        g = g.sort_values("date")
        order = g.index.to_numpy()
        n = len(g)
        if n == 0:
            continue
        x = g[dyn_cols].to_numpy(np.float32) if n_dyn else np.zeros((n, 0), np.float32)
        if stat_cols:
            s = g[stat_cols].iloc[0].to_numpy(np.float32)
        else:
            s = np.zeros((0,), np.float32)

        if n_dyn:
            pad = np.repeat(x[:1], seq_length - 1, axis=0)
            xp = np.concatenate([pad, x], axis=0)                  # (n+L-1, D)
            win = sliding_window_view(xp, seq_length, axis=0)      # (n, D, L)
            win = np.transpose(win, (0, 2, 1)).astype(np.float32)  # (n, L, D)
        else:
            win = np.zeros((n, seq_length, 0), np.float32)

        dyn_parts.append(win)
        stat_parts.append(np.repeat(s[None, :], n, axis=0))
        pos_parts.append(order)

    if not dyn_parts:
        return (np.zeros((0, seq_length, n_dyn), np.float32),
                np.zeros((0, len(stat_cols)), np.float32),
                np.zeros((0,), int))

    return (np.concatenate(dyn_parts, axis=0),
            np.concatenate(stat_parts, axis=0),
            np.concatenate(pos_parts, axis=0))


@register
class EALSTMCorrector(BaseCorrector):
    """Entity-Aware LSTM regional residual corrector.

    Parameters
    ----------
    seq_length : int, optional
        Look-back window length.  ``None`` selects 12 (decadal) or 90 (daily)
        from the training table's ``scale`` column.
    hidden_size : int
        LSTM hidden state dimensionality.
    dropout : float
        Dropout applied to the final hidden state before the head.
    max_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate, weight_decay : float
        Adam optimiser settings.
    patience : int
        Early-stopping patience (epochs without validation improvement).
    valid_fraction : float
        Tail fraction (by date) held out from ``train`` for early stopping when
        no explicit validation table is given to :meth:`fit`.
    seed : int
        RNG seed for reproducibility.
    device : str, optional
        ``"cuda"`` / ``"cpu"``; defaults to GPU when available.
    """

    name = "ealstm"
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

        # Learned state (populated by :meth:`fit`).
        self.dynamic_cols_: list[str] = []
        self.static_cols_: list[str] = []
        self.seq_length_: int = 0
        self.dyn_scaler_: _Scaler | None = None
        self.stat_scaler_: _Scaler | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.net_ = None
        self.history_: list[dict[str, float]] = []

    # -- helpers ------------------------------------------------------------ #
    def _resolve_device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_seq_length(self, df: pd.DataFrame) -> int:
        if self.seq_length is not None:
            return int(self.seq_length)
        scale = str(df["scale"].iloc[0]) if "scale" in df.columns and len(df) else "decadal"
        return _DEFAULT_SEQ_LENGTH.get(scale, 12)

    def _time_split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Hold out the latest ``valid_fraction`` of dates for early stopping."""
        if self.valid_fraction <= 0 or len(train) < 10:
            return train, train.iloc[0:0]
        cutoff = train["date"].quantile(1.0 - self.valid_fraction)
        tr = train[train["date"] <= cutoff]
        va = train[train["date"] > cutoff]
        if len(va) < 1 or len(tr) < 1:
            return train, train.iloc[0:0]
        return tr, va

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build and scale windows for ``df`` (uses fitted scalers/columns)."""
        work = df.copy()
        for c in self.dynamic_cols_ + self.static_cols_:
            if c not in work.columns:
                work[c] = np.nan
        x_dyn, x_stat, pos = _build_windows(
            work, self.dynamic_cols_, self.static_cols_, self.seq_length_)
        if x_dyn.shape[0]:
            x_dyn = self.dyn_scaler_.transform(
                x_dyn.reshape(-1, len(self.dynamic_cols_))
            ).reshape(x_dyn.shape) if self.dynamic_cols_ else x_dyn
            x_stat = self.stat_scaler_.transform(x_stat)
        return x_dyn, x_stat, pos

    # -- training ----------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "EALSTMCorrector":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        train = validate(train)
        self.dynamic_cols_ = dynamic_feature_columns(train)
        self.static_cols_ = static_feature_columns(train)
        if not self.dynamic_cols_:
            raise ValueError("EA-LSTM requires at least one dynamic feature column.")
        self.seq_length_ = self._resolve_seq_length(train)

        if valid is None:
            tr_df, va_df = self._time_split(train)
        else:
            tr_df, va_df = train, validate(valid)

        # Scalers from the *training* split only.
        self.dyn_scaler_ = _Scaler.fit(tr_df[self.dynamic_cols_].to_numpy(float))
        stat_per_gauge = (tr_df.groupby("code")[self.static_cols_].first().to_numpy(float)
                          if self.static_cols_ else np.zeros((tr_df["code"].nunique(), 0)))
        self.stat_scaler_ = _Scaler.fit(stat_per_gauge)
        y_tr_raw = tr_df[TARGET_COL].to_numpy(float)
        self.y_mean_ = float(np.nanmean(y_tr_raw)) if y_tr_raw.size else 0.0
        ystd = float(np.nanstd(y_tr_raw)) if y_tr_raw.size else 1.0
        self.y_std_ = ystd if np.isfinite(ystd) and ystd > 1e-6 else 1.0

        seed_gen = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._resolve_device()

        x_dyn, x_stat, pos = self._prepare(tr_df)
        y = (tr_df[TARGET_COL].to_numpy(float)[pos] - self.y_mean_) / self.y_std_
        if WEIGHT_COL in tr_df.columns:
            w = tr_df[WEIGHT_COL].to_numpy(float)[pos]
        else:
            w = np.ones(len(pos), float)
        ds = TensorDataset(
            torch.as_tensor(x_dyn, dtype=torch.float32),
            torch.as_tensor(x_stat, dtype=torch.float32),
            torch.as_tensor(np.nan_to_num(y), dtype=torch.float32),
            torch.as_tensor(np.nan_to_num(w, nan=0.0), dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=min(self.batch_size, max(1, len(ds))),
                            shuffle=True, generator=seed_gen, drop_last=False)

        net = _net_class()(len(self.dynamic_cols_), len(self.static_cols_),
                           self.hidden_size, self.dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        # Optional validation tensors for early stopping.
        has_valid = len(va_df) > 0
        if has_valid:
            vx_dyn, vx_stat, vpos = self._prepare(va_df)
            vy = (va_df[TARGET_COL].to_numpy(float)[vpos] - self.y_mean_) / self.y_std_
            vx_dyn_t = torch.as_tensor(vx_dyn, dtype=torch.float32, device=device)
            vx_stat_t = torch.as_tensor(vx_stat, dtype=torch.float32, device=device)
            vy_t = torch.as_tensor(np.nan_to_num(vy), dtype=torch.float32, device=device)

        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        best_loss = float("inf")
        bad_epochs = 0
        self.history_ = []

        for epoch in range(self.max_epochs):
            net.train()
            running, seen = 0.0, 0
            for xb, sb, yb, wb in loader:
                xb, sb, yb, wb = (xb.to(device), sb.to(device),
                                  yb.to(device), wb.to(device))
                opt.zero_grad()
                pred = net(xb, sb)
                se = wb * (pred - yb) ** 2
                wsum = wb.sum()
                loss = se.sum() / wsum if float(wsum) > 0 else se.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                running += float(loss) * len(yb)
                seen += len(yb)
            train_loss = running / max(seen, 1)

            if has_valid:
                net.eval()
                with torch.no_grad():
                    vpred = net(vx_dyn_t, vx_stat_t)
                    val_loss = float(nn.functional.mse_loss(vpred, vy_t))
            else:
                val_loss = train_loss

            self.history_.append({"epoch": epoch, "train_loss": train_loss,
                                  "val_loss": val_loss})
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    log.info("Early stopping at epoch %d (best val_loss=%.4g)",
                             epoch, best_loss)
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        log.info("EA-LSTM fitted: L=%d hidden=%d dyn=%d stat=%d epochs=%d device=%s",
                 self.seq_length_, self.hidden_size, len(self.dynamic_cols_),
                 len(self.static_cols_), len(self.history_), device.type)
        return self

    # -- inference ---------------------------------------------------------- #
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-space residual aligned 1:1 to ``df`` rows."""
        if self.net_ is None:
            raise RuntimeError("EALSTMCorrector.predict_residual called before fit().")
        import torch

        device = self._resolve_device()
        out = np.zeros(len(df), float)  # default 0 -> raw GloFAS fallback
        if len(df) == 0:
            return out

        x_dyn, x_stat, pos = self._prepare(df)
        if x_dyn.shape[0] == 0:
            return out

        self.net_.eval()
        preds = np.empty(len(pos), np.float32)
        bs = max(1, self.batch_size)
        with torch.no_grad():
            for start in range(0, len(pos), bs):
                sl = slice(start, start + bs)
                xb = torch.as_tensor(x_dyn[sl], dtype=torch.float32, device=device)
                sb = torch.as_tensor(x_stat[sl], dtype=torch.float32, device=device)
                preds[sl] = self.net_(xb, sb).cpu().numpy()

        resid = preds.astype(float) * self.y_std_ + self.y_mean_
        resid = np.clip(resid, -_RESIDUAL_CLIP, _RESIDUAL_CLIP)
        out[pos] = resid
        return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sbc.synthetic import generate
    from sbc.validation.metrics import kge_prime

    table = generate(n_basins=2, gauges_per_basin=(2, 3), years=12,
                     scale="decadal", seed=0)
    table = validate(table)
    cutoff = table["date"].quantile(0.8)
    train = table[table["date"] <= cutoff].copy()
    test = table[table["date"] > cutoff].copy()

    model = EALSTMCorrector(seq_length=6, hidden_size=16, max_epochs=8,
                            batch_size=128, patience=4, seed=0)
    model.fit(train)

    q_corr = model.predict(test)
    kge_raw = kge_prime(test["q_obs"].values, test["q_glofas"].values)["kge"]
    kge_corr = kge_prime(test["q_obs"].values, q_corr)["kge"]
    resid = model.predict_residual(test)

    print(f"[ea_lstm] gauges={table['code'].nunique()} "
          f"train={len(train)} test={len(test)} "
          f"resid_aligned={len(resid) == len(test)} "
          f"KGE' raw={kge_raw:.3f} -> corrected={kge_corr:.3f}")
