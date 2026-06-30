"""Transfer-learning bias-correction baseline (pretrain-then-fine-tune).

The 2025-26 large-sample hydrology paradigm — *pretrain a single
large-sample / global model, then fine-tune it locally* — is now the reference
recipe a reviewer will demand for any prediction-in-ungauged-regions (PUR)
claim, and in particular for the data-sparse Amu-Darya transfer track of this
study.  The intuition is that a model trained across many catchments learns a
*transferable* hydrological prior (how snow storage, melt timing and the GloFAS
volume bias co-vary with catchment attributes), and that only a handful of local
observations are then needed to close most of the residual gap at a new gauge.

:class:`TransferLSTMCorrector` (registry name ``"transfer_lstm"``) realises this
recipe on top of the project's deep baseline,
:class:`~sbc.models.ea_lstm.EALSTMCorrector`:

* :meth:`fit` **pretrains** an EA-LSTM on the *full* training pool (e.g. the core
  Syr-Darya / Chu / Talas domain).  The freshly fitted corrector is also the
  *zero-shot* transfer model — it can correct GloFAS at an unseen gauge using
  only the shared, attribute-conditioned prior.
* :meth:`fine_tune` **adapts** the pretrained weights to a target basin / gauge,
  either with *all* available target data (full fine-tune) or with a small
  ``n_shot`` sample (few-shot).  It is leakage-safe by construction: the
  pretrained feature scalers are *frozen* (never re-estimated on one sample), the
  few-shot rows are the chronologically *earliest* per gauge, and the method
  returns a **new** corrector so the pretrained model stays reusable for the next
  shot count.
* :meth:`transfer_curve` sweeps a list of shot counts (``0 / 1 / 3 / 5`` by
  default) and reports KGE' versus ``n_shot`` — the *transfer curve* that
  quantifies exactly how much target data is needed to close the PUR gap.

Pretraining here uses the in-study core domain.  The natural — and entirely
optional — extension is to pretrain on the external **Caravan** global
large-sample dataset (Kratzert et al., 2023, *Sci. Data* 10:61; Zenodo,
doi:10.5281/zenodo.7944025) and fine-tune on the Central-Asian gauges; because
Caravan shares the same dynamic-forcing / static-attribute schema this corrector
consumes, that swap requires no code change beyond passing a Caravan-derived
modelling table to :meth:`fit`.

The corrector keeps the plain :class:`~sbc.models.base.BaseCorrector` API
(:meth:`predict_residual` is aligned 1:1 to the input rows) and is fully
deterministic for a fixed seed, so it slots into the existing validation,
explainability and ensemble machinery unchanged.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..schemas import OBS_COL, TARGET_COL, WEIGHT_COL, validate
from ..utils import get_logger
from .base import BaseCorrector, register
from .ea_lstm import EALSTMCorrector

log = get_logger(__name__)

#: default number of fine-tuning epochs (kept short — fine-tuning needs far
#: fewer passes than training from scratch).
_DEFAULT_FT_EPOCHS = 40
#: default fine-tuning learning rate (an order of magnitude below the pretrain
#: rate so the adaptation nudges, rather than overwrites, the prior).
_DEFAULT_FT_LR = 1e-4


@register
class TransferLSTMCorrector(BaseCorrector):
    """Pretrain-then-fine-tune EA-LSTM transfer-learning corrector.

    The corrector wraps :class:`~sbc.models.ea_lstm.EALSTMCorrector`.
    :meth:`fit` pretrains one EA-LSTM on the full training pool; the resulting
    model is the *zero-shot* transfer corrector.  :meth:`fine_tune` warm-starts a
    deep copy of those pretrained weights and adapts them to a target basin /
    gauge with full or few-shot target data, returning a fresh, independently
    usable corrector.

    All constructor arguments up to ``device`` mirror :class:`EALSTMCorrector`
    and configure the *pretraining* network; the ``ft_*`` arguments configure the
    *fine-tuning* phase.

    Parameters
    ----------
    seq_length, hidden_size, dropout, max_epochs, batch_size, learning_rate,
    weight_decay, patience, valid_fraction, seed, device :
        Forwarded verbatim to the wrapped :class:`EALSTMCorrector` for
        pretraining (see that class for semantics).
    ft_epochs : int, default 40
        Number of fine-tuning epochs applied to the target sample.
    ft_learning_rate : float, default 1e-4
        Adam learning rate during fine-tuning (typically << ``learning_rate``).
    freeze_body : bool, default False
        When ``True`` the shared recurrent dynamics (the ``x->h`` and ``h->h``
        weights) are frozen during fine-tuning and only the entity-aware static
        input gate and the linear residual head adapt.  This is the robust choice
        for very-few-shot adaptation, where fine-tuning the whole network risks
        over-fitting the handful of target rows.

    Attributes
    ----------
    pretrained_ : EALSTMCorrector
        The EA-LSTM fitted on the full pool by :meth:`fit` (the zero-shot model).
    model_ : EALSTMCorrector
        The corrector currently backing :meth:`predict_residual`.  Equals
        ``pretrained_`` after :meth:`fit`; on a child returned by
        :meth:`fine_tune` it is the adapted copy.
    n_shot_ : int or None
        Shot count used to produce ``model_`` (``None`` = pretrained / full).
    n_finetune_rows_ : int
        Number of target rows actually used to fine-tune ``model_``.
    adapted_ : bool
        Whether ``model_`` was fine-tuned (``False`` for the zero-shot model).
    """

    name = "transfer_lstm"
    is_probabilistic = False

    def __init__(self, seq_length: int | None = None, hidden_size: int = 64,
                 dropout: float = 0.4, max_epochs: int = 100, batch_size: int = 256,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-6,
                 patience: int = 15, valid_fraction: float = 0.2,
                 ft_epochs: int = _DEFAULT_FT_EPOCHS,
                 ft_learning_rate: float = _DEFAULT_FT_LR,
                 freeze_body: bool = False,
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
        self.ft_epochs = int(ft_epochs)
        self.ft_learning_rate = float(ft_learning_rate)
        self.freeze_body = bool(freeze_body)
        self.seed = int(seed)
        self.device = device

        # learned / adaptation state
        self.pretrained_: EALSTMCorrector | None = None
        self.model_: EALSTMCorrector | None = None
        self.n_shot_: int | None = None
        self.n_finetune_rows_: int = 0
        self.adapted_: bool = False
        self._fitted = False

    # -- sklearn-style introspection (lets the ensemble clone this base) ------ #
    def get_params(self) -> dict:
        """Constructor kwargs (so :class:`StackedEnsemble` can clone us)."""
        return {"seq_length": self.seq_length, "hidden_size": self.hidden_size,
                "dropout": self.dropout, "max_epochs": self.max_epochs,
                "batch_size": self.batch_size, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "patience": self.patience,
                "valid_fraction": self.valid_fraction, "ft_epochs": self.ft_epochs,
                "ft_learning_rate": self.ft_learning_rate,
                "freeze_body": self.freeze_body, "seed": self.seed,
                "device": self.device}

    # -- construction helpers ------------------------------------------------- #
    def _make_base(self) -> EALSTMCorrector:
        """Build a fresh, unfitted EA-LSTM with the pretraining hyper-parameters."""
        return EALSTMCorrector(
            seq_length=self.seq_length, hidden_size=self.hidden_size,
            dropout=self.dropout, max_epochs=self.max_epochs,
            batch_size=self.batch_size, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, patience=self.patience,
            valid_fraction=self.valid_fraction, seed=self.seed, device=self.device)

    # -- pretraining ---------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "TransferLSTMCorrector":
        """Pretrain the EA-LSTM on the full training pool (the zero-shot model).

        Parameters
        ----------
        train : pandas.DataFrame
            Pretraining modelling table (e.g. the core domain, or an external
            Caravan large-sample table sharing the same schema).
        valid : pandas.DataFrame, optional
            Explicit validation table for early stopping; when omitted the
            wrapped EA-LSTM holds out a temporal tail of ``train``.

        Returns
        -------
        TransferLSTMCorrector
            ``self``, with :attr:`pretrained_` populated and :attr:`model_` set to
            the zero-shot (pretrained) corrector.
        """
        train = validate(train)
        pre = self._make_base()
        pre.fit(train, valid)

        self.pretrained_ = pre
        self.model_ = pre
        self.n_shot_ = None
        self.n_finetune_rows_ = 0
        self.adapted_ = False
        self._fitted = True
        log.info("transfer_lstm: pretrained on %d rows / %d gauges (zero-shot ready)",
                 len(train), train["code"].nunique())
        return self

    # -- few-shot sampling ---------------------------------------------------- #
    @staticmethod
    def _few_shot_sample(df: pd.DataFrame, n_shot: int) -> pd.DataFrame:
        """Earliest ``n_shot`` rows per gauge (chronological, leakage-safe).

        Sampling the *earliest* records per gauge mimics a deployment in which a
        new gauge has just started reporting: adaptation uses the first few
        observations and skill is assessed on the (strictly later) remainder.
        """
        if n_shot <= 0:
            return df.iloc[:0]
        parts = [g.sort_values("date").iloc[:n_shot]
                 for _, g in df.groupby("code", sort=True)]
        if not parts:
            return df.iloc[:0]
        return pd.concat(parts, axis=0)

    # -- weight adaptation ---------------------------------------------------- #
    def _adapt(self, ft: EALSTMCorrector, shot_df: pd.DataFrame,
               ft_epochs: int, ft_seed: int) -> EALSTMCorrector:
        """Fine-tune the (already pretrained) ``ft`` network on ``shot_df`` in place.

        Reuses the pretrained feature scalers, column layout and target
        normalisation carried on ``ft`` (so a single target row cannot collapse
        the standardisation), and warm-starts from the pretrained weights.
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        shot_df = validate(shot_df)
        x_dyn, x_stat, pos = ft._prepare(shot_df)
        if x_dyn.shape[0] == 0:
            return ft  # nothing usable to adapt on -> stays zero-shot

        y = (shot_df[TARGET_COL].to_numpy(float)[pos] - ft.y_mean_) / ft.y_std_
        if WEIGHT_COL in shot_df.columns:
            w = shot_df[WEIGHT_COL].to_numpy(float)[pos]
        else:
            w = np.ones(len(pos), float)

        device = ft._resolve_device()
        torch.manual_seed(ft_seed)
        np.random.seed(ft_seed)
        net = ft.net_.to(device)

        # Select the parameters that adapt; freeze the shared recurrent body for
        # robust very-few-shot transfer when requested.
        if self.freeze_body:
            for p in net.parameters():
                p.requires_grad_(False)
            for mod in (net.input_gate, net.head):
                for p in mod.parameters():
                    p.requires_grad_(True)
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = list(net.parameters())

        opt = torch.optim.Adam(params, lr=self.ft_learning_rate,
                               weight_decay=ft.weight_decay)
        ds = TensorDataset(
            torch.as_tensor(x_dyn, dtype=torch.float32),
            torch.as_tensor(x_stat, dtype=torch.float32),
            torch.as_tensor(np.nan_to_num(y), dtype=torch.float32),
            torch.as_tensor(np.nan_to_num(w, nan=0.0), dtype=torch.float32),
        )
        gen = torch.Generator().manual_seed(ft_seed)
        loader = DataLoader(ds, batch_size=min(ft.batch_size, max(1, len(ds))),
                            shuffle=True, generator=gen, drop_last=False)

        net.train()
        for _ in range(int(ft_epochs)):
            for xb, sb, yb, wb in loader:
                xb, sb = xb.to(device), sb.to(device)
                yb, wb = yb.to(device), wb.to(device)
                opt.zero_grad()
                pred = net(xb, sb)
                se = wb * (pred - yb) ** 2
                wsum = wb.sum()
                loss = se.sum() / wsum if float(wsum) > 0 else se.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
        net.eval()
        for p in net.parameters():  # restore a clean grad state
            p.requires_grad_(True)
        ft.net_ = net
        return ft

    def _spawn_child(self, model: EALSTMCorrector, n_shot: int | None,
                     n_rows: int, adapted: bool) -> "TransferLSTMCorrector":
        """Wrap an adapted EA-LSTM in a fresh corrector sharing this prior."""
        child = TransferLSTMCorrector(**self.get_params())
        child.pretrained_ = self.pretrained_  # shared, immutable prior
        child.model_ = model
        child.n_shot_ = n_shot
        child.n_finetune_rows_ = int(n_rows)
        child.adapted_ = bool(adapted)
        child._fitted = True
        return child

    # -- fine-tuning ---------------------------------------------------------- #
    def fine_tune(self, target_df: pd.DataFrame, n_shot: int | None = None,
                  ft_epochs: int | None = None) -> "TransferLSTMCorrector":
        """Adapt the pretrained weights to a target basin / gauge.

        Parameters
        ----------
        target_df : pandas.DataFrame
            Target-domain modelling table (one or more gauges of the held-out
            basin) carrying the same feature columns as the pretraining table.
        n_shot : int or None, default None
            Number of (earliest) target rows **per gauge** used for adaptation.
            ``None`` fine-tunes on *all* of ``target_df`` (full fine-tune);
            ``0`` performs no adaptation and returns the zero-shot model; a
            positive int performs few-shot adaptation.
        ft_epochs : int, optional
            Override the configured :attr:`ft_epochs` for this call.

        Returns
        -------
        TransferLSTMCorrector
            A **new** corrector backed by the adapted weights.  ``self`` (the
            pretrained prior) is left unchanged, so repeated calls with different
            ``n_shot`` all warm-start from the same prior.
        """
        if not self._fitted or self.pretrained_ is None:
            raise RuntimeError("TransferLSTMCorrector.fine_tune called before fit().")

        target_df = validate(target_df)
        shot_df = target_df if n_shot is None else self._few_shot_sample(
            target_df, int(n_shot))

        ft_model = copy.deepcopy(self.pretrained_)
        n_rows = len(shot_df)
        adapted = False
        if n_rows > 0 and (n_shot is None or int(n_shot) > 0):
            epochs = self.ft_epochs if ft_epochs is None else int(ft_epochs)
            ft_seed = self.seed * 31 + (0 if n_shot is None else int(n_shot)) + 1
            self._adapt(ft_model, shot_df, epochs, ft_seed)
            adapted = True
            log.info("transfer_lstm: fine-tuned on %d rows (n_shot=%s, epochs=%d, "
                     "freeze_body=%s)", n_rows, n_shot, epochs, self.freeze_body)
        else:
            log.info("transfer_lstm: zero-shot (no target rows adapted, n_shot=%s)",
                     n_shot)
        return self._spawn_child(ft_model, n_shot, n_rows, adapted)

    # -- transfer curve ------------------------------------------------------- #
    def transfer_curve(self, pool_df: pd.DataFrame, eval_df: pd.DataFrame,
                       shots=(0, 1, 3, 5), ft_epochs: int | None = None
                       ) -> pd.DataFrame:
        """Sweep shot counts and report KGE' vs ``n_shot`` on a fixed eval set.

        For each shot count the prior is fine-tuned on (earliest) ``n_shot`` rows
        per gauge drawn from ``pool_df`` and scored on ``eval_df`` — which should
        be **disjoint** from (and chronologically later than) ``pool_df`` to keep
        the curve leakage-safe and the evaluation comparable across shots.

        Parameters
        ----------
        pool_df : pandas.DataFrame
            Target rows available for fine-tuning (the few-shot sample is drawn
            from here).
        eval_df : pandas.DataFrame
            Fixed held-out target rows used to score every fine-tuned model.
        shots : sequence of int, default ``(0, 1, 3, 5)``
            Shot counts to evaluate.  ``None`` inside the sequence denotes a full
            fine-tune on all of ``pool_df``.
        ft_epochs : int, optional
            Override the configured fine-tuning epochs for every point.

        Returns
        -------
        pandas.DataFrame
            One row per shot count with columns ``n_shot``, ``n_finetune_rows``,
            ``kge`` and ``kge_beta`` (the KGE' bias ratio).
        """
        from ..validation.metrics import kge_prime

        eval_df = validate(eval_df)
        obs = eval_df[OBS_COL].to_numpy(float)
        rows = []
        for k in shots:
            ns = None if k is None else int(k)
            model = self.fine_tune(pool_df, n_shot=ns, ft_epochs=ft_epochs)
            q = model.predict(eval_df)
            m = kge_prime(obs, q)
            rows.append({"n_shot": ("full" if k is None else int(k)),
                         "n_finetune_rows": model.n_finetune_rows_,
                         "kge": m["kge"], "kge_beta": m["beta"]})
        return pd.DataFrame(rows)

    # -- inference ------------------------------------------------------------ #
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted log-space residual, aligned 1:1 to the rows of ``df``."""
        if not self._fitted or self.model_ is None:
            raise RuntimeError(
                "TransferLSTMCorrector.predict_residual called before fit().")
        out = np.asarray(self.model_.predict_residual(df), float).ravel()
        if out.shape[0] != len(df):  # pragma: no cover - defensive
            raise ValueError("transfer_lstm residual not aligned to df rows")
        return out


# --------------------------------------------------------------------------- #
#  Self-test (small synthetic; pretrain on 2 basins; zero-shot vs few-shot)   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from sbc.features.engineering import build_features
    from sbc.features.regimes import classify_regimes
    from sbc.synthetic import generate
    from sbc.validation.metrics import kge_prime
    from sbc.validation.splits import temporal_split

    # --- small synthetic decadal table -> features -> regimes --------------- #
    raw = generate(scale="decadal", years=8, n_basins=3,
                   gauges_per_basin=(2, 3), seed=0)
    df = classify_regimes(build_features(raw, scale="decadal"))
    df = validate(df).reset_index(drop=True)

    basins = sorted(df["basin"].unique())
    pretrain_basins, target_basin = basins[:2], basins[-1]
    pre_df = df[df["basin"].isin(pretrain_basins)].reset_index(drop=True)
    target = df[df["basin"] == target_basin].reset_index(drop=True)
    assert target_basin not in pretrain_basins, "target basin leaked into pretrain"

    # Fixed, disjoint eval set: the latest half of the held-out basin; the few-shot
    # samples are drawn from the earliest half (pool), so eval is comparable and
    # strictly posterior across all shot counts.
    ptr, pte = temporal_split(target, test_frac=0.5)
    pool, eval_df = target[ptr].reset_index(drop=True), target[pte].reset_index(drop=True)
    print(f"[transfer_lstm] pretrain basins={pretrain_basins} "
          f"({pre_df['code'].nunique()} gauges, {len(pre_df)} rows) | "
          f"held-out target={target_basin} "
          f"(pool={len(pool)}, eval={len(eval_df)} rows)")

    # tiny, fast network for the smoke test
    model = TransferLSTMCorrector(seq_length=6, hidden_size=16, max_epochs=8,
                                  batch_size=128, patience=4, ft_epochs=40,
                                  ft_learning_rate=5e-3, seed=0).fit(pre_df)

    obs = eval_df["q_obs"].to_numpy(float)
    kge_raw = kge_prime(obs, eval_df["q_glofas"].to_numpy(float))["kge"]

    # --- transfer curve: KGE' vs n_shot ------------------------------------- #
    curve = model.transfer_curve(pool, eval_df, shots=(0, 1, 3, 5))
    print(f"[transfer_lstm] raw GloFAS KGE' on eval = {kge_raw:+.3f}")
    print("[transfer_lstm] TRANSFER CURVE (KGE' vs n_shot):")
    for rec in curve.to_dict("records"):
        ns = rec["n_shot"]
        tag = "zero-shot" if ns == 0 else f"{ns}-shot"
        print(f"    {tag:>10}  n_ft_rows={int(rec['n_finetune_rows']):3d}  "
              f"KGE'={rec['kge']:+.3f}  beta={rec['kge_beta']:.3f}")

    # --- contract checks ---------------------------------------------------- #
    resid = model.predict_residual(eval_df)
    assert len(resid) == len(eval_df), "residual not aligned to df rows"

    # zero-shot child == pretrained model predictions
    zero = model.fine_tune(pool, n_shot=0)
    assert not zero.adapted_, "n_shot=0 should not adapt"
    assert np.allclose(zero.predict_residual(eval_df),
                       model.pretrained_.predict_residual(eval_df)), \
        "zero-shot must equal the pretrained model"

    # determinism: two identical fine-tunes -> identical predictions
    a = model.fine_tune(pool, n_shot=3)
    b = model.fine_tune(pool, n_shot=3)
    assert a.adapted_ and a.n_finetune_rows_ > 0, "3-shot should adapt some rows"
    assert np.allclose(a.predict_residual(eval_df), b.predict_residual(eval_df)), \
        "fine-tuning is not deterministic"

    # the prior is left untouched by fine-tuning (children are independent)
    assert model.adapted_ is False and model.n_shot_ is None, "prior was mutated"

    kge3 = kge_prime(obs, a.predict(eval_df))["kge"]
    print(f"[transfer_lstm] checks OK: aligned={len(resid) == len(eval_df)} | "
          f"zero-shot==pretrained | 3-shot deterministic | "
          f"3-shot KGE'={kge3:+.3f}")
    print("[transfer_lstm] SELF-TEST OK")
