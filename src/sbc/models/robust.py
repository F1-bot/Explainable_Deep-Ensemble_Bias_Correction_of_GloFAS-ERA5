"""Deep-ensemble and PUR-robust correctors for spatial extrapolation.

The framework's strongest generalisation test is *prediction in ungauged regions*
(PUR): train on the core Syr-Darya / Chu / Talas basins and correct GloFAS on the
fully held-out transfer (Amu Darya) domain (see
:func:`sbc.validation.splits.pur_split`).  On the real decadal run a *single*
seed of the flagship :class:`~sbc.models.regime_prob_net.RegimeProbNet` reaches a
median PUR KGE' of ~0.42 -- competitive, but *below* the boosting trio and even
the constant-scaling baseline (~0.44-0.45).  This is the classic symptom of a
high-capacity network over-fitting basin-specific quirks: a single deep model has
large *epistemic* (model) uncertainty exactly where it is asked to extrapolate.

This module supplies the standard, well-evidenced remedies and wires them into
the existing corrector / ensemble machinery without touching it:

``DeepEnsembleCorrector`` (registry name ``"deepens"``)
    A *deep ensemble* (Lakshminarayanan et al., 2017, *NeurIPS*) of ``K`` independently
    seeded deep correctors (RegimeProbNet and/or EA-LSTM).  Averaging the members'
    log-residual predictions cancels seed-specific extrapolation error -- the
    single most reliable fix for spatial-extrapolation over-fitting -- while the
    *disagreement* between members yields a calibrated epistemic-variance term
    that widens the predictive band precisely on out-of-support (PUR) rows.  The
    predictive distribution is the uniform mixture of the members', so its
    variance is the law-of-total-variance sum of the mean within-member
    (aleatoric) variance and the between-member (epistemic) variance.  It exposes
    the full probabilistic API (:meth:`predict_quantiles`, :meth:`predict_variance`,
    :meth:`sample`) so CRPS evaluation and uncertainty bands work unchanged.

``make_pur_robust_probnet``
    Factory returning a single :class:`RegimeProbNet` configured for
    *generalisation* rather than in-sample fit: stronger weight decay, a heavier
    physics (monotonicity) penalty, more mixture experts, and dropout when the
    backbone exposes it.  The rationale for each knob is documented on the
    function.  Used as the default member of the deep ensemble.

Both are ordinary :class:`~sbc.models.base.BaseCorrector` subclasses with a
``get_params``; the leakage-safe :class:`~sbc.models.ensemble.StackedEnsemble`
can therefore clone and blend them like any other base learner.
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import pandas as pd

from ..schemas import validate
from ..utils import get_logger
from .base import BaseCorrector, register
from .ea_lstm import EALSTMCorrector
from .regime_prob_net import RegimeProbNet

log = get_logger(__name__)

#: member back-bones understood by :class:`DeepEnsembleCorrector`
PROBNET, EALSTM, MIXED = "probnet", "ealstm", "mixed"
_VALID_BASES: tuple[str, ...] = (PROBNET, EALSTM, MIXED)

__all__ = ["DeepEnsembleCorrector", "make_pur_robust_probnet"]


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _supported_kwargs(cls: type, kwargs: dict) -> dict:
    """Keep only those ``kwargs`` accepted by ``cls.__init__``.

    Lets us forward a shared override dict (which may contain, e.g. ``dropout``)
    to heterogeneous member classes without raising ``TypeError`` when a class
    does not expose a given knob.  If ``__init__`` declares ``**kwargs`` the dict
    is passed through untouched.

    Parameters
    ----------
    cls : type
        Class whose constructor signature is inspected.
    kwargs : dict
        Candidate keyword arguments.

    Returns
    -------
    dict
        The accepted subset of ``kwargs``.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without signature
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    valid = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(valid))
    if dropped:
        log.debug("%s: ignoring unsupported kwargs %s", cls.__name__, dropped)
    return valid


def make_pur_robust_probnet(
    *,
    weight_decay: float = 1e-3,
    lambda_phys: float = 0.3,
    K: int = 6,
    dropout: float = 0.2,
    seed: int = 1234,
    verbose: bool = False,
    **overrides,
) -> RegimeProbNet:
    """Build a :class:`RegimeProbNet` tuned for ungauged-region generalisation.

    The defaults trade a little in-sample fit for transfer skill.  Each departure
    from the flagship's training defaults targets a known cause of PUR
    over-fitting:

    * **Stronger weight decay** (``1e-3`` vs ``1e-5``).  L2 shrinkage caps the
      norm of the EA-LSTM / expert weights, biasing the network toward smoother
      functions of the forcings.  Smooth functions extrapolate to unseen basins
      far more gracefully than the sharp, high-variance fits that minimise
      training loss.
    * **Heavier physics penalty** (``lambda_phys`` ``0.3`` vs ``0.1``).  The soft
      SWE / snowmelt monotonicity constraints encode *basin-invariant* physics:
      leaning on them harder substitutes a transferable inductive bias for the
      local statistical signal that is, by construction, absent in a held-out
      domain.
    * **More mixture experts** (``K=6`` vs ``5``).  The first five experts stay
      aligned (and gate-supervised) to the five hydrological regimes; the extra,
      free expert(s) absorb residual process structure that recurs *across*
      basins, so the regime-gated mixture transfers by process rather than by
      memorised gauge identity.
    * **Dropout** (``0.2``) on the deep backbone -- a further variance-reducing
      regulariser -- *forwarded only if the backbone constructor accepts it*.
      The current :class:`RegimeProbNet` exposes no ``dropout`` argument, so the
      value is silently dropped (see :func:`_supported_kwargs`); it is wired in
      ahead of time so that the moment the backbone gains the knob it is used.

    Parameters
    ----------
    weight_decay, lambda_phys, K, dropout, seed, verbose :
        See above; forwarded to :class:`RegimeProbNet` where supported.
    **overrides :
        Any other :class:`RegimeProbNet` keyword (e.g. ``epochs``, ``hidden``,
        ``seq_len``), overriding the robust defaults.

    Returns
    -------
    RegimeProbNet
        An unfitted, generalisation-oriented flagship corrector.
    """
    kwargs = dict(
        K=K, weight_decay=weight_decay, lambda_phys=lambda_phys, dropout=dropout,
        physics=True, seed=seed, verbose=verbose,
    )
    kwargs.update(overrides)
    return RegimeProbNet(**_supported_kwargs(RegimeProbNet, kwargs))


# --------------------------------------------------------------------------- #
#  Deep ensemble                                                              #
# --------------------------------------------------------------------------- #
@register
class DeepEnsembleCorrector(BaseCorrector):
    """Multi-seed deep ensemble of deep residual correctors.

    Fits ``n_members`` independently seeded deep correctors and treats their
    predictive distributions as a uniform mixture.  The point residual is the
    member mean; the predictive variance is

    ``var = mean_m(var_m)  +  var_m(mean_m)``

    i.e. the average *within*-member (aleatoric) variance plus the *between*-member
    (epistemic) variance.  The epistemic term -- the disagreement between seeds --
    is the component that grows on out-of-support PUR rows and is what makes the
    ensemble both more accurate (averaging cancels seed noise) and better
    calibrated there.  Deterministic members (EA-LSTM) contribute zero aleatoric
    variance, so a pure-EA-LSTM ensemble reports epistemic-only uncertainty.

    Parameters
    ----------
    base : {"probnet", "ealstm", "mixed"}, default "probnet"
        Member back-bone.  ``"probnet"`` ensembles :class:`RegimeProbNet`,
        ``"ealstm"`` ensembles :class:`EALSTMCorrector`, ``"mixed"`` alternates
        the two across seeds (a heterogeneous ensemble, which further decorrelates
        member errors).
    n_members : int, default 5
        Number of independently seeded members ``K``.
    pur_robust : bool, default True
        When ``True`` (and ``member_factory`` is not given) members are built with
        the generalisation-oriented configuration: RegimeProbNet members come from
        :func:`make_pur_robust_probnet`; EA-LSTM members get heavier weight decay
        and dropout.  Set ``False`` for a vanilla deep ensemble of the default
        configurations.
    member_kwargs : dict, optional
        Extra keyword overrides applied to every member (filtered per class).
    member_factory : callable, optional
        ``seed -> BaseCorrector`` escape hatch giving full control over member
        construction; when supplied it takes precedence over ``base`` /
        ``pur_robust`` / ``member_kwargs``.
    seed : int, default 0
        Base seed; member ``m`` is seeded with ``1000 * seed + m``.

    Attributes
    ----------
    members_ : list of BaseCorrector
        The fitted members (populated by :meth:`fit`).
    member_names_ : list of str
        Registry names of the fitted members.
    """

    name = "deepens"
    is_probabilistic = True

    def __init__(self, base: str = PROBNET, n_members: int = 5, *,
                 pur_robust: bool = True, member_kwargs: dict | None = None,
                 member_factory: Callable[[int], BaseCorrector] | None = None,
                 seed: int = 0) -> None:
        base = str(base).lower()
        if base not in _VALID_BASES:
            raise ValueError(f"base must be one of {_VALID_BASES}, got {base!r}")
        if int(n_members) < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        self.base = base
        self.n_members = int(n_members)
        self.pur_robust = bool(pur_robust)
        self.member_kwargs = dict(member_kwargs or {})
        self.member_factory = member_factory
        self.seed = int(seed)

        # learned state
        self.members_: list[BaseCorrector] = []
        self.member_names_: list[str] = []
        self._fitted = False

    # -- sklearn-style introspection so the ensemble can itself be cloned ----- #
    def get_params(self) -> dict:
        """Constructor kwargs (lets :class:`StackedEnsemble` clone this base)."""
        return {"base": self.base, "n_members": self.n_members,
                "pur_robust": self.pur_robust,
                "member_kwargs": dict(self.member_kwargs),
                "member_factory": self.member_factory, "seed": self.seed}

    # -- member construction ------------------------------------------------- #
    def _member_seed(self, idx: int) -> int:
        return 1000 * self.seed + idx

    def _build_member(self, idx: int, seed: int) -> BaseCorrector:
        """Instantiate the ``idx``-th member with its own seed."""
        if self.member_factory is not None:
            return self.member_factory(seed)

        base = self.base
        if base == MIXED:
            base = PROBNET if idx % 2 == 0 else EALSTM

        if base == PROBNET:
            kw = dict(self.member_kwargs)
            kw.pop("seed", None)
            if self.pur_robust:
                return make_pur_robust_probnet(seed=seed, **kw)
            kw = {"seed": seed, "verbose": False, **kw}
            return RegimeProbNet(**_supported_kwargs(RegimeProbNet, kw))

        # EA-LSTM member
        kw: dict = {"seed": seed}
        if self.pur_robust:
            kw.update(weight_decay=1e-3, dropout=0.5)
        kw.update(self.member_kwargs)
        kw["seed"] = seed
        return EALSTMCorrector(**_supported_kwargs(EALSTMCorrector, kw))

    # -- fitting ------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None
            ) -> "DeepEnsembleCorrector":
        """Fit every member on the same data (different seeds / shuffle order)."""
        train = validate(train)
        if valid is not None:
            valid = validate(valid)

        self.members_, self.member_names_ = [], []
        for i in range(self.n_members):
            seed = self._member_seed(i)
            member = self._build_member(i, seed)
            name = getattr(member, "name", f"member{i}")
            log.info("deepens: fitting member %d/%d (%s, seed=%d)",
                     i + 1, self.n_members, name, seed)
            try:
                member.fit(train, valid)
            except Exception as exc:  # pragma: no cover - one bad seed must not kill the run
                log.warning("deepens: member %d (%s) failed to fit (%s); skipping",
                            i, name, exc)
                continue
            self.members_.append(member)
            self.member_names_.append(name)

        if not self.members_:
            raise RuntimeError("DeepEnsembleCorrector: every member failed to fit")
        self._fitted = True
        log.info("deepens: %d/%d members fitted (base=%s, pur_robust=%s)",
                 len(self.members_), self.n_members, self.base, self.pur_robust)
        return self

    # -- member statistics --------------------------------------------------- #
    def _check_fitted(self) -> None:
        if not self._fitted or not self.members_:
            raise RuntimeError("DeepEnsembleCorrector is not fitted; call fit() first")

    def _member_stats(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-member (mean, variance) of the log-residual, shapes ``(n, M)``."""
        self._check_fitted()
        means, varis = [], []
        for member in self.members_:
            mu = np.asarray(member.predict_residual(df), float).ravel()
            var = None
            if getattr(member, "is_probabilistic", False):
                pv = getattr(member, "predict_variance", None)
                if callable(pv):
                    try:
                        var = np.asarray(pv(df), float).ravel()
                    except Exception as exc:  # pragma: no cover - defensive
                        log.debug("deepens: member variance unavailable (%s)", exc)
            if var is None or var.shape != mu.shape:
                var = np.zeros_like(mu)
            means.append(mu)
            varis.append(np.clip(var, 0.0, None))
        return np.column_stack(means), np.column_stack(varis)

    def _predictive_moments(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble (mean, total variance) of the log-residual, shapes ``(n,)``.

        ``total = mean_m(var_m) + var_m(mean_m)`` (law of total variance for a
        uniform mixture): mean within-member variance + between-member variance.
        """
        means, varis = self._member_stats(df)
        mean = np.nanmean(means, axis=1)
        aleatoric = np.nanmean(varis, axis=1)
        epistemic = np.nanvar(means, axis=1)  # ddof=0: population variance of means
        total = np.nan_to_num(aleatoric + epistemic, nan=0.0)
        return np.nan_to_num(mean, nan=0.0), np.clip(total, 0.0, None)

    # -- prediction ---------------------------------------------------------- #
    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """Ensemble-mean predicted log-residual, shape ``(n,)``."""
        mean, _ = self._predictive_moments(df)
        return mean

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Total predictive variance (aleatoric + epistemic), shape ``(n,)``."""
        _, total = self._predictive_moments(df)
        return total

    def predict_quantiles(self, df: pd.DataFrame, quantiles=(0.05, 0.5, 0.95)
                          ) -> np.ndarray:
        """Gaussian-moment-matched residual quantiles, shape ``(n, n_quantiles)``.

        The ensemble predictive distribution is summarised by its first two
        moments and inverted as a Gaussian.  This is the canonical deep-ensemble
        predictive summary (Lakshminarayanan et al., 2017): it is monotone in the
        requested probabilities by construction and works whether members are
        probabilistic or deterministic.  For a faithful (possibly multimodal)
        draw from the exact mixture use :meth:`sample`.
        """
        from scipy.stats import norm

        mean, total = self._predictive_moments(df)
        sd = np.sqrt(total)
        q = np.atleast_1d(np.asarray(quantiles, float))
        z = norm.ppf(q)
        return mean[:, None] + sd[:, None] * z[None, :]

    def sample(self, df: pd.DataFrame, n: int = 100, seed: int = 0) -> np.ndarray:
        """Pooled posterior residual samples from the member mixture, ``(n_rows, n)``.

        Draws are allocated as evenly as possible across members; probabilistic
        members are sampled from their own predictive law, deterministic members
        contribute point masses at their predicted residual.
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        n_rows, M = len(df), len(self.members_)
        counts = np.full(M, n // M, dtype=int)
        counts[: n % M] += 1

        parts: list[np.ndarray] = []
        for member, c in zip(self.members_, counts):
            if c <= 0:
                continue
            draws = None
            if getattr(member, "is_probabilistic", False):
                s_fn = getattr(member, "sample", None)
                if callable(s_fn):
                    try:
                        draws = np.asarray(
                            s_fn(df, n=int(c), seed=int(rng.integers(1 << 31))), float)
                    except Exception as exc:  # pragma: no cover - defensive
                        log.debug("deepens: member sampling unavailable (%s)", exc)
            if draws is None or draws.shape != (n_rows, c):
                mu = np.asarray(member.predict_residual(df), float).ravel()
                draws = np.repeat(mu[:, None], c, axis=1)  # point mass
            parts.append(draws)
        return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- #
#  Self-test (small synthetic; temporal + PUR; tiny 3-epoch nets)             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from sbc.features.engineering import build_features
    from sbc.features.regimes import classify_regimes
    from sbc.models.ensemble import StackedEnsemble
    from sbc.models.quantile_mapping import LinearScalingCorrector
    from sbc.synthetic import generate
    from sbc.validation.metrics import kge_prime
    from sbc.validation.splits import pur_split, temporal_split

    # --- small synthetic table -> features -> regimes ----------------------- #
    raw = generate(scale="decadal", years=8, n_basins=3,
                   gauges_per_basin=(2, 3), seed=11)
    df = classify_regimes(build_features(raw, scale="decadal"))
    df = df.reset_index(drop=True)
    print(f"[robust] table: {len(df)} rows | {df['code'].nunique()} gauges | "
          f"{df['basin'].nunique()} basins | domains={sorted(df['domain'].unique())}")

    # tiny, fast member config for the smoke test (3 epochs)
    mk = dict(epochs=3, hidden=20, seq_len=4, expert_hidden=16, gate_hidden=16,
              batch_size=1024, patience=3, K=4, lambda_phys=0.3)

    def _kge(test, q):
        return kge_prime(test["q_obs"].to_numpy(float), np.asarray(q, float))["kge"]

    # --- (1) temporal holdout: single robust probnet vs deep ensemble ------- #
    ttr, tte = temporal_split(df, test_frac=0.3)
    tr_t, te_t = df[ttr].reset_index(drop=True), df[tte].reset_index(drop=True)

    single = make_pur_robust_probnet(seed=0, **mk).fit(tr_t)
    ens = DeepEnsembleCorrector(base="probnet", n_members=2, pur_robust=True,
                                member_kwargs=mk, seed=0).fit(tr_t)

    raw_t = _kge(te_t, te_t["q_glofas"].to_numpy(float))
    single_t = _kge(te_t, single.predict(te_t))
    ens_t = _kge(te_t, ens.predict(te_t))
    print(f"[robust] TEMPORAL  KGE' raw={raw_t:+.3f} | single={single_t:+.3f} | "
          f"deepens(K=2)={ens_t:+.3f}")

    # --- (2) PUR: train on core, correct the held-out transfer domain ------- #
    ptr, pte = pur_split(df)
    tr_p, te_p = df[ptr].reset_index(drop=True), df[pte].reset_index(drop=True)
    ens_p_model = DeepEnsembleCorrector(base="probnet", n_members=2,
                                        pur_robust=True, member_kwargs=mk, seed=0).fit(tr_p)
    single_p = make_pur_robust_probnet(seed=0, **mk).fit(tr_p)
    raw_p = _kge(te_p, te_p["q_glofas"].to_numpy(float))
    print(f"[robust] PUR       KGE' raw={raw_p:+.3f} | "
          f"single={_kge(te_p, single_p.predict(te_p)):+.3f} | "
          f"deepens(K=2)={_kge(te_p, ens_p_model.predict(te_p)):+.3f}")

    # --- (3) probabilistic API: quantiles monotone, variance decomposition -- #
    qs = ens.predict_quantiles(te_t, (0.05, 0.25, 0.5, 0.75, 0.95))
    monotone = bool(np.all(np.diff(qs, axis=1) >= -1e-9))
    var = ens.predict_variance(te_t)
    draws = ens.sample(te_t, n=40, seed=1)
    disc_q = ens.predict_discharge_quantiles(te_t, (0.05, 0.5, 0.95))
    print(f"[robust] quantiles{qs.shape} monotone={monotone} | "
          f"var>=0={bool(np.all(var >= 0))} mean_var={var.mean():.4f} | "
          f"samples{draws.shape} | disc_q{disc_q.shape}")
    assert qs.shape == (len(te_t), 5) and monotone, "ensemble quantiles not calibrated-shaped"
    assert var.shape == (len(te_t),) and np.all(np.isfinite(var)), "bad variance"
    assert draws.shape == (len(te_t), 40), "bad sample shape"

    # --- (4) StackedEnsemble can clone & blend the deep ensemble ------------ #
    fast_ens = DeepEnsembleCorrector(
        base="ealstm", n_members=2, pur_robust=True,
        member_kwargs=dict(max_epochs=2, hidden_size=12, seq_length=4, batch_size=1024),
        seed=0)
    stack = StackedEnsemble([LinearScalingCorrector(), fast_ens],
                            meta="nnls", seed=0, n_folds=2).fit(tr_t)
    stack_t = _kge(te_t, stack.predict(te_t))
    print(f"[robust] STACKED(deepens included) weights={dict(stack.weights_.round(3))} "
          f"| KGE'={stack_t:+.3f}")
    assert "deepens" in stack.base_names_, "deepens not deployed inside StackedEnsemble"

    print("[robust] SELF-TEST OK")
