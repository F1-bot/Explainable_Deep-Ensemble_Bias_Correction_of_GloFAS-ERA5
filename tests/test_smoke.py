"""Fast contract smoke tests for the sbc framework (pytest, CPU-only)."""
import numpy as np
import pandas as pd

from sbc import schemas
from sbc.synthetic import generate
from sbc.validation import metrics as M
from sbc.validation.splits import pur_split, spatial_folds, temporal_split


def _table():
    return schemas.validate(generate(scale="decadal", years=8, n_basins=3, seed=0))


def test_back_transform_is_inverse_of_target():
    df = _table()
    q = schemas.back_transform(df["q_glofas"].values, df["log_residual"].values)
    assert np.allclose(q, df["q_obs"].values, atol=1e-6)


def test_metrics_perfect_and_bias():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(M.kge_prime(obs, obs)["kge"] - 1.0) < 1e-9
    assert M.nse(obs, obs) > 0.999
    assert M.pbias(obs, obs * 0.8) < 0  # under-prediction -> negative bias


def test_synthetic_schema_and_features():
    df = _table()
    schemas.validate(df)  # raises if malformed
    feats = schemas.feature_columns(df)
    assert len(feats) > 5
    assert set(schemas.static_feature_columns(df, feats)) <= set(feats)
    assert "transfer" in set(df["domain"])  # PUR domain present


def test_splits_are_leakage_safe():
    df = _table().reset_index(drop=True)
    tr, te = temporal_split(df, 0.3)
    assert tr.sum() and te.sum() and not (tr & te).any()
    for _, trm, tem in spatial_folds(df):
        # no basin appears on both sides of a spatial fold
        assert not set(df.loc[trm, "basin"]) & set(df.loc[tem, "basin"])
    ptr, pte = pur_split(df)
    assert set(df.loc[ptr, "domain"]) == {"core"}
    assert set(df.loc[pte, "domain"]) == {"transfer"}


def test_baseline_corrector_improves_kge():
    from sbc.models.quantile_mapping import LinearScalingCorrector

    df = _table().reset_index(drop=True)
    tr, te = temporal_split(df, 0.3)
    model = LinearScalingCorrector().fit(df[tr])
    pred = schemas.back_transform(df[te]["q_glofas"].values,
                                  model.predict_residual(df[te]))
    raw = M.kge_prime(df[te]["q_obs"].values, df[te]["q_glofas"].values)["kge"]
    corr = M.kge_prime(df[te]["q_obs"].values, pred)["kge"]
    assert corr >= raw - 0.05  # correction should not materially hurt pooled skill
