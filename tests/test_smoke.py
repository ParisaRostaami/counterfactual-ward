"""Smoke tests: AUC, counterfactual validity, sparsity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual import search_many
from src.evaluate import mean_sparsity_fraction, validity_rate
from src.model import heldout_auc, train_model
from src.patients import dataframe_for_model, generate_cohort, load_vignettes

SEED = 42


@pytest.fixture(scope="module")
def fitted():
    df = generate_cohort(n=360, seed=SEED)
    X, y = dataframe_for_model(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    model = train_model(X_tr, y_tr, seed=SEED)
    return model, X_te, y_te


def test_model_auc_above_08(fitted) -> None:
    model, X_te, y_te = fitted
    auc = heldout_auc(model, X_te, y_te)
    assert auc > 0.8, f"AUC {auc:.3f} was not > 0.8"


def test_counterfactuals_mostly_flip_and_stay_sparse(fitted) -> None:
    model, X_te, _ = fitted
    sample = X_te.head(16)
    cfs = search_many(model, sample, seed=SEED)
    valid = validity_rate(cfs)
    frac = mean_sparsity_fraction(cfs)
    assert valid >= 0.8, f"validity {valid:.2f} was not >= 0.8"
    assert frac < 0.5, f"mean sparsity fraction {frac:.2f} was not < 0.5"


def test_vignettes_json_loads() -> None:
    df = load_vignettes()
    assert len(df) >= 20
    assert "notes" in df.columns
    assert set(df["high_30d_risk"].unique()).issubset({0, 1})
    assert df["notes"].str.len().min() > 10
