"""Tabular + TF-IDF pipeline for synthetic 30-day risk."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .patients import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE


def build_pipeline(seed: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "txt",
                TfidfVectorizer(
                    max_features=96,
                    ngram_range=(1, 2),
                    min_df=1,
                    lowercase=True,
                ),
                TEXT_FEATURE,
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=800,
        class_weight="balanced",
        C=1.4,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
) -> Pipeline:
    model = build_pipeline(seed=seed)
    model.fit(X, y)
    return model


def predict_proba_positive(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    idx = classes.index(1) if 1 in classes else 1
    return proba[:, idx]


def predict_labels(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(int)


def row_frame(row: pd.Series | dict[str, Any], columns: list[str] | None = None) -> pd.DataFrame:
    s = pd.Series(row)
    if columns is None:
        columns = [c for c in s.index if c in set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + (TEXT_FEATURE,))]
        # keep a stable order
        order = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES) + [TEXT_FEATURE]
        columns = [c for c in order if c in s.index]
    return pd.DataFrame([s[columns].to_dict()], columns=columns)


def predict_row(model: Pipeline, row: pd.Series | dict) -> int:
    return int(predict_labels(model, row_frame(row))[0])


def proba_row(model: Pipeline, row: pd.Series | dict) -> float:
    return float(predict_proba_positive(model, row_frame(row))[0])


def heldout_auc(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
    p = predict_proba_positive(model, X)
    return float(roc_auc_score(y, p))
