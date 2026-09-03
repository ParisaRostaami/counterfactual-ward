"""Validity, proximity, and sparsity of a counterfactual batch."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .counterfactual import Counterfactual


def validity_rate(cfs: Iterable[Counterfactual]) -> float:
    items = list(cfs)
    if not items:
        return 0.0
    return float(np.mean([1.0 if c.flipped else 0.0 for c in items]))


def mean_proximity(cfs: Iterable[Counterfactual]) -> float:
    items = list(cfs)
    if not items:
        return 0.0
    return float(np.mean([c.proximity for c in items]))


def mean_sparsity(cfs: Iterable[Counterfactual]) -> float:
    items = list(cfs)
    if not items:
        return 0.0
    return float(np.mean([c.sparsity for c in items]))


def mean_sparsity_fraction(cfs: Iterable[Counterfactual]) -> float:
    items = list(cfs)
    if not items:
        return 0.0
    return float(np.mean([c.sparsity / max(c.n_features, 1) for c in items]))


def mean_cost(cfs: Iterable[Counterfactual]) -> float:
    items = list(cfs)
    if not items:
        return 0.0
    return float(np.mean([c.cost for c in items]))


def summarize(cfs: list[Counterfactual]) -> dict[str, float]:
    return {
        "n": float(len(cfs)),
        "validity": validity_rate(cfs),
        "mean_proximity": mean_proximity(cfs),
        "mean_sparsity": mean_sparsity(cfs),
        "mean_sparsity_fraction": mean_sparsity_fraction(cfs),
        "mean_cost": mean_cost(cfs),
    }
