"""DiCE-like search: greedy then random minimal edits that flip the prediction.

Cost is MAD-style normalized change times an actionability weight. Age is
expensive; blood pressure and glucose are cheap; sex is treated as immutable.
Notes can change via a small phrase lexicon (not free-text generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .model import predict_row, proba_row
from .patients import (
    ACTIONABILITY,
    ALL_MODEL_FEATURES,
    FEATURE_BOUNDS,
    LOW_RISK_DIRECTION,
    NUMERIC_FEATURES,
    STEP,
    TEXT_FEATURE,
)

IMMUTABLE = frozenset({"sex"})

# Phrase substitutions that move documentation toward lower / higher risk.
_TO_SAFE: tuple[tuple[str, str], ...] = (
    ("chest pain", "denies chest pain"),
    ("shortness of breath", "breathing comfortably"),
    ("dyspnea", "breathing comfortably"),
    ("confusion", "alert and oriented"),
    ("altered mental status", "at baseline"),
    ("edema", "no edema"),
    ("oliguria", "normal urine output"),
    ("worsening", "improving"),
    ("family worried", "family reassured"),
)
_TO_RISK: tuple[tuple[str, str], ...] = (
    ("denies chest pain", "chest pain"),
    ("breathing comfortably", "shortness of breath"),
    ("alert and oriented", "confusion"),
    ("at baseline", "altered mental status"),
    ("stable", "worsening"),
    ("improving", "worsening"),
    ("no edema", "edema"),
    ("normal urine output", "oliguria"),
)

SAFE_NOTE = (
    "Alert and oriented. Breathing comfortably. Denies chest pain. "
    "Stable and improving. At baseline."
)
RISK_NOTE = (
    "Chest pain and shortness of breath. Confusion and altered mental status. "
    "Edema and oliguria."
)
CHEAP_FEATURES: tuple[str, ...] = (
    "systolic_bp",
    "spo2",
    "glucose",
    "heart_rate",
    "temperature",
    "diastolic_bp",
)


@dataclass
class Counterfactual:
    original: dict[str, Any]
    counterfactual: dict[str, Any]
    original_label: int
    counterfactual_label: int
    original_proba: float
    counterfactual_proba: float
    cost: float
    proximity: float
    sparsity: int
    n_features: int
    flipped: bool
    changes: list[dict[str, Any]] = field(default_factory=list)


def _as_dict(row: pd.Series | dict) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        d = row.to_dict()
    else:
        d = dict(row)
    return {k: d[k] for k in ALL_MODEL_FEATURES if k in d}


def _clip_feature(name: str, value: float) -> float:
    lo, hi = FEATURE_BOUNDS[name]
    return float(np.clip(value, lo, hi))


def _numeric_changed(a: float, b: float, name: str) -> bool:
    return abs(float(a) - float(b)) > 0.25 * STEP.get(name, 1.0)


def describe_changes(original: dict, cf: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in NUMERIC_FEATURES:
        o, n = float(original[name]), float(cf[name])
        if _numeric_changed(o, n, name):
            out.append(
                {
                    "feature": name,
                    "from": round(o, 3),
                    "to": round(n, 3),
                    "actionability": ACTIONABILITY[name],
                }
            )
    if str(original[TEXT_FEATURE]) != str(cf[TEXT_FEATURE]):
        out.append(
            {
                "feature": TEXT_FEATURE,
                "from": str(original[TEXT_FEATURE]),
                "to": str(cf[TEXT_FEATURE]),
                "actionability": ACTIONABILITY[TEXT_FEATURE],
            }
        )
    return out


def feature_cost(original: dict, cf: dict) -> float:
    total = 0.0
    for name in NUMERIC_FEATURES:
        lo, hi = FEATURE_BOUNDS[name]
        scale = max(hi - lo, 1e-6)
        delta = abs(float(cf[name]) - float(original[name])) / scale
        total += ACTIONABILITY[name] * delta
    if str(cf[TEXT_FEATURE]) != str(original[TEXT_FEATURE]):
        total += ACTIONABILITY[TEXT_FEATURE] * 0.35
    return float(total)


def proximity(original: dict, cf: dict) -> float:
    """Unweighted L1 in unit-range coordinates (lower is closer)."""
    acc = 0.0
    n = 0
    for name in NUMERIC_FEATURES:
        lo, hi = FEATURE_BOUNDS[name]
        scale = max(hi - lo, 1e-6)
        acc += abs(float(cf[name]) - float(original[name])) / scale
        n += 1
    if str(cf[TEXT_FEATURE]) != str(original[TEXT_FEATURE]):
        acc += 0.25
        n += 1
    else:
        n += 1
    return float(acc / max(n, 1))


def sparsity_count(original: dict, cf: dict) -> int:
    n = 0
    for name in NUMERIC_FEATURES:
        if _numeric_changed(float(original[name]), float(cf[name]), name):
            n += 1
    if str(original[TEXT_FEATURE]) != str(cf[TEXT_FEATURE]):
        n += 1
    return n


def _note_candidates(notes: str, target_low_risk: bool) -> list[str]:
    pairs = _TO_SAFE if target_low_risk else _TO_RISK
    text = notes
    out: list[str] = []
    lower = text.lower()
    for src, dst in pairs:
        if src in lower:
            idx = lower.find(src)
            new = text[:idx] + dst + text[idx + len(src) :]
            if new != text:
                out.append(new)
    template = SAFE_NOTE if target_low_risk else RISK_NOTE
    if template not in out and template != text:
        out.append(template)
    return out


def _step_value(current: dict, name: str, toward_low: bool, steps: int) -> float:
    direction = LOW_RISK_DIRECTION[name] if toward_low else -LOW_RISK_DIRECTION[name]
    nxt = float(current[name]) + direction * STEP[name] * steps
    return _clip_feature(name, nxt)


def _candidate_rows(current: dict, toward_low: bool) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    for name in CHEAP_FEATURES:
        for k in (1, 2, 4, 8):
            trial = dict(current)
            trial[name] = _step_value(current, name, toward_low, k)
            if abs(trial[name] - float(current[name])) < 1e-9:
                continue
            cands.append(trial)
    for new_notes in _note_candidates(str(current[TEXT_FEATURE]), toward_low):
        trial = dict(current)
        trial[TEXT_FEATURE] = new_notes
        cands.append(trial)
    return cands


def _target_proba(model, row: dict, target: int) -> float:
    p = proba_row(model, row)
    return p if target == 1 else (1.0 - p)


def _score_candidate(
    model,
    original: dict,
    trial: dict,
    target: int,
) -> tuple[float, float, int]:
    """Rank by target probability, then cheaper / sparser edits."""
    p_target = _target_proba(model, trial, target)
    cost = feature_cost(original, trial)
    spar = sparsity_count(original, trial)
    return (p_target, -cost, -spar)


def _random_mutate(
    rng: np.random.Generator,
    original: dict,
    current: dict,
    toward_low: bool,
    n_edit: int,
) -> dict[str, Any]:
    trial = dict(current)
    mutable = list(CHEAP_FEATURES)
    chosen = rng.choice(mutable, size=min(n_edit, len(mutable)), replace=False)
    for name in chosen:
        k = int(rng.integers(2, 9))
        trial[name] = _step_value(trial, name, toward_low, k)
    if rng.random() < 0.6:
        opts = _note_candidates(str(trial[TEXT_FEATURE]), toward_low)
        if opts:
            trial[TEXT_FEATURE] = str(rng.choice(opts))
    return trial


def _prototype_push(original: dict, toward_low: bool, strength: float = 0.65) -> dict[str, Any]:
    """Move cheap vitals a fraction of the way toward a healthy/unhealthy pole."""
    trial = dict(original)
    poles_low = {
        "systolic_bp": 112.0,
        "diastolic_bp": 72.0,
        "heart_rate": 70.0,
        "spo2": 99.0,
        "glucose": 92.0,
        "creatinine": 0.85,
        "temperature": 36.7,
        "wbc": 7.0,
        "bmi": 25.0,
        "age": float(original["age"]),
    }
    poles_high = {
        "systolic_bp": 178.0,
        "diastolic_bp": 98.0,
        "heart_rate": 116.0,
        "spo2": 88.0,
        "glucose": 230.0,
        "creatinine": 2.2,
        "temperature": 38.3,
        "wbc": 14.5,
        "bmi": 34.0,
        "age": min(float(original["age"]) + 2.0, FEATURE_BOUNDS["age"][1]),
    }
    poles = poles_low if toward_low else poles_high
    for name in CHEAP_FEATURES:
        cur = float(trial[name])
        tgt = poles[name]
        trial[name] = _clip_feature(name, cur + strength * (tgt - cur))
    trial[TEXT_FEATURE] = SAFE_NOTE if toward_low else RISK_NOTE
    return trial


def _hard_flip(original: dict, toward_low: bool) -> dict[str, Any]:
    """Last-resort sparse pole assignment on the four cheapest levers + notes."""
    trial = dict(original)
    if toward_low:
        trial["systolic_bp"] = 110.0
        trial["spo2"] = 99.0
        trial["glucose"] = 90.0
        trial["heart_rate"] = 68.0
        trial[TEXT_FEATURE] = SAFE_NOTE
    else:
        trial["systolic_bp"] = 184.0
        trial["spo2"] = 87.0
        trial["glucose"] = 250.0
        trial["heart_rate"] = 120.0
        trial[TEXT_FEATURE] = RISK_NOTE
    return trial


def search_counterfactual(
    model,
    row: pd.Series | dict,
    seed: int = 42,
    max_greedy: int = 10,
    n_random: int = 20,
) -> Counterfactual:
    """Greedy coordinate steps, random search, then a sparse pole fallback."""
    rng = np.random.default_rng(seed)
    original = _as_dict(row)
    y0 = predict_row(model, original)
    p0 = proba_row(model, original)
    target = 1 - y0
    toward_low = target == 0

    current = dict(original)
    best_p = _target_proba(model, current, target)

    for _ in range(max_greedy):
        if predict_row(model, current) == target:
            break
        scored: list[tuple[tuple[float, float, int], dict]] = []
        for trial in _candidate_rows(current, toward_low):
            scored.append((_score_candidate(model, original, trial, target), trial))
        if not scored:
            break
        scored.sort(key=lambda t: t[0], reverse=True)
        _, trial = scored[0]
        p_target = _target_proba(model, trial, target)
        if p_target + 1e-4 <= best_p and predict_row(model, trial) != target:
            break
        current = trial
        best_p = max(best_p, p_target)

    if predict_row(model, current) != target:
        best_trial = dict(current)
        best_sc = _score_candidate(model, original, current, target)
        for _ in range(n_random):
            if predict_row(model, best_trial) == target:
                break
            n_edit = int(rng.integers(1, 4))
            trial = _random_mutate(rng, original, current, toward_low, n_edit)
            sc = _score_candidate(model, original, trial, target)
            if predict_row(model, trial) == target:
                if feature_cost(original, trial) < feature_cost(original, best_trial) or predict_row(
                    model, best_trial
                ) != target:
                    best_trial = trial
                    best_sc = sc
            elif sc > best_sc:
                best_trial = trial
                best_sc = sc
        current = best_trial

    if predict_row(model, current) != target:
        for strength in (0.8, 1.0):
            trial = _prototype_push(original, toward_low, strength=strength)
            if predict_row(model, trial) == target:
                current = trial
                break
            current = trial
    if predict_row(model, current) != target:
        current = _hard_flip(original, toward_low)
    if predict_row(model, current) != target:
        trial = _hard_flip(original, toward_low)
        trial["creatinine"] = 0.75 if toward_low else 2.6
        trial["wbc"] = 6.5 if toward_low else 16.0
        current = trial

    y1 = predict_row(model, current)
    p1 = proba_row(model, current)
    return Counterfactual(
        original=original,
        counterfactual=current,
        original_label=int(y0),
        counterfactual_label=int(y1),
        original_proba=float(p0),
        counterfactual_proba=float(p1),
        cost=feature_cost(original, current),
        proximity=proximity(original, current),
        sparsity=sparsity_count(original, current),
        n_features=len(NUMERIC_FEATURES) + 1,
        flipped=bool(y1 != y0),
        changes=describe_changes(original, current),
    )


def search_many(
    model,
    frames: pd.DataFrame,
    seed: int = 42,
) -> list[Counterfactual]:
    out: list[Counterfactual] = []
    for i, (_, row) in enumerate(frames.iterrows()):
        out.append(search_counterfactual(model, row, seed=seed + i))
    return out
