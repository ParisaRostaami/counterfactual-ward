"""Synthetic ward vignettes: vitals, labs, short notes, and a noisy risk label."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SEED = 42
NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "spo2",
    "glucose",
    "creatinine",
    "bmi",
    "temperature",
    "wbc",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("sex",)
TEXT_FEATURE = "notes"
LABEL = "high_30d_risk"
ALL_MODEL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + (TEXT_FEATURE,)

# sex: 0 = female, 1 = male. Immutable in counterfactual search.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (28.0, 95.0),
    "systolic_bp": (88.0, 210.0),
    "diastolic_bp": (48.0, 125.0),
    "heart_rate": (48.0, 145.0),
    "spo2": (84.0, 100.0),
    "glucose": (65.0, 320.0),
    "creatinine": (0.40, 4.20),
    "bmi": (17.0, 48.0),
    "temperature": (35.8, 40.2),
    "wbc": (2.8, 22.0),
}

# Higher weight = less actionable. Age is hard to "edit"; BP is not.
ACTIONABILITY: dict[str, float] = {
    "age": 7.5,
    "sex": 1e6,
    "systolic_bp": 0.45,
    "diastolic_bp": 0.65,
    "heart_rate": 0.70,
    "spo2": 0.55,
    "glucose": 0.60,
    "creatinine": 1.60,
    "bmi": 2.80,
    "temperature": 0.50,
    "wbc": 1.90,
    "notes": 0.90,
}

STEP: dict[str, float] = {
    "age": 2.0,
    "systolic_bp": 4.0,
    "diastolic_bp": 3.0,
    "heart_rate": 4.0,
    "spo2": 1.0,
    "glucose": 8.0,
    "creatinine": 0.15,
    "bmi": 1.0,
    "temperature": 0.2,
    "wbc": 0.6,
}

# Direction that *lowers* latent risk.
LOW_RISK_DIRECTION: dict[str, float] = {
    "age": -1.0,
    "systolic_bp": -1.0,
    "diastolic_bp": -1.0,
    "heart_rate": -1.0,
    "spo2": 1.0,
    "glucose": -1.0,
    "creatinine": -1.0,
    "bmi": -1.0,
    "temperature": -1.0,
    "wbc": -1.0,
}

NOTE_PHRASES_RISK = (
    "chest pain",
    "shortness of breath",
    "dyspnea",
    "confusion",
    "altered mental status",
    "edema",
    "oliguria",
)
NOTE_PHRASES_SAFE = (
    "denies chest pain",
    "breathing comfortably",
    "alert and oriented",
    "stable",
    "improving",
    "at baseline",
)


@dataclass(frozen=True)
class RiskRule:
    """Known generating process plus Gaussian noise. Not a clinical guideline."""

    threshold: float = 2.20
    noise_std: float = 0.42


def latent_score(row: pd.Series | dict) -> float:
    """Continuous risk used to mint labels. The classifier never sees this."""
    r = row
    s = 0.0
    s += 0.038 * max(0.0, float(r["age"]) - 58.0)
    s += 0.022 * max(0.0, float(r["systolic_bp"]) - 128.0)
    s += 0.012 * max(0.0, float(r["diastolic_bp"]) - 82.0)
    s += 0.018 * max(0.0, float(r["heart_rate"]) - 88.0)
    s += 0.055 * max(0.0, 96.0 - float(r["spo2"]))
    s += 0.010 * max(0.0, float(r["glucose"]) - 108.0)
    s += 1.15 * max(0.0, float(r["creatinine"]) - 1.05)
    s += 0.045 * max(0.0, float(r["bmi"]) - 30.0)
    s += 0.55 * max(0.0, float(r["temperature"]) - 37.4)
    s += 0.09 * max(0.0, float(r["wbc"]) - 10.5)
    notes = str(r["notes"]).lower()
    if "chest pain" in notes and "denies chest pain" not in notes:
        s += 1.05
    if "shortness of breath" in notes or "dyspnea" in notes:
        s += 0.85
    if "confusion" in notes or "altered mental" in notes:
        s += 0.75
    if "edema" in notes:
        s += 0.35
    if "oliguria" in notes:
        s += 0.45
    if "denies chest pain" in notes:
        s -= 0.40
    if "stable" in notes:
        s -= 0.32
    if "improving" in notes:
        s -= 0.28
    if "alert and oriented" in notes or "at baseline" in notes:
        s -= 0.22
    if "breathing comfortably" in notes:
        s -= 0.30
    return float(s)


def label_from_score(score: float, noise: float, rule: RiskRule | None = None) -> int:
    rule = rule or RiskRule()
    return int((score + noise) > rule.threshold)


_NOTE_BANK_HIGH = [
    "Overnight chest pain and shortness of breath. Family worried.",
    "Confusion since evening. Altered mental status, not at baseline.",
    "Dyspnea on exertion, bilateral edema, poor oral intake.",
    "Chest pain radiating to arm. Diaphoretic on arrival.",
    "Oliguria and edema. Shortness of breath at rest.",
    "Worsening confusion with low-grade fever.",
    "Chest pain plus dyspnea. Needs closer monitoring.",
]

_NOTE_BANK_LOW = [
    "Follow-up visit. Patient is stable and improving. Denies chest pain.",
    "Alert and oriented. Breathing comfortably. At baseline.",
    "Routine check. Stable, improving appetite, denies chest pain.",
    "Post-treatment review. Breathing comfortably, at baseline.",
    "Unremarkable rounding note. Stable overnight, alert and oriented.",
]

_NOTE_BANK_MIXED = [
    "Denies chest pain but mild dyspnea after walking the hall.",
    "Stable overall; family mentions brief confusion last night.",
    "Improving, alert and oriented, still some edema in the ankles.",
    "Breathing comfortably now. History of chest pain yesterday.",
    "At baseline mentation. Glucose running high, otherwise stable.",
]


def _sample_notes(rng: np.random.Generator, latent_hint: str) -> str:
    if latent_hint == "high":
        bank = _NOTE_BANK_HIGH + _NOTE_BANK_MIXED[:2]
    elif latent_hint == "low":
        bank = _NOTE_BANK_LOW + _NOTE_BANK_MIXED[2:]
    else:
        bank = _NOTE_BANK_MIXED + _NOTE_BANK_LOW[:1] + _NOTE_BANK_HIGH[:1]
    return str(rng.choice(bank))


def _clip(name: str, value: float) -> float:
    lo, hi = FEATURE_BOUNDS[name]
    return float(np.clip(value, lo, hi))


def generate_patient(rng: np.random.Generator, toward: str | None = None) -> dict:
    """Draw one synthetic patient. ``toward`` biases vitals, not the label."""
    if toward == "high":
        age = rng.uniform(68, 90)
        sbp = rng.uniform(145, 195)
        dbp = rng.uniform(82, 110)
        hr = rng.uniform(92, 128)
        spo2 = rng.uniform(86, 95)
        glu = rng.uniform(140, 270)
        cr = rng.uniform(1.2, 3.2)
        bmi = rng.uniform(29, 42)
        temp = rng.uniform(37.2, 39.1)
        wbc = rng.uniform(9.5, 18.0)
        notes = _sample_notes(rng, "high")
    elif toward == "low":
        age = rng.uniform(32, 62)
        sbp = rng.uniform(105, 132)
        dbp = rng.uniform(62, 84)
        hr = rng.uniform(58, 88)
        spo2 = rng.uniform(96, 100)
        glu = rng.uniform(78, 118)
        cr = rng.uniform(0.55, 1.10)
        bmi = rng.uniform(20, 29)
        temp = rng.uniform(36.4, 37.3)
        wbc = rng.uniform(4.5, 9.5)
        notes = _sample_notes(rng, "low")
    else:
        age = rng.uniform(35, 88)
        sbp = rng.uniform(100, 180)
        dbp = rng.uniform(58, 105)
        hr = rng.uniform(55, 120)
        spo2 = rng.uniform(88, 100)
        glu = rng.uniform(75, 240)
        cr = rng.uniform(0.5, 2.8)
        bmi = rng.uniform(19, 40)
        temp = rng.uniform(36.3, 38.6)
        wbc = rng.uniform(4.0, 16.0)
        notes = _sample_notes(rng, "mix")

    sex = int(rng.integers(0, 2))
    row = {
        "age": _clip("age", age),
        "sex": sex,
        "systolic_bp": _clip("systolic_bp", sbp),
        "diastolic_bp": _clip("diastolic_bp", dbp),
        "heart_rate": _clip("heart_rate", hr),
        "spo2": _clip("spo2", spo2),
        "glucose": _clip("glucose", glu),
        "creatinine": _clip("creatinine", cr),
        "bmi": _clip("bmi", bmi),
        "temperature": round(_clip("temperature", temp), 1),
        "wbc": round(_clip("wbc", wbc), 1),
        "notes": notes,
    }
    return row


def assign_label(row: dict, rng: np.random.Generator, rule: RiskRule | None = None) -> dict:
    rule = rule or RiskRule()
    score = latent_score(row)
    noise = float(rng.normal(0.0, rule.noise_std))
    out = dict(row)
    out["latent_score"] = round(score, 4)
    out[LABEL] = label_from_score(score, noise, rule)
    return out


def generate_cohort(
    n: int = 360,
    seed: int = SEED,
    rule: RiskRule | None = None,
) -> pd.DataFrame:
    """Balanced-ish cohort: 40% high-biased, 40% low-biased, 20% mixed draws."""
    rng = np.random.default_rng(seed)
    rows = []
    n_high = int(0.40 * n)
    n_low = int(0.40 * n)
    n_mix = n - n_high - n_low
    for toward, count in (("high", n_high), ("low", n_low), (None, n_mix)):
        for _ in range(count):
            rows.append(assign_label(generate_patient(rng, toward=toward), rng, rule))
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df["patient_id"] = [f"P{i:04d}" for i in range(len(df))]
    cols = ["patient_id", *ALL_MODEL_FEATURES, "latent_score", LABEL]
    return df[cols]


def load_vignettes(path: str | Path | None = None) -> pd.DataFrame:
    """Load the shipped JSON vignettes (no internet)."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "vignettes.json"
    path = Path(path)
    df = pd.read_json(path)
    return df


def dataframe_for_model(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.loc[:, list(ALL_MODEL_FEATURES)].copy()
    y = df[LABEL].astype(int).to_numpy()
    return X, y
