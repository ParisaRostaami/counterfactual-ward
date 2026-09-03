# Counterfactual ward (synthetic clinical XAI)

**Parisa Rostami** · PhD student, Wichita State University

A self-contained prototype for *“what is the smallest change that would flip this clinical decision?”* on tabular vitals/labs plus a short free-text note. The patients are invented. The point is the explanation machinery, not a deployed risk score.

This sits next to explainable-AI / healthcare interests (feature attribution, inspectable models) without reproducing any published Parkinson’s or EHR study.

## Decision being explained

The binary label is **high 30-day risk** (a stand-in for “refer / watch closely,” not a real outcome). Labels come from a **known scoring rule** on age, blood pressure, SpO2, glucose, creatinine, heart rate, BMI, temperature, WBC, and a handful of note phrases, plus Gaussian noise. A logistic pipeline is then trained only on the observed columns. The model is learnable but not a lookup table.

Shipped vignettes live in `data/vignettes.json`. `demo.py` trains on a larger on-the-fly cohort (`src/patients.py`, seed 42) so TF-IDF sees enough notes.

## Method

1. **`src/model.py`** — `ColumnTransformer`: StandardScaler on numerics, one-hot sex, TF-IDF (1–2 grams) on notes, logistic regression.

2. **`src/counterfactual.py`** — DiCE-like search without the DiCE package:
   - greedy coordinate steps in the *actionable* direction (lower BP, raise SpO2, …)
   - optional note **phrase swaps** from a fixed lexicon
   - random local search
   - a prototype-push fallback if the prediction still has not flipped
   Cost = range-normalized change × **actionability** weight. Age is expensive; sex is immutable; systolic BP and glucose are cheap.

3. **`src/report.py`** — Markdown cards a clinician could skim (this is still toy data).

4. **`src/evaluate.py`** — validity (label flip), proximity (normalized L1), sparsity (count of edited features).

## Honest limits

- No real EHR, no IRB, no calibration to 30-day readmission.
- Phrase swaps are not clinical documentation advice.
- A flip of the **model** is not a flip of physiology. Counterfactuals here are a debugging lens for the classifier.
- Validity ≥ 80% is required on this synthetic generator; it is not a claim about DiCE on MIMIC or similar.
- No network calls. Seed **42**.

## Setup

Python 3.10+. From this directory:

```text
python -m pip install -r requirements.txt
python demo.py
python -m pytest -q
```

`demo.py` writes `outputs/cf_cards.md` (five vignette patients) and `outputs/proximity_vs_sparsity.png`.

## Layout

```text
data/vignettes.json
demo.py
src/patients.py
src/model.py
src/counterfactual.py
src/report.py
src/evaluate.py
tests/test_smoke.py
```

## Tests

- Held-out ROC AUC > 0.8 on a fresh synthetic split.
- At least 80% of generated counterfactuals flip the predicted class.
- Average fraction of features changed is under one half.
