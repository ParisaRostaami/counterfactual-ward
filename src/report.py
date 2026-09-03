"""Human-readable counterfactual cards (markdown)."""

from __future__ import annotations

from pathlib import Path

from .counterfactual import Counterfactual

_LABEL_NAME = {1: "HIGH 30-day risk", 0: "LOW 30-day risk"}


def _vitals_line(d: dict) -> str:
    sex = "male" if int(d["sex"]) == 1 else "female"
    return (
        f"{sex}, age {d['age']:.0f}; "
        f"BP {d['systolic_bp']:.0f}/{d['diastolic_bp']:.0f}; "
        f"HR {d['heart_rate']:.0f}; SpO2 {d['spo2']:.0f}%; "
        f"glucose {d['glucose']:.0f}; Cr {d['creatinine']:.2f}; "
        f"BMI {d['bmi']:.1f}; T {d['temperature']:.1f}°C; WBC {d['wbc']:.1f}"
    )


def format_card(cf: Counterfactual, patient_id: str | None = None) -> str:
    pid = patient_id or "patient"
    head = f"## {pid} — predicted {_LABEL_NAME[cf.original_label]} (p={cf.original_proba:.2f})"
    orig = (
        f"**Original**\n"
        f"- {_vitals_line(cf.original)}\n"
        f"- Notes: {cf.original['notes']}"
    )
    if cf.flipped:
        status = (
            f"**Counterfactual** (flips to {_LABEL_NAME[cf.counterfactual_label]}, "
            f"p_high={cf.counterfactual_proba:.2f})"
        )
    else:
        status = (
            f"**Counterfactual** (did not flip; still {_LABEL_NAME[cf.counterfactual_label]}, "
            f"p_high={cf.counterfactual_proba:.2f})"
        )
    if cf.changes:
        bullets = []
        for ch in cf.changes:
            feat = ch["feature"]
            if feat == "notes":
                bullets.append(f"- notes: “{ch['from']}” → “{ch['to']}”")
            else:
                bullets.append(f"- {feat}: {ch['from']} → {ch['to']}")
        change_md = "\n".join(bullets)
    else:
        change_md = "- (no recorded feature edits)"
    metrics = (
        f"**Cost** {cf.cost:.3f} · **proximity** {cf.proximity:.3f} · "
        f"**sparsity** {cf.sparsity}/{cf.n_features} features changed"
    )
    return "\n\n".join([head, orig, status, change_md, metrics])


def write_cards(
    cfs: list[Counterfactual],
    path: str | Path,
    patient_ids: list[str] | None = None,
    extra_preamble: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = patient_ids or [f"patient_{i}" for i in range(len(cfs))]
    parts = [
        "# Counterfactual cards",
        "",
        "Synthetic ward patients. Each card asks: *what is a small, mostly actionable",
        "change that would flip the model's 30-day risk call?* This is an explanation",
        "of the fitted classifier, not a care plan.",
        "",
    ]
    if extra_preamble:
        parts.extend([extra_preamble, ""])
    for pid, cf in zip(ids, cfs):
        parts.append(format_card(cf, patient_id=pid))
        parts.append("")
        parts.append("---")
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")
