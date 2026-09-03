"""Train a ward-risk model, explain five patients, write cards and a scatter plot."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.counterfactual import search_many
from src.evaluate import summarize
from src.model import heldout_auc, predict_row, train_model
from src.patients import dataframe_for_model, generate_cohort, load_vignettes
from src.report import write_cards

OUT = ROOT / "outputs"
SEED = 42


def plot_proximity_sparsity(cfs, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = [c.proximity for c in cfs]
    ys = [c.sparsity for c in cfs]
    colors = ["#2a9d8f" if c.flipped else "#e76f51" for c in cfs]
    ax.scatter(xs, ys, c=colors, s=42, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("proximity (normalized L1, lower = closer)")
    ax.set_ylabel("sparsity (features changed)")
    ax.set_title("Counterfactuals: proximity vs sparsity")
    ax.scatter([], [], c="#2a9d8f", label="flipped", edgecolors="k")
    ax.scatter([], [], c="#e76f51", label="did not flip", edgecolors="k")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pick_demo_rows(vignettes: pd.DataFrame, model, n: int = 5) -> pd.DataFrame:
    """Prefer a mix of predicted high and low risk from the shipped vignettes."""
    labels = [predict_row(model, row) for _, row in vignettes.iterrows()]
    df = vignettes.copy()
    df["_pred"] = labels
    high = df[df["_pred"] == 1]
    low = df[df["_pred"] == 0]
    take_high = high.head(3)
    take_low = low.head(2)
    picked = pd.concat([take_high, take_low], axis=0)
    if len(picked) < n:
        extra = df.drop(index=picked.index).head(n - len(picked))
        picked = pd.concat([picked, extra], axis=0)
    return picked.head(n).drop(columns=["_pred"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vignettes = load_vignettes()
    cohort = generate_cohort(n=420, seed=SEED)
    X, y = dataframe_for_model(cohort)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    model = train_model(X_tr, y_tr, seed=SEED)
    auc = heldout_auc(model, X_te, y_te)
    print(f"Held-out AUC: {auc:.3f}")
    print(f"Train n={len(X_tr)}  test n={len(X_te)}  positive rate={y.mean():.2f}")

    demo_rows = pick_demo_rows(vignettes, model, n=5)
    cards = search_many(model, demo_rows, seed=SEED)
    ids = demo_rows["patient_id"].tolist()
    write_cards(
        cards,
        OUT / "cf_cards.md",
        patient_ids=ids,
        extra_preamble=f"Classifier held-out AUC on a larger synthetic draw: {auc:.3f}. Seed=42.",
    )

    # Broader cloud for the scatter (vignettes + a slice of test patients).
    cloud_src = pd.concat(
        [vignettes.head(8), cohort.sample(n=10, random_state=SEED)],
        ignore_index=True,
    )
    cloud = search_many(model, cloud_src, seed=SEED + 11)
    plot_proximity_sparsity(cloud, OUT / "proximity_vs_sparsity.png")
    stats = summarize(cloud)
    print("Batch CF metrics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    print(f"Wrote {OUT / 'cf_cards.md'} and {OUT / 'proximity_vs_sparsity.png'}")
    for pid, cf in zip(ids, cards):
        print(f"  {pid}: pred={cf.original_label} -> {cf.counterfactual_label} flipped={cf.flipped} sparsity={cf.sparsity}")


if __name__ == "__main__":
    main()
