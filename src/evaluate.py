"""
Phase 10 - Model comparison figures.

Reads the baseline results written by train.py and produces the comparison
figures for the report.

    08_accuracy_vs_f1.png   why accuracy alone is misleading here
    09_model_cost.png       training and prediction cost per model

Usage (from the project root):
    python src/evaluate.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import RESULTS_DIR
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save

TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_PATH = TABLES_DIR / "baseline_models_all.csv"

EXPERIMENTS = [
    ("A (binary)", "Experiment A — BENIGN vs ATTACK"),
    ("B (multi)", "Experiment B — 9 traffic classes"),
]


def plot_accuracy_vs_f1(results: pd.DataFrame) -> None:
    """
    Dumbbell chart: accuracy and F1 for each model, joined by a line.

    The length of the line is the point of the figure. A model with high
    accuracy and low F1 is not doing well on 83%-benign data - it is mostly
    predicting 'BENIGN' and being rewarded for it. Accuracy and F1 are both
    scores on the same 0-1 scale, so showing them on one axis is a fair
    comparison rather than a dual-axis trick.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)

    for ax, (experiment, title) in zip(axes, EXPERIMENTS):
        subset = (results[results["experiment"] == experiment]
                  .sort_values("f1")
                  .reset_index(drop=True))
        y = np.arange(len(subset))

        ax.hlines(y, subset["f1"], subset["accuracy"],
                  color=GRID, linewidth=2.5, zorder=1)
        ax.scatter(subset["accuracy"], y, s=85, color=ORANGE, zorder=2,
                   edgecolor="white", linewidth=1.5, label="Accuracy")
        ax.scatter(subset["f1"], y, s=85, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=1.5, label="F1 (macro)")

        for i, row in subset.iterrows():
            gap = row["accuracy"] - row["f1"]
            if gap > 0.05:
                ax.text((row["f1"] + row["accuracy"]) / 2, i + 0.28,
                        f"gap {gap:.2f}", ha="center", fontsize=7.5,
                        color=INK_SOFT)

        ax.set_yticks(y, subset["model"].tolist(), fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score")
        ax.set_xlim(0.3, 1.04)
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("High accuracy can hide a model that detects almost nothing",
                 y=1.15, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.06,
             "Every model scores above 0.92 accuracy because 83% of traffic is benign. "
             "F1 shows which ones actually find attacks.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, "08_accuracy_vs_f1.png")


def plot_cost(results: pd.DataFrame) -> None:
    """
    Training and prediction cost.

    Two panels rather than one chart with two scales: a model can be cheap to
    train and expensive to predict, which is the interesting case here, and
    the two costs matter at different times in a real deployment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    panels = [
        ("train_seconds", "Training time (seconds)", BLUE),
        ("predict_seconds", "Prediction time on 80,008 test flows (seconds)", ORANGE),
    ]

    order = (results[results["experiment"] == "B (multi)"]
             .sort_values("train_seconds")["model"].tolist())

    for ax, (column, label, color) in zip(axes, panels):
        pivot = (results.pivot(index="model", columns="experiment", values=column)
                 .reindex(order))
        y = np.arange(len(pivot))

        for offset, experiment, shade in [(-0.18, "A (binary)", color),
                                          (0.18, "B (multi)", GRID)]:
            bars = ax.barh(y + offset, pivot[experiment], height=0.34,
                           color=shade, label=experiment,
                           edgecolor="white", linewidth=0.5)
            for bar, value in zip(bars, pivot[experiment]):
                ax.text(value + pivot.to_numpy().max() * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}", va="center", fontsize=7.5, color=INK)

        ax.set_yticks(y, pivot.index.tolist(), fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, pivot.to_numpy().max() * 1.18)
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Cost is not symmetric: KNN trains instantly but predicts slowly",
                 y=1.13, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.05,
             "SVM and KNN were trained on reduced sets (30,000 and 50,000 rows); "
             "all models predict on the same 80,008 test flows.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, "09_model_cost.png")


def summarise(results: pd.DataFrame) -> None:
    """Print the comparison table and call out the accuracy/F1 gap."""
    for experiment, title in EXPERIMENTS:
        subset = results[results["experiment"] == experiment].copy()
        subset["accuracy_minus_f1"] = (subset["accuracy"] - subset["f1"]).round(4)
        # The Windows console uses cp1252, which cannot encode an em dash, so
        # the title is flattened to ASCII for printing. The figures keep it.
        print(f"\n{title.replace(chr(8212), '-')}")
        print(subset[["model", "accuracy", "f1", "accuracy_minus_f1",
                      "precision", "recall"]].round(4).to_string(index=False))

    worst = results.loc[(results["accuracy"] - results["f1"]).idxmax()]
    print(f"\nLargest accuracy/F1 gap: {worst['model']} in {worst['experiment']} "
          f"- accuracy {worst['accuracy']:.4f} but F1 {worst['f1']:.4f}")


def main() -> None:
    apply_style()

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run 'python src/train.py' first."
        )

    results = pd.read_csv(RESULTS_PATH)
    summarise(results)

    print("\nBuilding figures:")
    plot_accuracy_vs_f1(results)
    plot_cost(results)


if __name__ == "__main__":
    main()
