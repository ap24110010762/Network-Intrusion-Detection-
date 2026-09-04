"""
Phase 12 - Handling class imbalance.

Compares four strategies for the 83%-benign dataset:

    baseline        train on the data as it is
    class weights   penalise mistakes on rare classes more heavily
    SMOTE           synthesise new minority rows
    undersampling   discard majority rows

Two rules drive the design:

1. SMOTE must never touch validation data. Oversampling before splitting
   copies synthetic rows derived from validation rows into training, which
   inflates every score. Here SMOTE lives inside an imblearn Pipeline, so
   cross_validate applies it to each training fold only.

2. Strategy selection uses cross-validation on the training split, not the
   test set. Choosing a strategy by its test score would make the final test
   result optimistic, since the test set would have influenced the choice.

SMOTE is applied to the binary task only. On the multi-class task it cannot
run: SMOTE interpolates between a rare row and its k nearest neighbours of the
same class, and Heartbleed has 9 rows in the training split - fewer than the
neighbours the algorithm needs once a fold is carved out. That limitation is
itself a finding, so the multi-class comparison covers the strategies that do
apply.

Usage (from the project root):
    python src/imbalance.py
"""

import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_loader import RESULTS_DIR
from datasets import load_split
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save

TABLES_DIR = RESULTS_DIR / "tables"
SELECTED_FEATURES_PATH = TABLES_DIR / "selected_features.csv"

RANDOM_STATE = 42
N_FOLDS = 3

# SMOTE inflates a fold to roughly twice the majority class size, so the
# comparison runs on a subsample of the training split. Random Forest also
# uses fewer trees here than in the final model: the goal is the relative
# ranking of four strategies, not a final score.
SUBSAMPLE = 150_000
N_ESTIMATORS = 60


def base_model(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=20, n_jobs=-1,
        random_state=RANDOM_STATE, class_weight=class_weight,
    )


def cap_majority(y) -> dict:
    """
    Undersample only the majority class, down to 3x the next largest.

    RandomUnderSampler's default reduces EVERY class to the size of the
    smallest one. With Heartbleed at 4 rows in a fold, that leaves about 36
    training rows in total and the model collapses - which measures the
    default setting, not undersampling as a technique. Capping the majority
    instead is what the method is actually for: it removes the excess benign
    rows while leaving every attack class intact.

    Passed to RandomUnderSampler as a callable so the targets are recomputed
    for each cross-validation fold rather than fixed from the whole split.
    """
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return {}
    majority = counts.index[0]
    target = min(int(counts.iloc[0]), int(counts.iloc[1]) * 3)
    return {majority: target}


def strategies(is_binary: bool) -> dict:
    """The strategies to compare, each as a ready-to-fit estimator."""
    options = {
        "baseline": base_model(),
        "class weights": base_model(class_weight="balanced"),
        # Kept alongside the capped version because the contrast between them
        # is the finding: the same technique either helps or destroys the
        # model depending purely on how far it is taken.
        "undersample (to minority)": ImbPipeline([
            ("resample", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("model", base_model()),
        ]),
        "undersample (majority only)": ImbPipeline([
            ("resample", RandomUnderSampler(
                sampling_strategy=cap_majority, random_state=RANDOM_STATE)),
            ("model", base_model()),
        ]),
    }

    if is_binary:
        # k_neighbors must be smaller than the count of the rarest class in
        # any training fold; the binary minority class has tens of thousands
        # of rows, so the default is safe.
        options["SMOTE"] = ImbPipeline([
            ("resample", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("model", base_model()),
        ])

    return options


def subsample(X: pd.DataFrame, y: pd.Series, n: int):
    """Stratified subsample that keeps every class present."""
    if len(X) <= n:
        return X, y

    rate = n / len(X)
    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for label in pd.unique(y):
        idx = np.flatnonzero((y == label).to_numpy())
        take = max(N_FOLDS, int(round(len(idx) * rate)))  # keep folds feasible
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))

    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep], y.iloc[keep]


def compare(X: pd.DataFrame, y: pd.Series, is_binary: bool) -> pd.DataFrame:
    """Cross-validate every applicable strategy on the training split."""
    label = "A - BINARY" if is_binary else "B - MULTI-CLASS"
    Xs, ys = subsample(X, y, SUBSAMPLE)

    print(f"\n{'='*78}")
    print(f"EXPERIMENT {label}  |  {N_FOLDS}-fold CV on {len(Xs):,} rows")
    print(f"{'='*78}")
    print("\nClass counts in this subsample:")
    print(ys.value_counts().to_string())
    print()

    suffix = "" if is_binary else "_macro"
    scoring = {
        "accuracy": "accuracy",
        "precision": f"precision{suffix}",
        "recall": f"recall{suffix}",
        "f1": f"f1{suffix}",
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, estimator in strategies(is_binary).items():
        print(f"  {name:15} ...", end="", flush=True)
        start = time.perf_counter()
        scores = cross_validate(
            estimator, Xs, ys, cv=cv, scoring=scoring, n_jobs=1,
            error_score="raise",
        )
        elapsed = time.perf_counter() - start

        row = {"strategy": name, "seconds": round(elapsed, 1)}
        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)

        print(f" F1 {row['f1_mean']:.4f} +/- {row['f1_std']:.4f} | "
              f"recall {row['recall_mean']:.4f} | "
              f"precision {row['precision_mean']:.4f}  ({elapsed:.0f}s)")

    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False)


def plot_strategies(binary: pd.DataFrame, multi: pd.DataFrame) -> None:
    """
    Precision and recall per strategy, side by side with F1.

    Precision and recall are shown together because that is where the
    strategies differ: resampling usually buys recall by giving up precision.
    F1 alone hides that trade.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panels = [
        (axes[0], binary, "Experiment A - BENIGN vs ATTACK"),
        (axes[1], multi, "Experiment B - 9 traffic classes"),
    ]

    for ax, frame, title in panels:
        frame = frame.sort_values("f1_mean")
        y = np.arange(len(frame))

        ax.barh(y - 0.2, frame["precision_mean"], height=0.36, color=BLUE,
                label="Precision")
        ax.barh(y + 0.2, frame["recall_mean"], height=0.36, color=ORANGE,
                label="Recall")

        for i, row in enumerate(frame.itertuples()):
            ax.text(row.precision_mean + 0.008, i - 0.2, f"{row.precision_mean:.3f}",
                    va="center", fontsize=8, color=INK)
            ax.text(row.recall_mean + 0.008, i + 0.2, f"{row.recall_mean:.3f}",
                    va="center", fontsize=8, color=INK)
            ax.text(1.16, i, f"F1 {row.f1_mean:.3f}", va="center",
                    fontsize=8.5, color=INK_SOFT)

        ax.set_yticks(y, frame["strategy"].tolist(), fontsize=9.5)
        ax.set_xlim(0, 1.32)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score")
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Resampling trades precision for recall",
                 y=1.13, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.06,
             "SMOTE is absent from Experiment B: Heartbleed has 9 training rows, "
             "too few neighbours to interpolate between.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, "12_class_imbalance.png")


def report(frame: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(frame[["strategy", "f1_mean", "f1_std", "precision_mean",
                 "recall_mean", "accuracy_mean", "seconds"]]
          .round(5).to_string(index=False))

    best = frame.iloc[0]
    baseline = frame[frame["strategy"] == "baseline"].iloc[0]
    delta = best["f1_mean"] - baseline["f1_mean"]
    if best["strategy"] == "baseline":
        print("\n  No resampling strategy beat the untouched data.")
    else:
        print(f"\n  {best['strategy']} beats baseline by {delta:+.5f} F1 "
              f"(recall {baseline['recall_mean']:.4f} -> {best['recall_mean']:.4f}, "
              f"precision {baseline['precision_mean']:.4f} -> {best['precision_mean']:.4f})")


def main() -> None:
    apply_style()

    split = load_split()
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()
    X = split["X_train"][features]

    binary = compare(X, split["y_train_binary"], is_binary=True)
    report(binary, "Experiment A - BINARY")
    binary.to_csv(TABLES_DIR / "imbalance_binary.csv", index=False)

    multi = compare(X, split["y_train_multi"], is_binary=False)
    report(multi, "Experiment B - MULTI-CLASS")
    multi.to_csv(TABLES_DIR / "imbalance_multi.csv", index=False)

    print("\nBuilding figure:")
    plot_strategies(binary, multi)


if __name__ == "__main__":
    main()
