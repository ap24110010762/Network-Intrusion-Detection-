"""
Phase 11 - Stratified 5-fold cross-validation.

The baseline scores in train.py each come from one train/test split. A single
split can be lucky or unlucky, so a 0.002 difference between two models means
nothing on its own. Cross-validation refits each model five times on different
slices of the training data and reports mean F1 with a standard deviation,
which shows whether a lead is real or noise.

Only the strongest candidates are cross-validated - the tree ensembles, which
were the only models to score above 0.95 F1 on both experiments. Running it on
Logistic Regression or SVM would cost time to confirm a result already known.

Cross-validation runs on the TRAINING split only. The test set stays untouched
until the final evaluation in Phase 14.

Usage (from the project root):
    python src/cross_validation.py
"""

import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_loader import RESULTS_DIR
from datasets import load_split
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save

TABLES_DIR = RESULTS_DIR / "tables"
SELECTED_FEATURES_PATH = TABLES_DIR / "selected_features.csv"

RANDOM_STATE = 42
N_FOLDS = 5


def candidate_models() -> dict:
    """The three tree-based models that led the baseline comparison."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=20
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.3,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def scoring_for(is_binary: bool) -> dict:
    """Metric names understood by cross_validate, matched to the task."""
    suffix = "" if is_binary else "_macro"
    return {
        "accuracy": "accuracy",
        "precision": f"precision{suffix}",
        "recall": f"recall{suffix}",
        "f1": f"f1{suffix}",
    }


def cross_validate_experiment(X: pd.DataFrame, y: pd.Series, is_binary: bool) -> pd.DataFrame:
    """Run stratified K-fold cross-validation for every candidate model."""
    label = "A - BINARY" if is_binary else "B - MULTI-CLASS"
    print(f"\n{'='*78}")
    print(f"EXPERIMENT {label}  |  {N_FOLDS}-fold stratified CV on {len(X):,} training rows")
    print(f"{'='*78}\n")

    # XGBoost needs integer labels; encoding once here keeps every fold
    # consistent and leaves the other models working on the original strings.
    encoder = None if is_binary else LabelEncoder().fit(y)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = scoring_for(is_binary)

    rows = []
    for name, model in candidate_models().items():
        fit_y = y if (is_binary or name != "XGBoost") else pd.Series(
            encoder.transform(y), index=y.index
        )

        print(f"  {name:15} running {N_FOLDS} folds ...", end="", flush=True)
        start = time.perf_counter()
        scores = cross_validate(
            model, X, fit_y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
        )
        elapsed = time.perf_counter() - start

        row = {"model": name, "total_seconds": round(elapsed, 1)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        row["f1_folds"] = np.round(scores["test_f1"], 5).tolist()
        rows.append(row)

        print(f" F1 {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}  ({elapsed:.0f}s)")

    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False)


def plot_stability(binary: pd.DataFrame, multi: pd.DataFrame) -> None:
    """
    Mean F1 with the spread across folds.

    The error bar is what this figure is for. Two models whose bars overlap
    are not meaningfully different, however their mean ranks them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
    panels = [
        (axes[0], binary, "Experiment A - BENIGN vs ATTACK", BLUE),
        (axes[1], multi, "Experiment B - 9 traffic classes", ORANGE),
    ]

    for ax, frame, title, color in panels:
        frame = frame.sort_values("f1_mean")
        y = np.arange(len(frame))

        ax.errorbar(
            frame["f1_mean"], y, xerr=frame["f1_std"], fmt="o",
            markersize=10, color=color, markeredgecolor="white",
            markeredgewidth=1.6, ecolor=GRID, elinewidth=3, capsize=5,
        )
        # Individual folds behind the summary, so the spread is visible
        # rather than only summarised.
        for i, folds in enumerate(frame["f1_folds"]):
            ax.scatter(folds, [i] * len(folds), s=14, color=GRID, zorder=0)

        for i, (mean, std) in enumerate(zip(frame["f1_mean"], frame["f1_std"])):
            ax.text(mean, i + 0.24, f"{mean:.4f} ± {std:.4f}",
                    ha="center", fontsize=8.5, color=INK)

        ax.set_yticks(y, frame["model"].tolist(), fontsize=9.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("F1 across 5 folds (higher is better)")
        ax.set_ylim(-0.6, len(frame) - 0.3)
        ax.grid(axis="y", visible=False)

    fig.suptitle(f"{N_FOLDS}-fold cross-validation: is the ranking stable?",
                 y=1.06, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.06,
             "Grey dots are the individual folds. Overlapping error bars mean "
             "the difference between two models is not meaningful.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, "10_cross_validation.png")


def per_class_stability(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Measure F1 per class per fold on the multi-class task.

    The multi-class fold spread is roughly 40x the binary one, and macro-F1
    weights every class equally regardless of size, so a small class moves the
    headline score as much as BENIGN's 265,465 rows.

    The result is not simply "rare classes are unstable". Heartbleed scores a
    perfect 1.0000 on every fold from 9 training rows, because its traffic is
    unmistakable. Infiltration, with three times as many rows, swings between
    0.667 and 0.923. What destabilises the score is a class being hard to
    separate, not merely small - and Bot, the weakest class at 0.78 mean F1,
    has 250 rows.
    """
    from sklearn.metrics import f1_score

    classes = np.array(sorted(y.unique()))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=20)

    print(f"\nPer-class F1 across {N_FOLDS} folds (Decision Tree):")
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores = f1_score(y.iloc[val_idx], preds, average=None,
                          labels=classes, zero_division=0)
        rows.append(dict(zip(classes, scores)))

    frame = pd.DataFrame(rows, index=[f"fold{i}" for i in range(1, N_FOLDS + 1)]).T
    frame["mean"] = frame.mean(axis=1)
    frame["spread"] = frame.iloc[:, :N_FOLDS].max(axis=1) - frame.iloc[:, :N_FOLDS].min(axis=1)
    frame["train_rows"] = y.value_counts().reindex(frame.index)
    frame = frame.sort_values("train_rows", ascending=False)

    print(frame.round(4).to_string())
    frame.to_csv(TABLES_DIR / "per_class_stability.csv")
    return frame


def plot_per_class_stability(frame: pd.DataFrame) -> None:
    """Show F1 spread against class size - the source of the instability."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = np.arange(len(frame))[::-1]

    for i, row in zip(y, frame.itertuples()):
        folds = [getattr(row, f"fold{k}") for k in range(1, N_FOLDS + 1)]
        ax.hlines(i, min(folds), max(folds), color=GRID, linewidth=3, zorder=1)
        ax.scatter(folds, [i] * len(folds), s=22, color=GRID, zorder=2)
        ax.scatter(row.mean, i, s=95, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=1.6)
        ax.text(1.04, i, f"{int(row.train_rows):,} rows", va="center",
                fontsize=8.5, color=INK_SOFT)

    ax.set_yticks(y, frame.index.tolist(), fontsize=9.5)
    ax.set_xlim(-0.05, 1.28)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("F1 per fold (grey) and mean (blue)")
    ax.set_title("Hard-to-separate classes, not small ones, destabilise the score")
    ax.grid(axis="y", visible=False)

    fig.text(0.5, -0.03,
             "Heartbleed scores 1.0000 on every fold from 9 rows. Infiltration, with 29, "
             "swings from 0.667 to 0.923.\nMacro-F1 weights all 9 classes equally, so that "
             "swing alone moves the headline score.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, "11_per_class_stability.png")


def report(frame: pd.DataFrame, label: str) -> None:
    """Print the fold-by-fold summary and check whether the lead is real."""
    print(f"\n{label}")
    columns = ["model", "f1_mean", "f1_std", "accuracy_mean",
               "precision_mean", "recall_mean", "total_seconds"]
    print(frame[columns].round(5).to_string(index=False))

    best, second = frame.iloc[0], frame.iloc[1]
    margin = best["f1_mean"] - second["f1_mean"]
    combined_spread = best["f1_std"] + second["f1_std"]

    print(f"\n  {best['model']} leads {second['model']} by {margin:.5f} F1.")
    if margin > combined_spread:
        print("  The margin is larger than the combined fold spread, "
              "so the lead is meaningful.")
    else:
        print(f"  The combined fold spread is {combined_spread:.5f}, which is "
              "larger than the margin,\n  so these two models are statistically "
              "indistinguishable here - prefer the cheaper one.")


def main() -> None:
    apply_style()

    split = load_split()
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()
    X = split["X_train"][features]

    binary = cross_validate_experiment(X, split["y_train_binary"], is_binary=True)
    report(binary, "Experiment A - BINARY")
    binary.to_csv(TABLES_DIR / "cross_validation_binary.csv", index=False)

    multi = cross_validate_experiment(X, split["y_train_multi"], is_binary=False)
    report(multi, "Experiment B - MULTI-CLASS")
    multi.to_csv(TABLES_DIR / "cross_validation_multi.csv", index=False)

    print("\nBuilding figure:")
    plot_stability(binary, multi)


if __name__ == "__main__":
    main()
