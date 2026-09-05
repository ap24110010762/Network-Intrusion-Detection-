"""
Phase 14 - Final evaluation on the held-out test set.

This is the only place the test set is scored. Everything before it - feature
selection, model choice, resampling strategy, hyperparameters - was decided
using cross-validation on the training split, so the numbers this module
produces are an honest estimate of performance on traffic the pipeline has
never seen.

The final model is XGBoost for both experiments:

    Experiment A  it leads cross-validation at 0.99678 +/- 0.00059 and
                  predicts 80,008 flows in 0.2s against SVM's 56.8s. For a
                  system watching live traffic, prediction latency is a
                  requirement rather than a tiebreak.

    Experiment B  the three tree models are statistically indistinguishable
                  under cross-validation, so the choice falls to cost and
                  consistency rather than a decimal place.

Tuned hyperparameters are loaded from results/tables/best_params.json when it
exists. Phase 13 found the gains to be smaller than the fold spread, so the
tuned and default settings are both scored here and reported side by side.

Usage (from the project root):
    python src/final_evaluation.py
"""

import json
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from data_loader import RESULTS_DIR
from datasets import load_split
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save

TABLES_DIR = RESULTS_DIR / "tables"
SELECTED_FEATURES_PATH = TABLES_DIR / "selected_features.csv"
BEST_PARAMS_PATH = TABLES_DIR / "best_params.json"

RANDOM_STATE = 42

# The Phase 9 settings, used when no tuned parameters are available and as the
# comparison point when they are.
DEFAULT_PARAMS = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.3}

# Sequential ramp for the confusion matrices: one hue, light to dark.
CONF_CMAP = LinearSegmentedColormap.from_list("blues", ["#ffffff", BLUE])


def load_best_params(target: str) -> dict | None:
    """Tuned hyperparameters for one experiment, if Phase 13 has run."""
    if not BEST_PARAMS_PATH.exists():
        return None
    params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return params.get(target, {}).get("XGBoost")


def build_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
        verbosity=0, **params,
    )


def score(y_true, y_pred, proba, classes, is_binary: bool) -> dict:
    """Every metric the project document asks for, at full precision."""
    average = "binary" if is_binary else "macro"
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if is_binary:
        result["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        result["pr_auc"] = average_precision_score(y_true, proba[:, 1])
    else:
        result["roc_auc"] = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=classes
        )
        onehot = np.zeros_like(proba)
        index = {c: i for i, c in enumerate(classes)}
        for row, label in enumerate(y_true):
            onehot[row, index[label]] = 1
        result["pr_auc"] = average_precision_score(onehot, proba, average="macro")

    return result


def plot_confusion(cm: np.ndarray, classes, title: str, filename: str) -> None:
    """
    Confusion matrix with counts and row-normalised shading.

    Shading is normalised per true class because the classes differ in size by
    five orders of magnitude; raw counts would leave every row except BENIGN
    invisible. Each cell still prints its actual count.
    """
    normalised = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(classes)
    size = max(5.5, 0.85 * n + 2.5)

    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    ax.imshow(normalised, cmap=CONF_CMAP, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if count == 0:
                text = "·"
            elif count >= 10_000:
                text = f"{count/1000:.0f}k"
            else:
                text = f"{count:,}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8.5 if n > 4 else 11,
                    color="white" if normalised[i, j] > 0.55 else INK)

    ax.set_xticks(range(n), classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)

    fig.text(0.5, 0.005,
             "Cells show counts; shading is normalised within each row, "
             "so a full-strength diagonal cell means that class was almost always right.",
             ha="center", fontsize=8, color=INK_SOFT)
    save(fig, filename)


def plot_curves(y_true, proba, filename: str) -> None:
    """ROC and precision-recall curves for the binary task."""
    fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
    precision, recall, _ = precision_recall_curve(y_true, proba[:, 1])
    roc_auc = roc_auc_score(y_true, proba[:, 1])
    pr_auc = average_precision_score(y_true, proba[:, 1])
    positive_rate = np.mean(y_true)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=GRID)
    ax1.plot(fpr, tpr, linewidth=2, color=BLUE)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"ROC curve — AUC {roc_auc:.5f}", fontsize=11)
    ax1.text(0.55, 0.28, "a random classifier\nfollows the dashed line",
             fontsize=8.5, color=INK_SOFT)

    ax2.axhline(positive_rate, linestyle="--", linewidth=1.2, color=GRID)
    ax2.plot(recall, precision, linewidth=2, color=ORANGE)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-recall curve — AP {pr_auc:.5f}", fontsize=11)
    ax2.text(0.05, positive_rate + 0.04,
             f"baseline {positive_rate:.3f} = the attack rate",
             fontsize=8.5, color=INK_SOFT)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle("Experiment A — final model on the held-out test set",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.04,
             "The precision-recall curve is the more informative of the two on "
             "imbalanced data: its baseline sits at the attack rate, not at 0.5.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, filename)


def plot_per_class(report_df: pd.DataFrame, support: pd.Series) -> None:
    """Per-class precision, recall and F1 on the test set."""
    frame = report_df.copy().sort_values("support")
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y - 0.22, frame["precision"], height=0.4, color=BLUE, label="Precision")
    ax.barh(y + 0.22, frame["recall"], height=0.4, color=ORANGE, label="Recall")

    for i, row in enumerate(frame.itertuples()):
        ax.text(row.precision + 0.01, i - 0.22, f"{row.precision:.3f}",
                va="center", fontsize=8, color=INK)
        ax.text(row.recall + 0.01, i + 0.22, f"{row.recall:.3f}",
                va="center", fontsize=8, color=INK)
        ax.text(1.17, i, f"{int(row.support):,} test flows",
                va="center", fontsize=8, color=INK_SOFT)

    ax.set_yticks(y, frame.index.tolist(), fontsize=9.5)
    ax.set_xlim(0, 1.42)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Score on the held-out test set")
    ax.set_title("Experiment B — per-class performance")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "17_final_per_class.png")


def run(split: dict, features: list[str], target: str) -> pd.DataFrame:
    """Refit on the full training split and score once on the test set."""
    is_binary = target == "binary"
    label = "A - BINARY" if is_binary else "B - MULTI-CLASS"

    X_train = split["X_train"][features]
    X_test = split["X_test"][features]
    y_train = split[f"y_train_{target}"]
    y_test = split[f"y_test_{target}"]

    print(f"\n{'='*78}")
    # ASCII in console output: the Windows console is cp1252 and cannot encode
    # an em dash. Figure titles keep the typographic version.
    print(f"FINAL EVALUATION - EXPERIMENT {label}")
    print(f"{'='*78}")
    print(f"refit on {len(X_train):,} rows | score once on {len(X_test):,} held-out rows\n")

    encoder = None if is_binary else LabelEncoder().fit(pd.concat([y_train, y_test]))
    classes = np.array([0, 1]) if is_binary else np.array(sorted(y_test.unique()))

    tuned = load_best_params(target)
    settings = {"default": DEFAULT_PARAMS}
    if tuned:
        settings["tuned"] = tuned
    else:
        print("  (no best_params.json yet — scoring default settings only)\n")

    rows = {}
    for name, params in settings.items():
        model = build_model(params)
        fit_y = y_train if is_binary else pd.Series(
            encoder.transform(y_train), index=y_train.index
        )

        start = time.perf_counter()
        model.fit(X_train, fit_y)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        preds = model.predict(X_test)
        predict_seconds = time.perf_counter() - start
        proba = model.predict_proba(X_test)

        if not is_binary:
            preds = encoder.inverse_transform(preds)
            order = [list(encoder.classes_).index(c) for c in classes]
            proba = proba[:, order]

        metrics = score(y_test, preds, proba, classes, is_binary)
        metrics.update({"fit_seconds": round(fit_seconds, 2),
                        "predict_seconds": round(predict_seconds, 2)})
        rows[name] = metrics

        print(f"  {name:8} settings")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            print(f"    {key:10} {metrics[key]:.6f}")
        print(f"    fit {fit_seconds:.1f}s | predict {predict_seconds:.2f}s "
              f"on {len(X_test):,} flows\n")

        # Figures and per-class detail come from the settings actually shipped.
        if name == list(settings)[-1]:
            cm = confusion_matrix(y_test, preds, labels=classes)
            names = ["BENIGN", "ATTACK"] if is_binary else classes
            plot_confusion(
                cm, names,
                f"Experiment {'A' if is_binary else 'B'} — confusion matrix "
                f"({len(X_test):,} test flows)",
                f"{'14' if is_binary else '15'}_confusion_{target}.png",
            )
            if is_binary:
                plot_curves(y_test, proba, "16_roc_pr_curves.png")
            else:
                report = classification_report(
                    y_test, preds, labels=classes, output_dict=True,
                    zero_division=0,
                )
                report_df = pd.DataFrame(
                    {c: report[c] for c in classes}
                ).T[["precision", "recall", "f1-score", "support"]]
                report_df.to_csv(TABLES_DIR / "final_per_class_report.csv")
                print(report_df.round(4).to_string())
                plot_per_class(report_df, report_df["support"])

    frame = pd.DataFrame(rows).T
    frame.index.name = "settings"
    return frame


def main() -> None:
    apply_style()

    split = load_split()
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()

    results = []
    for target in ("binary", "multi"):
        frame = run(split, features, target)
        frame.insert(0, "experiment", "A (binary)" if target == "binary" else "B (multi)")
        results.append(frame.reset_index())

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(TABLES_DIR / "final_evaluation.csv", index=False)
    print(f"\nSaved -> {TABLES_DIR / 'final_evaluation.csv'}")

    print("\nNOTE: these are the only numbers produced by scoring the test set. "
          "Every earlier\nchoice was made on cross-validation over the training "
          "split alone.")


if __name__ == "__main__":
    main()
