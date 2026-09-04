"""
Phase 9-10 - Baseline models and model comparison.

Trains six classical algorithms on both experiments:

    Experiment A (binary)      BENIGN vs ATTACK
    Experiment B (multi-class) the 9 traffic classes

and scores every one of them on the same held-out test set using accuracy,
precision, recall, F1, ROC-AUC, PR-AUC, training time and prediction time.

Two models get a reduced training set, recorded in the results table:

    SVM  - training cost grows with the square of the sample size, so 320,032
           rows is not feasible on 4 CPU cores.
    KNN  - trains instantly but predicts slowly, since every test row is
           compared against every training row.

Both are still scored on the full test set, so the comparison stays fair on
the measurement that matters. Scale-sensitive models (Logistic Regression,
KNN, SVM) are wrapped in a Pipeline with StandardScaler, so scaling is fitted
on training folds only and never sees the test set.

Usage (from the project root):
    python src/train.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_loader import RESULTS_DIR
from datasets import load_split

TABLES_DIR = RESULTS_DIR / "tables"
SELECTED_FEATURES_PATH = TABLES_DIR / "selected_features.csv"

RANDOM_STATE = 42

# Training-set caps for the two algorithms that cannot take the full split.
# Both still predict on the complete test set.
TRAIN_CAPS = {"SVM (RBF)": 30_000, "KNN": 50_000}


def build_models() -> dict:
    """The six algorithms named in the project document."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=20
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.3,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def cap_training_set(X: pd.DataFrame, y: pd.Series, cap: int | None):
    """Stratified reduction of the training set for the slow algorithms."""
    if cap is None or len(X) <= cap:
        return X, y, len(X)

    rate = cap / len(X)
    parts = []
    for label in pd.unique(y):
        idx = np.flatnonzero((y == label).to_numpy())
        # Keep at least one row of every class, however rare.
        take = max(1, int(round(len(idx) * rate)))
        rng = np.random.default_rng(RANDOM_STATE)
        parts.append(rng.choice(idx, size=min(take, len(idx)), replace=False))

    keep = np.sort(np.concatenate(parts))
    return X.iloc[keep], y.iloc[keep], len(keep)


def probability_scores(model, X: pd.DataFrame):
    """
    Return calibrated class probabilities, or None when unavailable.

    ROC-AUC and PR-AUC need a score per class that behaves like a probability.
    SVC without probability=True offers only decision_function, whose
    multi-class form is a one-vs-one voting margin that does not sum to 1, so
    the AUC metrics are left blank for that model rather than computed from a
    quantity they do not apply to. Enabling probability=True would trigger an
    internal cross-validation that is far too slow here.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except (AttributeError, NotImplementedError):
            return None
    return None


def evaluate(y_true, y_pred, proba, classes, is_binary: bool) -> dict:
    """Compute every metric the project document asks for."""
    average = "binary" if is_binary else "macro"
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if proba is None:
        return scores

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if is_binary:
                scores["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
                scores["pr_auc"] = average_precision_score(y_true, proba[:, 1])
            else:
                scores["roc_auc"] = roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro", labels=classes
                )
                # One-hot the truth so average_precision can be macro-averaged
                # over the classes the same way ROC-AUC is.
                onehot = np.zeros_like(proba)
                index = {c: i for i, c in enumerate(classes)}
                for row, label in enumerate(y_true):
                    onehot[row, index[label]] = 1
                scores["pr_auc"] = average_precision_score(
                    onehot, proba, average="macro"
                )
        except ValueError:
            # A class absent from the test set makes these undefined.
            pass

    return scores


def run_experiment(split: dict, features: list[str], target: str) -> pd.DataFrame:
    """Train and score all six models for one experiment."""
    is_binary = target == "binary"
    X_train = split["X_train"][features]
    X_test = split["X_test"][features]
    y_train = split[f"y_train_{target}"]
    y_test = split[f"y_test_{target}"]

    # XGBoost needs integer class labels, so encode for the multi-class task.
    encoder = None
    if not is_binary:
        encoder = LabelEncoder().fit(pd.concat([y_train, y_test]))

    print(f"\n{'='*78}")
    print(f"EXPERIMENT {'A - BINARY (BENIGN vs ATTACK)' if is_binary else 'B - MULTI-CLASS (9 classes)'}")
    print(f"{'='*78}")
    print(f"train {len(X_train):,} | test {len(X_test):,} | features {len(features)}\n")

    rows = []
    for name, model in build_models().items():
        cap = TRAIN_CAPS.get(name)
        Xt, yt, n_used = cap_training_set(X_train, y_train, cap)

        fit_y = yt
        if not is_binary and name == "XGBoost":
            fit_y = pd.Series(encoder.transform(yt), index=yt.index)

        print(f"  {name:20} training on {n_used:>7,} rows ...", end="", flush=True)
        start = time.perf_counter()
        model.fit(Xt, fit_y)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        preds = model.predict(X_test)
        predict_seconds = time.perf_counter() - start

        if not is_binary and name == "XGBoost":
            preds = encoder.inverse_transform(preds)

        classes = np.unique(y_test) if not is_binary else np.array([0, 1])
        proba = probability_scores(model, X_test)
        if proba is not None and not is_binary and name == "XGBoost":
            # Reorder XGBoost's encoded columns to match the label order used
            # by the other models, so the AUC metrics stay comparable.
            order = [list(encoder.classes_).index(c) for c in classes]
            proba = proba[:, order]

        scores = evaluate(y_test, preds, proba, classes, is_binary)
        scores.update({
            "model": name,
            "train_rows": n_used,
            "train_seconds": round(train_seconds, 2),
            "predict_seconds": round(predict_seconds, 2),
        })
        rows.append(scores)

        print(f" F1 {scores['f1']:.4f} | acc {scores['accuracy']:.4f} | "
              f"{train_seconds:6.1f}s fit | {predict_seconds:5.1f}s predict")

    results = pd.DataFrame(rows)
    results = results[[
        "model", "train_rows", "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "train_seconds", "predict_seconds",
    ]].sort_values("f1", ascending=False)

    return results


def main() -> None:
    split = load_split()

    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{SELECTED_FEATURES_PATH} not found. "
            "Run 'python src/feature_selection.py' first."
        )
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()

    all_results = []
    for target in ("binary", "multi"):
        results = run_experiment(split, features, target)
        results.insert(1, "experiment", "A (binary)" if target == "binary" else "B (multi)")

        print(f"\nRanked by F1:")
        print(results.round(4).to_string(index=False))

        path = TABLES_DIR / f"baseline_models_{target}.csv"
        results.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(TABLES_DIR / "baseline_models_all.csv", index=False)
    print(f"\nSaved combined results -> {TABLES_DIR / 'baseline_models_all.csv'}")


if __name__ == "__main__":
    main()
