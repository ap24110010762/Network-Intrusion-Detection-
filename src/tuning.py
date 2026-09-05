"""
Phase 13 - Hyperparameter optimisation.

Tunes Random Forest and XGBoost with RandomizedSearchCV on both experiments,
and compares each tuned model against the untuned defaults used in Phase 9.

Randomised search rather than grid search: a full grid over these ranges is
thousands of combinations, and randomised search reaches a comparable result
from a small sample of them because only a few hyperparameters usually matter.

Two concessions to a 4-core machine, both recorded in the results:

    - the search runs on a subsample of the training split
    - 3-fold CV and 15 candidates per model

The winning parameters are saved to results/tables/best_params.json so Phase
14 can refit on the FULL training split before the single, final scoring run
against the held-out test set.

Usage (from the project root):
    python src/tuning.py
"""

import json
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from data_loader import RESULTS_DIR
from datasets import load_split
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save

TABLES_DIR = RESULTS_DIR / "tables"
SELECTED_FEATURES_PATH = TABLES_DIR / "selected_features.csv"
BEST_PARAMS_PATH = TABLES_DIR / "best_params.json"

RANDOM_STATE = 42
N_FOLDS = 3
N_CANDIDATES = 15
SUBSAMPLE = 150_000


def search_spaces() -> dict:
    """
    The search space for each model.

    The ranges cover the hyperparameters named in the project document, plus
    the sampling parameters that matter most for XGBoost.
    """
    return {
        "Random Forest": (
            RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
            {
                "n_estimators": randint(100, 400),
                "max_depth": [10, 15, 20, 30, None],
                "min_samples_split": randint(2, 12),
                "min_samples_leaf": randint(1, 6),
                "max_features": ["sqrt", "log2", 0.4],
            },
            # The Phase 9 defaults, for the before/after comparison.
            {"n_estimators": 100, "max_depth": 20},
        ),
        "XGBoost": (
            XGBClassifier(tree_method="hist", n_jobs=-1,
                          random_state=RANDOM_STATE, verbosity=0),
            {
                "n_estimators": randint(100, 400),
                "max_depth": randint(4, 12),
                "learning_rate": uniform(0.05, 0.35),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "min_child_weight": randint(1, 8),
            },
            {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.3},
        ),
    }


def subsample(X: pd.DataFrame, y: pd.Series, n: int):
    """Stratified subsample that keeps every class present in every fold."""
    if len(X) <= n:
        return X, y

    rate = n / len(X)
    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for label in pd.unique(y):
        idx = np.flatnonzero((y == label).to_numpy())
        take = max(N_FOLDS, int(round(len(idx) * rate)))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))

    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep], y.iloc[keep]


def tune_experiment(X: pd.DataFrame, y: pd.Series, is_binary: bool) -> tuple[pd.DataFrame, dict]:
    """Run the randomised search for both models on one experiment."""
    label = "A - BINARY" if is_binary else "B - MULTI-CLASS"
    scoring = "f1" if is_binary else "f1_macro"
    Xs, ys = subsample(X, y, SUBSAMPLE)

    print(f"\n{'='*78}")
    print(f"EXPERIMENT {label}  |  {N_CANDIDATES} candidates x {N_FOLDS} folds "
          f"on {len(Xs):,} rows")
    print(f"{'='*78}\n")

    # XGBoost needs integer labels on the multi-class task.
    encoder = None if is_binary else LabelEncoder().fit(ys)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, best_params = [], {}
    for name, (estimator, space, defaults) in search_spaces().items():
        fit_y = ys if (is_binary or name != "XGBoost") else pd.Series(
            encoder.transform(ys), index=ys.index
        )

        # Score the untuned defaults first, using the same folds and the same
        # data, so the comparison isolates the effect of tuning.
        print(f"  {name:15} scoring defaults ...", end="", flush=True)
        baseline_scores = cross_val_score(
            estimator.__class__(**{**estimator.get_params(), **defaults}),
            Xs, fit_y, cv=cv, scoring=scoring, n_jobs=1,
        )
        print(f" {baseline_scores.mean():.5f}")

        print(f"  {name:15} searching ...", end="", flush=True)
        start = time.perf_counter()
        search = RandomizedSearchCV(
            estimator, space, n_iter=N_CANDIDATES, cv=cv, scoring=scoring,
            random_state=RANDOM_STATE, n_jobs=1, refit=False,
        )
        search.fit(Xs, fit_y)
        elapsed = time.perf_counter() - start

        gain = search.best_score_ - baseline_scores.mean()
        print(f" {search.best_score_:.5f}  ({gain:+.5f}, {elapsed/60:.1f} min)")
        print(f"      best: {search.best_params_}")

        rows.append({
            "model": name,
            "baseline_cv_f1": baseline_scores.mean(),
            "baseline_cv_std": baseline_scores.std(),
            "tuned_cv_f1": search.best_score_,
            "tuned_cv_std": search.cv_results_["std_test_score"][search.best_index_],
            "gain": gain,
            "search_minutes": round(elapsed / 60, 2),
        })
        best_params[name] = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        }

    return pd.DataFrame(rows), best_params


def plot_tuning(binary: pd.DataFrame, multi: pd.DataFrame) -> None:
    """Before and after tuning, per model, on a zoomed axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    panels = [
        (axes[0], binary, "Experiment A - BENIGN vs ATTACK"),
        (axes[1], multi, "Experiment B - 9 traffic classes"),
    ]

    for ax, frame, title in panels:
        y = np.arange(len(frame))
        ax.hlines(y, frame["baseline_cv_f1"], frame["tuned_cv_f1"],
                  color=GRID, linewidth=3, zorder=1)
        ax.scatter(frame["baseline_cv_f1"], y, s=85, color=GRID, zorder=2,
                   edgecolor="white", linewidth=1.5, label="Default parameters")
        ax.scatter(frame["tuned_cv_f1"], y, s=95, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=1.5, label="Tuned")

        for i, row in enumerate(frame.itertuples()):
            ax.text(max(row.baseline_cv_f1, row.tuned_cv_f1) + 0.0015, i,
                    f"{row.tuned_cv_f1:.5f}  ({row.gain:+.5f})",
                    va="center", fontsize=8.5, color=INK)

        ax.set_yticks(y, frame["model"].tolist(), fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Cross-validated F1 — note the zoomed axis")
        span = frame[["baseline_cv_f1", "tuned_cv_f1"]].to_numpy()
        ax.set_xlim(span.min() - 0.004, span.max() + 0.016)
        ax.set_ylim(-0.6, len(frame) - 0.4)
        ax.grid(axis="y", visible=False)

        # The right-hand padding exists to hold the value labels, but F1 cannot
        # exceed 1, so any tick past it would label a value the scale can never
        # reach. Drop those ticks and keep the padding.
        ax.set_xticks([t for t in ax.get_xticks() if t <= 1.0 + 1e-9])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("Tuning moves these models very little", y=1.19,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.08,
             "The defaults were already close to the ceiling. Gains this small "
             "are worth checking against the fold spread before being claimed.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, "13_hyperparameter_tuning.png")


def report(frame: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(frame.round(5).to_string(index=False))

    for row in frame.itertuples():
        spread = row.baseline_cv_std + row.tuned_cv_std
        verdict = ("real" if row.gain > spread
                   else "within the fold spread, so not meaningful")
        print(f"  {row.model}: gain {row.gain:+.5f} vs combined spread "
              f"{spread:.5f} -> {verdict}")


def main() -> None:
    apply_style()

    split = load_split()
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()
    X = split["X_train"][features]

    binary, binary_params = tune_experiment(X, split["y_train_binary"], is_binary=True)
    report(binary, "Experiment A - BINARY")
    binary.to_csv(TABLES_DIR / "tuning_binary.csv", index=False)

    multi, multi_params = tune_experiment(X, split["y_train_multi"], is_binary=False)
    report(multi, "Experiment B - MULTI-CLASS")
    multi.to_csv(TABLES_DIR / "tuning_multi.csv", index=False)

    BEST_PARAMS_PATH.write_text(
        json.dumps({"binary": binary_params, "multi": multi_params}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved best parameters -> {BEST_PARAMS_PATH}")

    print("\nBuilding figure:")
    plot_tuning(binary, multi)


if __name__ == "__main__":
    main()
