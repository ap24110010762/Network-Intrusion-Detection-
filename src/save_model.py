"""
Phase 17 - Train and persist the final models.

Fits XGBoost on the full 320,032-row training split for both experiments and
saves each one to models/ with everything needed to score a raw flow.

Default hyperparameters are used, not the tuned ones. Phase 13 found every
tuning gain to be inside the fold spread, and Phase 14 confirmed it on the
test set: the tuned multi-class model scored lower (0.957744 against 0.958281)
and took 56% longer to fit.

What is saved
-------------
A dictionary rather than a bare estimator, because a bare estimator does not
record which columns it expects or in what order - and a column order mismatch
produces silently wrong predictions rather than an error. The bundle carries:

    model         the fitted XGBClassifier
    features      the 25 feature names, in the order the model was fitted on
    classes       label names, for decoding multi-class predictions
    task          "binary" or "multi"
    metrics       the Phase 14 test-set scores, so a loaded model can state
                  its own measured performance
    created       UTC timestamp

A dict of plain objects also unpickles anywhere xgboost and scikit-learn are
installed. A custom Pipeline class would require this project's source to be
importable wherever the model is loaded, which is a poor property for a saved
artefact.

Usage (from the project root):
    python src/save_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from data_loader import PROJECT_ROOT, RESULTS_DIR
from datasets import load_split

MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = RESULTS_DIR / "tables"
SELECTED_FEATURES_PATH = TABLES_DIR / "selected_features.csv"
FINAL_METRICS_PATH = TABLES_DIR / "final_evaluation.csv"

RANDOM_STATE = 42
PARAMS = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.3}

TASKS = {
    "binary": {"filename": "binary_model.pkl", "target": "y_train_binary"},
    "multi": {"filename": "multiclass_model.pkl", "target": "y_train_multi"},
}


def final_metrics(task: str) -> dict:
    """The Phase 14 test-set scores for the settings actually shipped."""
    if not FINAL_METRICS_PATH.exists():
        return {}
    frame = pd.read_csv(FINAL_METRICS_PATH)
    experiment = "A (binary)" if task == "binary" else "B (multi)"
    row = frame[(frame["experiment"] == experiment) & (frame["settings"] == "default")]
    if row.empty:
        return {}
    keep = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return {k: float(row.iloc[0][k]) for k in keep if k in row.columns}


def train_and_save(split: dict, features: list[str], task: str) -> Path:
    """Fit on the full training split and write the model bundle."""
    y_train = split[TASKS[task]["target"]]
    X_train = split["X_train"][features]

    print(f"\n  {task:6} | fitting on {len(X_train):,} rows x {len(features)} features ...",
          end="", flush=True)

    classes = None
    fit_y = y_train
    if task == "multi":
        # Store the class names alongside the model so predictions decode
        # without the training data being present.
        classes = sorted(y_train.unique())
        lookup = {name: i for i, name in enumerate(classes)}
        fit_y = y_train.map(lookup)

    model = XGBClassifier(
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
        verbosity=0, **PARAMS,
    )
    model.fit(X_train, fit_y)
    print(" done")

    bundle = {
        "model": model,
        "features": features,
        "classes": classes if classes is not None else ["BENIGN", "ATTACK"],
        "task": task,
        "params": PARAMS,
        "train_rows": int(len(X_train)),
        "metrics": final_metrics(task),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / TASKS[task]["filename"]
    joblib.dump(bundle, path, compress=3)
    print(f"         saved -> {path.name}  ({path.stat().st_size/1024**2:.1f} MB)")
    return path


def verify(path: Path, split: dict) -> None:
    """
    Reload the saved bundle and confirm it predicts.

    A model that pickles but cannot be loaded and used is worse than no model,
    and the failure usually appears at demo time. This checks the round trip
    on a handful of held-out rows.
    """
    bundle = joblib.load(path)
    model, features, classes = bundle["model"], bundle["features"], bundle["classes"]

    sample = split["X_test"][features].head(5)
    codes = model.predict(sample)
    labels = [classes[int(c)] for c in codes] if bundle["task"] == "multi" else [
        classes[int(c)] for c in codes
    ]
    truth_col = "y_test_multi" if bundle["task"] == "multi" else "y_test_binary"
    truth = split[truth_col].head(5).tolist()
    if bundle["task"] == "binary":
        truth = ["BENIGN" if t == 0 else "ATTACK" for t in truth]

    print(f"         reload check: predicted {labels}")
    print(f"                       actual    {truth}")
    if bundle["metrics"]:
        print(f"         carries its own test scores: F1 {bundle['metrics']['f1']:.6f}")


def main() -> None:
    split = load_split()
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()

    print("=" * 78)
    print("PHASE 17 - SAVING FINAL MODELS")
    print("=" * 78)
    print(f"\nXGBoost, default settings {PARAMS}")

    for task in TASKS:
        path = train_and_save(split, features, task)
        verify(path, split)

    manifest = {
        task: {
            "file": f"models/{TASKS[task]['filename']}",
            "metrics": final_metrics(task),
        }
        for task in TASKS
    }
    (TABLES_DIR / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nSaved manifest -> {TABLES_DIR / 'model_manifest.json'}")


if __name__ == "__main__":
    main()
