"""
Phase 8 - Sampling and the train/test split.

Two jobs:

1. Build a ~400,000-row modelling sample from the 2.5M-row cleaned dataset.
   Training six algorithms on 2.5M rows is not practical on 4 CPU cores -
   SVM and KNN in particular scale badly - so the model comparison runs on a
   sample and only the final chosen model is refitted on the full data.

2. Produce a stratified, leakage-free 80/20 train/test split.

Nothing here fits a transformation on the data. Scaling and any other fitted
preprocessing belong inside a Pipeline built on the training split only, so
that no information from the test set can reach the model.

Usage (from the project root):
    python src/datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import LABEL_COL, RESULTS_DIR
from preprocessing import BINARY_LABEL_COL, GROUPED_LABEL_COL, PROCESSED_DIR

CLEANED_PATH = PROCESSED_DIR / "cleaned.parquet"
SAMPLE_PATH = PROCESSED_DIR / "sample.parquet"
SPLIT_DIR = PROCESSED_DIR / "split"
TABLES_DIR = RESULTS_DIR / "tables"

SAMPLE_SIZE = 400_000
TEST_SIZE = 0.20
RANDOM_STATE = 42

# A class whose proportional share falls below this is kept in full instead.
# At a 16% sampling rate Heartbleed's share is 1.8 rows, which would leave the
# test split with none at all; keeping all 11 costs 9 extra rows out of
# 400,000 and leaves the distribution effectively unchanged.
MIN_ROWS_PER_CLASS = 50

# Columns that are labels or bookkeeping, not model inputs.
NON_FEATURE_COLS = [LABEL_COL, GROUPED_LABEL_COL, BINARY_LABEL_COL, "source_file"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The numeric predictor columns, excluding every target and helper column."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in NON_FEATURE_COLS
    ]


def build_sample(
    df: pd.DataFrame,
    size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draw a stratified sample of roughly `size` rows.

    Classes are sampled proportionally, except that any class whose
    proportional share would fall below MIN_ROWS_PER_CLASS is kept in full.
    """
    rate = size / len(df)
    parts = []
    report = []

    for label, group in df.groupby(GROUPED_LABEL_COL, observed=True):
        proportional = int(round(len(group) * rate))
        if proportional < MIN_ROWS_PER_CLASS:
            take = len(group)          # keep every row of an ultra-rare class
            rule = "kept in full"
        else:
            take = proportional
            rule = "proportional"

        part = group.sample(take, random_state=random_state) if take < len(group) else group
        parts.append(part)
        report.append(
            {
                "class": str(label),
                "rows_full": len(group),
                "rows_sampled": len(part),
                "percent_full": round(100 * len(group) / len(df), 4),
                "rule": rule,
            }
        )

    sample = pd.concat(parts).sample(frac=1.0, random_state=random_state)

    report_df = pd.DataFrame(report).sort_values("rows_full", ascending=False)
    report_df["percent_sampled"] = (
        100 * report_df["rows_sampled"] / len(sample)
    ).round(4)

    print(f"\nSampling {len(df):,} rows down to {len(sample):,} "
          f"(rate {rate:.2%})")
    print(report_df.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(TABLES_DIR / "sampling_report.csv", index=False)

    return sample


def make_split(
    df: pd.DataFrame,
    target: str = GROUPED_LABEL_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Stratified 80/20 train/test split.

    Stratifying on the 9-class label also keeps the binary target balanced,
    so one split serves both Experiment A and Experiment B. That matters:
    using the same split for both makes their results directly comparable.
    """
    features = feature_columns(df)
    X = df[features]
    y_multi = df[GROUPED_LABEL_COL].astype(str)
    y_binary = df[BINARY_LABEL_COL]

    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=df[target].astype(str),
    )

    split = {
        "X_train": X.iloc[idx_train],
        "X_test": X.iloc[idx_test],
        "y_train_multi": y_multi.iloc[idx_train],
        "y_test_multi": y_multi.iloc[idx_test],
        "y_train_binary": y_binary.iloc[idx_train],
        "y_test_binary": y_binary.iloc[idx_test],
        "features": features,
    }

    print(f"\nSplit: {len(idx_train):,} train / {len(idx_test):,} test "
          f"({1-test_size:.0%}/{test_size:.0%}), stratified on '{target}'")
    print(f"Features: {len(features)}")

    comparison = pd.DataFrame({
        "train_%": split["y_train_multi"].value_counts(normalize=True).mul(100).round(3),
        "test_%": split["y_test_multi"].value_counts(normalize=True).mul(100).round(3),
        "train_n": split["y_train_multi"].value_counts(),
        "test_n": split["y_test_multi"].value_counts(),
    }).sort_values("train_n", ascending=False)
    print("\nClass balance preserved across the split:")
    print(comparison.to_string())

    comparison.to_csv(TABLES_DIR / "split_class_balance.csv")
    return split


def verify_no_leakage(split: dict) -> dict:
    """
    Check whether identical feature rows appear in both train and test, and
    separate the two very different reasons that can happen.

    True leakage means the same features AND the same label on both sides:
    the model can memorise the answer, and every downstream metric is
    inflated. preprocessing.py removes exact duplicates, so this should be
    zero.

    A label conflict means the same features with DIFFERENT labels. That is
    not leakage - the model gains nothing, because the two copies disagree
    about the answer. It is label noise in the source dataset: CIC-IDS2017
    contains flows whose measurements are identical but which were labelled
    benign in one capture and as an attack in another. These rows put a hard
    ceiling on achievable accuracy, since no model can be right about both.
    """
    train_hashes = pd.util.hash_pandas_object(split["X_train"], index=False).to_numpy()
    test_hashes = pd.util.hash_pandas_object(split["X_test"], index=False).to_numpy()
    overlap = np.intersect1d(train_hashes, test_hashes)

    y_train = split["y_train_multi"].to_numpy()
    y_test = split["y_test_multi"].to_numpy()

    leaked, conflicting = 0, 0
    for h in overlap:
        train_labels = set(y_train[train_hashes == h])
        test_labels = set(y_test[test_hashes == h])
        if train_labels & test_labels:
            leaked += 1
        else:
            conflicting += 1

    print("\n  Leakage check:")
    if leaked:
        print(f"    LEAKAGE: {leaked:,} feature rows share a label across "
              "train and test. Results would be inflated - investigate.")
    else:
        print("    no rows share both features and label across train and test")

    if conflicting:
        print(f"    {conflicting:,} row(s) have identical features but "
              "conflicting labels (label noise in the source data, not leakage)")

    return {"leaked": leaked, "conflicting": conflicting}


def save_split(split: dict) -> None:
    """Persist the split so every later phase trains and scores on the same rows."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    split["X_train"].to_parquet(SPLIT_DIR / "X_train.parquet", index=False)
    split["X_test"].to_parquet(SPLIT_DIR / "X_test.parquet", index=False)
    pd.DataFrame({
        "multi": split["y_train_multi"].to_numpy(),
        "binary": split["y_train_binary"].to_numpy(),
    }).to_parquet(SPLIT_DIR / "y_train.parquet", index=False)
    pd.DataFrame({
        "multi": split["y_test_multi"].to_numpy(),
        "binary": split["y_test_binary"].to_numpy(),
    }).to_parquet(SPLIT_DIR / "y_test.parquet", index=False)

    print(f"\nSaved split -> {SPLIT_DIR}")


def load_split() -> dict:
    """Reload the split saved by save_split()."""
    if not (SPLIT_DIR / "X_train.parquet").exists():
        raise FileNotFoundError(
            f"{SPLIT_DIR} is empty. Run 'python src/datasets.py' first."
        )

    X_train = pd.read_parquet(SPLIT_DIR / "X_train.parquet")
    X_test = pd.read_parquet(SPLIT_DIR / "X_test.parquet")
    y_train = pd.read_parquet(SPLIT_DIR / "y_train.parquet")
    y_test = pd.read_parquet(SPLIT_DIR / "y_test.parquet")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_multi": y_train["multi"],
        "y_test_multi": y_test["multi"],
        "y_train_binary": y_train["binary"],
        "y_test_binary": y_test["binary"],
        "features": X_train.columns.tolist(),
    }


def main() -> None:
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_PATH} not found. Run 'python src/preprocessing.py' first."
        )

    print(f"Loading {CLEANED_PATH.name}...")
    df = pd.read_parquet(CLEANED_PATH)
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    sample = build_sample(df)
    sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"\nSaved sample -> {SAMPLE_PATH} "
          f"({SAMPLE_PATH.stat().st_size / 1024**2:,.0f} MB)")

    split = make_split(sample)
    verify_no_leakage(split)
    save_split(split)


if __name__ == "__main__":
    main()
