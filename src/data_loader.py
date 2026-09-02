"""
Phase 3 - Data Integration for CIC-IDS2017.

Loads the 8 MachineLearningCSV files, cleans up the column names,
merges them into one master DataFrame, and prints a profile of it.

Usage (from the project root, with venv active):
    python src/data_loader.py
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths - resolved relative to this file, so it works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

LABEL_COL = "Label"


def find_csv_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return every CSV in the data folder, sorted by name."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            "Download MachineLearningCSV.zip from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and extract the "
            "CSVs into the data/ folder. See data/README.md."
        )
    return files


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing spaces from column names.

    CIC-IDS2017 ships with names like ' Flow Duration' and ' Label'. Leaving
    the spaces in causes silent KeyErrors later, so we fix them on load.
    """
    df.columns = [str(c).strip() for c in df.columns]
    return df


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink float64 -> float32 and int64 -> the smallest safe integer type.

    This roughly halves memory use. This project runs on a machine with about
    8 GB of RAM and the merged dataset is ~2.8 million rows, so without this
    step the merge alone would exhaust memory. float32 keeps roughly 7
    significant digits, far more precision than these flow statistics carry.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_single_csv(path: Path, shrink: bool = True) -> pd.DataFrame:
    """Load one CIC-IDS2017 CSV with cleaned column names."""
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df = clean_column_names(df)
    if shrink:
        df = downcast_numeric(df)
    # 'category' stores each distinct label string once, not once per row.
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().astype("category")
    df["source_file"] = pd.Categorical([path.name] * len(df))
    return df


def load_all_data(data_dir: Path = DATA_DIR, verbose: bool = True) -> pd.DataFrame:
    """Load and concatenate every CIC-IDS2017 CSV into one master DataFrame."""
    files = find_csv_files(data_dir)
    frames = []

    for path in files:
        df = load_single_csv(path)
        if verbose:
            mb = df.memory_usage(deep=True).sum() / 1024**2
            print(
                f"  loaded {path.name:<58} {df.shape[0]:>9,} rows x "
                f"{df.shape[1]:>3} cols  ({mb:,.0f} MB)"
            )
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    # Release the per-file frames so the merged copy is not held twice in RAM.
    frames.clear()
    gc.collect()

    # concat drops the 'category' dtype when the files carry different label
    # sets (Monday is BENIGN-only, Friday has DDoS, ...), so restore it here.
    for col in (LABEL_COL, "source_file"):
        if col in master.columns:
            master[col] = master[col].astype("category")

    if verbose:
        mb = master.memory_usage(deep=True).sum() / 1024**2
        print(
            f"\n  MERGED: {master.shape[0]:,} rows x {master.shape[1]} columns"
            f"  ({mb:,.0f} MB in RAM)"
        )
    return master


def profile(df: pd.DataFrame) -> None:
    """
    Print the Phase 3 data profile the project document asks for:
    shape, dtypes, missing values, infinities, duplicates, class distribution.
    """
    print("\n" + "=" * 70)
    print("DATA PROFILE")
    print("=" * 70)

    print(f"\nRows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- Data types ---")
    print(df.dtypes.value_counts().to_string())

    print("\n--- Missing values (columns with at least one) ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string() if len(missing) else "  none")

    print("\n--- Infinite values (columns with at least one) ---")
    numeric = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric).sum()
    inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)
    print(inf_counts.to_string() if len(inf_counts) else "  none")

    print("\n--- Duplicate rows ---")
    n_dupes = df.duplicated().sum()
    print(f"  {n_dupes:,}  ({n_dupes / len(df):.2%} of the dataset)")

    if LABEL_COL in df.columns:
        print(f"\n--- Class distribution ('{LABEL_COL}') ---")
        counts = df[LABEL_COL].value_counts()
        pct = df[LABEL_COL].value_counts(normalize=True) * 100
        dist = pd.DataFrame({"count": counts, "percent": pct.round(4)})
        print(dist.to_string())

        benign_mask = df[LABEL_COL].astype(str).str.strip().str.upper() == "BENIGN"
        print(f"\n  BENIGN: {benign_mask.sum():,}  ({benign_mask.mean():.2%})")
        print(f"  ATTACK: {(~benign_mask).sum():,}  ({(~benign_mask).mean():.2%})")
    else:
        print(f"\n  WARNING: no '{LABEL_COL}' column found. Columns are:")
        print(f"  {list(df.columns)}")

    print("\n" + "=" * 70)


def main() -> None:
    print(f"Looking for CSVs in: {DATA_DIR}\n")
    df = load_all_data()
    profile(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tables" / "class_distribution.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if LABEL_COL in df.columns:
        df[LABEL_COL].value_counts().to_csv(out, header=["count"])
        print(f"Saved class distribution -> {out}")


if __name__ == "__main__":
    main()
