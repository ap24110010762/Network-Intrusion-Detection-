"""
Phase 4 - Data Cleaning for CIC-IDS2017.

Every decision made here is logged to the console and to
results/tables/cleaning_log.csv, because the project document requires each
cleaning decision to be documented.

Cleaning steps, in order:
    1. Repair the corrupted label names shipped in the raw CSVs.
    2. Drop duplicate feature columns (the dataset ships two 'Fwd Header
       Length' columns).
    3. Replace +/-inf with NaN (they come from divide-by-zero on flows of
       zero duration) and drop the affected rows.
    4. Drop exact duplicate rows, ignoring 'source_file'.
    5. Drop constant (zero-variance) columns, which carry no information.
    6. Add the two target columns: 'Label_grouped' and 'Label_binary'.

Usage (from the project root):
    python src/preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import DATA_DIR, LABEL_COL, PROJECT_ROOT, RESULTS_DIR, load_all_data

PROCESSED_DIR = DATA_DIR / "processed"

# The raw CSVs contain the byte sequence EF BF BD where an en-dash should be.
# That is UTF-8 for U+FFFD, the replacement character, so the corruption is
# baked into the published files themselves.
#
# We read the CSVs as latin-1, which maps each byte to its own character, so
# the label arrives as three characters (U+00EF U+00BF U+00BD) rather than one
# U+FFFD. The pattern below matches a run of either form, so the repair works
# whichever encoding the files are read with.
BAD_CHAR_PATTERN = r"[ï¿½�]+"

# Phase 6 grouping: collapse the attack variants into families. The four DoS
# variants become one 'DoS' class and the three web attacks become one
# 'Web Attack' class. Set GROUP_LABELS = False in main() to keep all 15.
LABEL_GROUPS = {
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
}

GROUPED_LABEL_COL = "Label_grouped"
BINARY_LABEL_COL = "Label_binary"
BENIGN = "BENIGN"


class CleaningLog:
    """Collects one row per cleaning decision so it can be saved and cited."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record(self, step: str, detail: str, rows_before: int, rows_after: int) -> None:
        removed = rows_before - rows_after
        self.entries.append(
            {
                "step": step,
                "detail": detail,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": removed,
                "percent_removed": round(100 * removed / rows_before, 4) if rows_before else 0.0,
            }
        )
        print(f"  [{step}] {detail}")
        if removed:
            print(f"      rows {rows_before:,} -> {rows_after:,}  (removed {removed:,}, {100*removed/rows_before:.2f}%)")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


def fix_label_text(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Replace the corrupted character in the Web Attack labels with a dash."""
    labels = df[LABEL_COL].astype(str)
    n_bad = int(labels.str.contains(BAD_CHAR_PATTERN, regex=True).sum())

    repaired = (
        labels.str.replace(BAD_CHAR_PATTERN, "-", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[LABEL_COL] = repaired.astype("category")

    fixed_names = sorted(set(repaired[labels != repaired]))
    log.record(
        "fix_labels",
        f"repaired {n_bad:,} labels containing corrupted bytes -> {fixed_names or 'none'}",
        len(df),
        len(df),
    )

    # The grouping in Phase 6 relies on these exact strings, so fail loudly
    # here rather than silently producing an unexpected number of classes.
    remaining = [n for n in set(repaired) if any(c in n for c in "ï¿½�")]
    if remaining:
        raise ValueError(f"labels still contain corrupted characters: {remaining}")

    return df


def drop_duplicate_columns(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Drop repeated column names.

    CIC-IDS2017 ships two columns both named 'Fwd Header Length'. Pandas
    keeps both, which breaks column selection later.
    """
    before = df.shape[1]
    dupe_names = df.columns[df.columns.duplicated()].unique().tolist()
    df = df.loc[:, ~df.columns.duplicated()]

    log.record(
        "drop_duplicate_columns",
        f"dropped {before - df.shape[1]} repeated column(s): {dupe_names or 'none'}",
        len(df),
        len(df),
    )
    return df


def handle_infinities(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Replace +/-inf with NaN, then drop rows that hold any NaN.

    The infinities come from rate features (Flow Bytes/s, Flow Packets/s)
    computed on flows with zero duration. They affect well under 1% of rows,
    so dropping is cleaner than imputing a value that would be fabricated.
    """
    # Only float columns can hold inf, and we work one column at a time:
    # materialising all 76 numeric columns as one float64 array would need
    # roughly 1.7 GB on the full dataset.
    float_cols = df.select_dtypes(include=["float32", "float64"]).columns

    n_inf = 0
    rows_before = len(df)
    affected = pd.Series(False, index=df.index)

    for col in float_cols:
        values = df[col].to_numpy()
        bad = np.isinf(values) | np.isnan(values)
        n_bad = int(bad.sum())
        if n_bad:
            n_inf += int(np.isinf(values).sum())
            df.loc[bad, col] = np.nan
            affected |= bad

    # Report which classes lose rows, so we notice if a rare class is hit hard.
    if affected.any():
        hit = df.loc[affected, LABEL_COL].value_counts()
        hit = hit[hit > 0]
        print(f"      rows affected by inf/NaN, by class:\n{hit.to_string()}")

    df = df.loc[~affected].copy()
    log.record(
        "handle_infinities",
        f"converted {n_inf:,} infinite values to NaN, then dropped rows containing NaN",
        rows_before,
        len(df),
    )
    return df


def drop_duplicate_rows(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Drop exact duplicate rows.

    'source_file' is excluded from the comparison so that identical flows
    appearing in two different day-files are still caught. Duplicates must go
    before the train/test split: the same row landing in both sets lets the
    model memorise test answers and produces falsely high scores.
    """
    rows_before = len(df)
    compare_cols = [c for c in df.columns if c != "source_file"]
    df = df.drop_duplicates(subset=compare_cols).copy()

    log.record(
        "drop_duplicate_rows",
        "dropped exact duplicate rows (ignoring source_file) to prevent train/test leakage",
        rows_before,
        len(df),
    )
    return df


def drop_constant_columns(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Drop numeric columns with only one distinct value - they carry no signal."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nunique = df[numeric_cols].nunique()
    constant = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant)

    log.record(
        "drop_constant_columns",
        f"dropped {len(constant)} zero-variance column(s): {constant or 'none'}",
        len(df),
        len(df),
    )
    return df


def add_target_columns(df: pd.DataFrame, log: CleaningLog, group: bool = True) -> pd.DataFrame:
    """Add the grouped multi-class target and the binary BENIGN/ATTACK target."""
    labels = df[LABEL_COL].astype(str)

    grouped = labels.map(lambda x: LABEL_GROUPS.get(x, x)) if group else labels
    df[GROUPED_LABEL_COL] = grouped.astype("category")

    # 0 = BENIGN, 1 = ATTACK, matching Experiment A in the project document.
    df[BINARY_LABEL_COL] = (labels != BENIGN).astype("int8")

    # Labels that are neither BENIGN nor mapped are expected to stand alone.
    # Anything outside this set means a label string did not match as intended
    # (a stray space, a bad character), which would silently inflate the class
    # count - exactly the bug this guard exists to catch.
    EXPECTED_STANDALONE = {BENIGN, "DDoS", "PortScan", "Bot", "Infiltration", "Heartbleed"}
    if group:
        ungrouped = set(df[GROUPED_LABEL_COL].astype(str)) - set(LABEL_GROUPS.values())
        unexpected = ungrouped - EXPECTED_STANDALONE
        if unexpected:
            raise ValueError(
                f"unexpected label(s) after grouping: {sorted(unexpected)}. "
                "Check LABEL_GROUPS keys against the repaired label strings."
            )

    n_classes = df[GROUPED_LABEL_COL].nunique()
    log.record(
        "add_targets",
        f"added '{GROUPED_LABEL_COL}' ({n_classes} classes) and '{BINARY_LABEL_COL}' (0=BENIGN, 1=ATTACK)",
        len(df),
        len(df),
    )
    return df


def clean(df: pd.DataFrame, group_labels: bool = True) -> tuple[pd.DataFrame, CleaningLog]:
    """Run the full Phase 4 cleaning pipeline."""
    log = CleaningLog()
    print("\nCLEANING")
    print("-" * 70)

    df = fix_label_text(df, log)
    df = drop_duplicate_columns(df, log)
    df = handle_infinities(df, log)
    df = drop_duplicate_rows(df, log)
    df = drop_constant_columns(df, log)
    df = add_target_columns(df, log, group=group_labels)

    return df, log


def summarise(df: pd.DataFrame) -> None:
    """Print the class distribution of the cleaned dataset."""
    print("\n" + "=" * 70)
    print("CLEANED DATASET")
    print("=" * 70)
    print(f"\nRows:    {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    print(f"\n--- Multi-class distribution ('{GROUPED_LABEL_COL}') ---")
    counts = df[GROUPED_LABEL_COL].value_counts()
    pct = (counts / len(df) * 100).round(4)
    print(pd.DataFrame({"count": counts, "percent": pct}).to_string())

    print(f"\n--- Binary distribution ('{BINARY_LABEL_COL}') ---")
    b = df[BINARY_LABEL_COL].value_counts().sort_index()
    print(f"  0 BENIGN: {b.get(0, 0):,}  ({b.get(0, 0)/len(df):.2%})")
    print(f"  1 ATTACK: {b.get(1, 0):,}  ({b.get(1, 0)/len(df):.2%})")

    print(f"\nImbalance ratio (largest : smallest class) = "
          f"{counts.max() / counts.min():,.0f} : 1")
    print("=" * 70)


def main() -> None:
    GROUP_LABELS = True  # set False to keep all 15 original attack labels

    print("Loading raw data...")
    df = load_all_data(DATA_DIR)

    df, log = clean(df, group_labels=GROUP_LABELS)
    summarise(df)

    # Save the cleaning log for the report.
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    log_path = tables_dir / "cleaning_log.csv"
    log.to_frame().to_csv(log_path, index=False)
    print(f"\nSaved cleaning log      -> {log_path}")

    dist_path = tables_dir / "class_distribution_clean.csv"
    df[GROUPED_LABEL_COL].value_counts().to_csv(dist_path, header=["count"])
    print(f"Saved class distribution -> {dist_path}")

    # Save the cleaned data as Parquet: far faster to reload than 8 CSVs, and
    # it preserves dtypes, so every later phase starts from the same table.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "cleaned.parquet"
    df.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1024**2
    print(f"Saved cleaned dataset    -> {out}  ({size_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
