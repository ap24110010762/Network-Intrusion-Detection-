"""
Phase 5 - Exploratory Data Analysis for CIC-IDS2017.

Reads the cleaned Parquet file produced by preprocessing.py and writes
report-ready figures to results/figures/ plus supporting tables to
results/tables/.

Figures produced:
    01_class_distribution_multiclass.png
    02_class_distribution_binary.png
    03_correlation_heatmap.png
    04_feature_distributions.png
    05_feature_separation.png

Usage (from the project root):
    python src/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files without needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from data_loader import RESULTS_DIR
from preprocessing import BINARY_LABEL_COL, GROUPED_LABEL_COL, PROCESSED_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CLEANED_PATH = PROCESSED_DIR / "cleaned.parquet"

# Palette, validated with the data-viz palette checker (light surface):
# adjacent CVD dE 24.7, normal-vision dE 33.6, both slots >= 3:1 contrast.
BLUE = "#2a78d6"        # categorical slot 1 / sequential hue
ORANGE = "#eb6834"      # categorical slot 2
RED = "#e34948"         # diverging warm pole
GRAY_MID = "#f0efec"    # diverging neutral midpoint
INK = "#0b0b0b"         # text primary
INK_SOFT = "#52514e"    # text secondary
GRID = "#e6e5e1"

# Diverging map for correlations: two opposite hues, neutral gray midpoint.
CORR_CMAP = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, GRAY_MID, RED])

# Plotting on 2.5 million points is slow and adds nothing visually, so the
# distribution figures work from a stratified sample. Counts and correlations
# are still computed on the full dataset.
PLOT_SAMPLE_SIZE = 200_000
RANDOM_STATE = 42


def apply_style() -> None:
    """Set a consistent, recessive chart style for every figure."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def thousands(x, _pos) -> str:
    """Axis formatter: 1500000 -> '1.5M', 2500 -> '2.5K'."""
    if x >= 1_000_000:
        return f"{x/1_000_000:g}M"
    if x >= 1_000:
        return f"{x/1_000:g}K"
    return f"{x:g}"


def save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


# --------------------------------------------------------------------------
# Figure 1 - multi-class distribution
# --------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Class counts on a log axis, drawn as a lollipop chart.

    A log axis is unavoidable here: BENIGN has 2,072,444 rows and Heartbleed
    has 11, a ratio of about 188,000:1, so on a linear axis every attack class
    would be an invisible sliver. Lollipops encode the value by the position of
    the dot rather than by bar length, which stays honest under a log axis in a
    way that bars starting from an arbitrary left edge would not.
    """
    counts = df[GROUPED_LABEL_COL].value_counts().sort_values()
    total = len(df)
    y = np.arange(len(counts))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hlines(y, 1, counts.values, color=GRID, linewidth=2, zorder=1)
    ax.scatter(counts.values, y, s=90, color=BLUE, zorder=2,
               edgecolor="white", linewidth=2)

    for i, (name, value) in enumerate(counts.items()):
        pct = value / total * 100
        # Rare classes need extra decimals: at 2 places Heartbleed would read
        # '0.00%', which looks like zero rather than 11 real flows.
        label = f"{value:,}  ({pct:.4f}%)" if pct < 0.01 else f"{value:,}  ({pct:.2f}%)"
        ax.text(value * 1.35, i, label, va="center", ha="left",
                fontsize=9, color=INK)

    ax.set_yticks(y, counts.index.tolist())
    ax.set_xscale("log")
    ax.set_xlim(1, counts.max() * 12)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel("Number of network flows (log scale)")
    ax.set_title("Class distribution after cleaning: a 188,000 : 1 imbalance")
    ax.grid(axis="y", visible=False)

    # Positioned in axes fractions so it cannot collide with the value labels,
    # which sit in data coordinates on a log axis.
    ax.annotate(
        "Only 11 flows — too few\nfor any model to learn reliably",
        xy=(counts.iloc[0], 0), xycoords="data",
        xytext=(0.42, 0.12), textcoords="axes fraction",
        fontsize=8.5, color=INK_SOFT, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=INK_SOFT, linewidth=1,
                        connectionstyle="arc3,rad=0.15"),
    )

    fig.text(0.5, -0.02, f"CIC-IDS2017, {total:,} flows after cleaning",
             ha="center", fontsize=8, color=INK_SOFT)
    save(fig, "01_class_distribution_multiclass.png")


# --------------------------------------------------------------------------
# Figure 2 - binary distribution
# --------------------------------------------------------------------------

def plot_binary_distribution(df: pd.DataFrame) -> None:
    """BENIGN vs ATTACK on a linear axis - here the two bars are comparable."""
    counts = df[BINARY_LABEL_COL].value_counts().sort_index()
    names = ["BENIGN (0)", "ATTACK (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars = ax.barh(names, values, color=[BLUE, ORANGE], height=0.55)
    for bar, value in zip(bars, values):
        ax.text(value + total * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:,}   ({value/total:.2%})",
                va="center", fontsize=10, color=INK)

    ax.set_xlim(0, total * 1.28)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel("Number of network flows")
    ax.set_title("Experiment A target: benign traffic outnumbers attacks about 5 to 1")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    fig.text(0.5, -0.08,
             "A model predicting 'BENIGN' for everything would score 82.96% accuracy "
             "and detect nothing —\nwhich is why accuracy alone is not reported.",
             ha="center", fontsize=8, color=INK_SOFT)
    save(fig, "02_class_distribution_binary.png")


# --------------------------------------------------------------------------
# Figure 3 - correlation heatmap
# --------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Correlation heatmap for the features most correlated with the target.

    All 70 features would be an unreadable wall, so we show the top_n features
    ranked by absolute correlation with the binary label.
    """
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[BINARY_LABEL_COL])
    target = df[BINARY_LABEL_COL]

    target_corr = numeric.corrwith(target).abs().sort_values(ascending=False)
    target_corr = target_corr.dropna()
    top_features = target_corr.head(top_n).index.tolist()

    corr = df[top_features].corr()

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap=CORR_CMAP, vmin=-1, vmax=1)

    ax.set_xticks(range(len(top_features)), top_features,
                  rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(top_features)), top_features, fontsize=7.5)
    ax.set_title(f"Correlation between the {top_n} features most predictive of attack traffic")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Pearson correlation", fontsize=9, color=INK_SOFT)
    cbar.outline.set_visible(False)

    # Negative y keeps the caption clear of the rotated tick labels; the
    # figure is saved with bbox_inches='tight', so it is still included.
    fig.text(0.5, -0.04,
             "Deep blue or deep red pairs carry nearly the same information — "
             "candidates for removal in Phase 7 feature selection.",
             ha="center", fontsize=8, color=INK_SOFT)
    save(fig, "03_correlation_heatmap.png")

    target_corr.to_csv(TABLES_DIR / "feature_target_correlation.csv",
                       header=["abs_correlation_with_attack"])
    return target_corr


# --------------------------------------------------------------------------
# Figure 4 - feature distributions, benign vs attack
# --------------------------------------------------------------------------

def plot_feature_distributions(sample: pd.DataFrame, features: list[str]) -> None:
    """Small multiples: how each feature is distributed for benign vs attack."""
    benign = sample[sample[BINARY_LABEL_COL] == 0]
    attack = sample[sample[BINARY_LABEL_COL] == 1]

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        # log1p keeps the heavy right tails readable; these features span
        # many orders of magnitude and are non-negative.
        b = np.log1p(np.clip(benign[feature].to_numpy(), 0, None))
        a = np.log1p(np.clip(attack[feature].to_numpy(), 0, None))

        bins = np.linspace(0, max(b.max(), a.max(), 1), 60)
        ax.hist(b, bins=bins, color=BLUE, alpha=0.75, label="BENIGN", density=True)
        ax.hist(a, bins=bins, color=ORANGE, alpha=0.7, label="ATTACK", density=True)
        ax.set_title(feature, fontsize=10)
        ax.set_ylabel("density")
        ax.set_xlabel("log(1 + value)")
        ax.grid(axis="x", visible=False)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Where benign and attack traffic separate", y=1.045, fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_feature_distributions.png")


# --------------------------------------------------------------------------
# Figure 5 - feature separation ranking
# --------------------------------------------------------------------------

def plot_feature_separation(target_corr: pd.Series, top_n: int = 20) -> None:
    """Rank features by how strongly they correlate with attack traffic."""
    top = target_corr.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top.index, top.values, color=BLUE, height=0.65)
    for i, value in enumerate(top.values):
        ax.text(value + 0.006, i, f"{value:.3f}", va="center",
                fontsize=8.5, color=INK)

    ax.set_xlim(0, top.max() * 1.15)
    ax.set_xlabel("Absolute Pearson correlation with the attack label")
    ax.set_title(f"Top {top_n} single features for separating benign from attack traffic")
    ax.grid(axis="y", visible=False)

    fig.text(0.5, -0.02,
             "Pearson correlation only captures linear relationships — "
             "tree models in Phase 9 may rank these differently.",
             ha="center", fontsize=8, color=INK_SOFT)
    save(fig, "05_feature_separation.png")


def pick_features(df: pd.DataFrame, target_corr: pd.Series, n: int = 6) -> list[str]:
    """Choose interpretable features for the distribution figure."""
    preferred = [
        "Flow Duration", "Total Fwd Packets", "Flow Bytes/s",
        "Fwd Packet Length Mean", "Flow IAT Mean", "Average Packet Size",
    ]
    chosen = [f for f in preferred if f in df.columns]
    # Top up from the correlation ranking if any preferred name is missing.
    for name in target_corr.index:
        if len(chosen) >= n:
            break
        if name not in chosen:
            chosen.append(name)
    return chosen[:n]


def main() -> None:
    apply_style()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_PATH} not found. Run 'python src/preprocessing.py' first."
        )

    print(f"Loading {CLEANED_PATH.name}...")
    df = pd.read_parquet(CLEANED_PATH)
    print(f"  {len(df):,} rows x {df.shape[1]} columns\n")

    print("Building figures:")
    plot_class_distribution(df)
    plot_binary_distribution(df)
    target_corr = plot_correlation_heatmap(df)

    # Stratified sample, built by concatenating per-class samples. Note that
    # groupby().apply() is not used here: it drops the grouping column, which
    # is the very column the distribution plots need.
    per_class = PLOT_SAMPLE_SIZE // 2
    sample = pd.concat(
        [
            group.sample(min(len(group), per_class), random_state=RANDOM_STATE)
            for _, group in df.groupby(BINARY_LABEL_COL, observed=True)
        ]
    )
    features = pick_features(df, target_corr)
    plot_feature_distributions(sample, features)
    plot_feature_separation(target_corr)

    print(f"\nTop 10 features by correlation with the attack label:")
    print(target_corr.head(10).round(4).to_string())
    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
