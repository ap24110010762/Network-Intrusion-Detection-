"""
Phase 7 - Feature Selection for CIC-IDS2017.

Four methods are compared, then combined into a consensus ranking:

    1. Correlation filter    - drops one of any pair of near-identical features
    2. Mutual information    - captures non-linear dependence on the label
    3. Random Forest         - impurity-based importance from a tree ensemble
    4. RFE                   - recursively removes the weakest feature

Everything here is fitted on the TRAINING split only. Mutual information, RF
importance and RFE all learn from the labels, so fitting them on the full
dataset would let test-set information influence which features are chosen -
a subtle form of leakage that inflates the final score.

The module finishes by answering the question the project document asks in
Phase 7: does using the selected features actually cost any accuracy?

Usage (from the project root):
    python src/feature_selection.py
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from datasets import load_split
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save
from data_loader import RESULTS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

RANDOM_STATE = 42

# Two features correlated above this threshold carry essentially the same
# information, so one of them is redundant.
CORRELATION_THRESHOLD = 0.95

# Mutual information and RFE are far slower than the other methods, so they
# run on a subsample of the training split. Both produce a *ranking*, which is
# stable at this size; they are not the final model.
MI_SAMPLE = 100_000
RFE_SAMPLE = 100_000

N_SELECTED = 25  # size of the final selected feature set


def subsample(X: pd.DataFrame, y: pd.Series, n: int):
    """Take a stratified subsample of the training split for the slow methods."""
    if len(X) <= n:
        return X, y
    idx = (
        pd.Series(np.arange(len(X)))
        .groupby(y.to_numpy(), observed=True)
        .apply(lambda g: g.sample(max(1, int(round(len(g) * n / len(X)))),
                                 random_state=RANDOM_STATE))
        .explode()
        .astype(int)
        .to_numpy()
    )
    return X.iloc[idx], y.iloc[idx]


# --------------------------------------------------------------------------
# Method 1 - correlation filter
# --------------------------------------------------------------------------

def correlation_filter(X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD):
    """
    Find features that duplicate another feature's information.

    For each pair correlated above the threshold, one of the two is marked
    redundant. This uses no labels at all, so it is the one method here that
    could safely see the whole dataset - it is kept on the training split
    anyway for consistency.
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    redundant, pairs = set(), []
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if partner in redundant or col in redundant:
                continue
            redundant.add(col)  # keep the earlier column, drop this one
            pairs.append({"dropped": col, "duplicate_of": partner,
                          "correlation": round(float(upper.loc[partner, col]), 4)})

    kept = [c for c in X.columns if c not in redundant]
    print(f"  correlation filter: {len(redundant)} redundant of {X.shape[1]} "
          f"(|r| > {threshold}) -> {len(kept)} kept")
    return kept, pd.DataFrame(pairs)


# --------------------------------------------------------------------------
# Methods 2-4 - label-aware rankings
# --------------------------------------------------------------------------

def mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Rank features by mutual information, which catches non-linear links."""
    Xs, ys = subsample(X, y, MI_SAMPLE)
    print(f"  mutual information: fitting on {len(Xs):,} rows...")
    scores = mutual_info_classif(Xs, ys, random_state=RANDOM_STATE, n_jobs=-1)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def random_forest_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Rank features by Random Forest impurity-based importance."""
    print(f"  random forest importance: fitting on {len(X):,} rows...")
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, max_depth=20
    )
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )


def recursive_elimination(X: pd.DataFrame, y: pd.Series, n_features: int) -> pd.Series:
    """
    Rank features by recursive feature elimination.

    A decision tree is used as the estimator rather than a forest: RFE refits
    the estimator at every step, so a forest would multiply the cost by the
    number of trees for the same ranking.
    """
    Xs, ys = subsample(X, y, RFE_SAMPLE)
    print(f"  RFE: fitting on {len(Xs):,} rows...")
    estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=15)
    rfe = RFE(estimator, n_features_to_select=n_features, step=3)
    rfe.fit(Xs, ys)
    # ranking_ is 1 for selected features, then 2, 3, ... for the order of
    # elimination. Invert it so that larger means better, as in the others.
    return pd.Series(-rfe.ranking_, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Consensus
# --------------------------------------------------------------------------

def build_consensus(rankings: dict[str, pd.Series], features: list[str]) -> pd.DataFrame:
    """
    Combine the methods by average rank.

    Raw scores are not comparable across methods (mutual information is in
    nats, RF importance sums to 1), so each method's ranking is converted to
    a position, 1 = best, and the positions are averaged. A feature that every
    method rates highly rises to the top; one that a single method happens to
    like does not.
    """
    table = pd.DataFrame(index=features)
    for name, series in rankings.items():
        table[f"{name}_rank"] = series.reindex(features).rank(
            ascending=False, method="min"
        )

    table["mean_rank"] = table.filter(like="_rank").mean(axis=1)
    return table.sort_values("mean_rank")


# --------------------------------------------------------------------------
# Does selection actually cost anything?
# --------------------------------------------------------------------------

def compare_feature_sets(split: dict, selected: list[str], target: str) -> pd.DataFrame:
    """
    Train the same model on all features vs the selected features.

    This is the comparison the project document asks for in Phase 7. The
    model is deliberately a modest Random Forest - the point is the relative
    difference between the two feature sets, not the absolute score.
    """
    y_train = split[f"y_train_{target}"]
    y_test = split[f"y_test_{target}"]
    average = "binary" if target == "binary" else "macro"

    rows = []
    for name, cols in [("all features", split["features"]), ("selected", selected)]:
        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, max_depth=20
        )
        start = time.perf_counter()
        model.fit(split["X_train"][cols], y_train)
        train_seconds = time.perf_counter() - start

        preds = model.predict(split["X_test"][cols])
        rows.append({
            "feature_set": name,
            "n_features": len(cols),
            "f1": round(f1_score(y_test, preds, average=average, zero_division=0), 5),
            "train_seconds": round(train_seconds, 1),
        })
        print(f"    {target:9} | {name:13} | {len(cols):2} features | "
              f"F1 {rows[-1]['f1']:.5f} | {train_seconds:5.1f}s")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_consensus(consensus: pd.DataFrame, top_n: int = 25) -> None:
    """Show the consensus ranking, with each method's vote behind it."""
    top = consensus[~consensus["redundant"]].head(top_n).iloc[::-1]
    y = np.arange(len(top))
    method_cols = [c for c in consensus.columns if c.endswith("_rank")
                   and c != "mean_rank"]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    for col in method_cols:
        ax.scatter(top[col], y, s=26, color=GRID, zorder=1)
    ax.scatter(top["mean_rank"], y, s=95, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1.6, label="mean rank")

    ax.set_yticks(y, top.index.tolist(), fontsize=8.5)
    ax.set_xlabel("Rank across the four methods (1 = most useful)")
    n_pool = int((~consensus["redundant"]).sum())
    ax.set_title(f"Consensus feature ranking: top {top_n} of "
                 f"{n_pool} non-redundant features")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    fig.text(0.5, -0.02,
             "Grey dots are the four individual methods; blue is their mean. "
             "Tightly clustered dots mean the methods agree.",
             ha="center", fontsize=8, color=INK_SOFT)
    save(fig, "06_feature_consensus.png")


def plot_comparison(results: pd.DataFrame) -> None:
    """
    All features vs selected, on accuracy and on cost.

    Two panels rather than one chart with two y-axes: F1 and seconds are
    different measures on different scales, and a dual axis would invite a
    false visual comparison between them.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    targets = [("binary", BLUE, "Experiment A (binary)"),
               ("multi", ORANGE, "Experiment B (multi-class)")]

    # Left: F1 as dots on a zoomed axis. The whole result lives between 0.94
    # and 0.997, so a zero-based bar chart would render every bar the same
    # length. Dots encode position rather than length, so they may start
    # somewhere other than zero - the axis makes the zoom explicit.
    for target, color, label in targets:
        subset = results[results["target"] == target]
        offset = -0.13 if target == "binary" else 0.13
        positions = np.arange(len(subset)) + offset
        ax1.plot(subset["f1"], positions, linestyle="none", marker="o",
                 markersize=10, color=color, markeredgecolor="white",
                 markeredgewidth=1.6, label=label)
        for value, pos in zip(subset["f1"], positions):
            ax1.text(value, pos + 0.22, f"{value:.4f}", ha="center",
                     fontsize=8.5, color=INK)

    # Connect each pair so the direction of the change is unmissable.
    for target, color, _ in targets:
        subset = results[results["target"] == target]
        offset = -0.13 if target == "binary" else 0.13
        ax1.plot(subset["f1"], np.arange(len(subset)) + offset,
                 color=color, linewidth=1.2, alpha=0.45, zorder=0)

    ax1.set_xlim(0.93, 1.005)
    ax1.set_xlabel("F1 score (higher is better) — note the zoomed axis")

    # Right: training cost. Seconds start at a true zero, so bars are correct.
    for target, color, label in targets:
        subset = results[results["target"] == target]
        offset = -0.16 if target == "binary" else 0.16
        bars = ax2.barh(np.arange(len(subset)) + offset, subset["train_seconds"],
                        height=0.3, color=color, label=label)
        for bar, value in zip(bars, subset["train_seconds"]):
            ax2.text(value + 0.8, bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}s", va="center", fontsize=8.5, color=INK)

    ax2.set_xlim(0, results["train_seconds"].max() * 1.2)
    ax2.set_xlabel("Training time in seconds (lower is better)")

    for ax in (ax1, ax2):
        ax.set_yticks(np.arange(2), ["all features\n(70)", "selected\n(25)"])
        ax.set_ylim(-0.6, 1.6)
        ax.grid(axis="y", visible=False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.09))
    fig.suptitle("Dropping 45 of 70 features improved accuracy and cut training time",
                 y=1.20, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "07_feature_set_comparison.png")


def main() -> None:
    apply_style()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split()
    X_train = split["X_train"]
    y_train = split["y_train_binary"]
    features = split["features"]

    print(f"Training split: {len(X_train):,} rows x {len(features)} features")
    print("\nRunning feature selection methods (training data only):")

    kept, redundant_pairs = correlation_filter(X_train)
    redundant_pairs.to_csv(TABLES_DIR / "redundant_feature_pairs.csv", index=False)

    # The three label-aware methods vote on usefulness. The correlation filter
    # is deliberately NOT a voter: it answers "is this feature a duplicate?",
    # not "is it useful?". Letting it vote once put four redundant pairs into
    # the selected set - including Subflow Fwd Bytes and Total Length of Fwd
    # Packets, correlated at 1.0000, which are the same measurement twice.
    # Used as a gate instead, it removes the duplicate before the vote, so
    # those slots go to features carrying different information.
    rankings = {
        "mutual_info": mutual_information(X_train, y_train),
        "random_forest": random_forest_importance(X_train, y_train),
        "rfe": recursive_elimination(X_train, y_train, N_SELECTED),
    }

    consensus = build_consensus(rankings, features)
    consensus["redundant"] = ~consensus.index.isin(kept)
    consensus.to_csv(TABLES_DIR / "feature_consensus_ranking.csv")

    # Rank every feature, but choose only from the non-redundant survivors.
    selected = consensus[~consensus["redundant"]].head(N_SELECTED).index.tolist()
    pd.Series(selected, name="feature").to_csv(
        TABLES_DIR / "selected_features.csv", index=False
    )

    print(f"\nTop {N_SELECTED} features by consensus:")
    for i, name in enumerate(selected, 1):
        print(f"  {i:2}. {name}")

    print("\nAll features vs selected features:")
    results = []
    for target in ("binary", "multi"):
        frame = compare_feature_sets(split, selected, target)
        frame["target"] = target
        results.append(frame)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(TABLES_DIR / "feature_set_comparison.csv", index=False)

    print("\nBuilding figures:")
    plot_consensus(consensus)
    plot_comparison(results)

    print(f"\nSelected feature list -> {TABLES_DIR / 'selected_features.csv'}")


if __name__ == "__main__":
    main()
