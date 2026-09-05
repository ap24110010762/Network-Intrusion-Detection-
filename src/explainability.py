"""
Phase 15-16 - Feature importance and SHAP explanations.

Phase 15 compares two views of importance, because they answer different
questions and often disagree:

    gain          how much each feature reduced impurity while the model was
                  being built. Cheap, but biased toward features with many
                  distinct values, and it describes the TRAINING data.
    permutation   how much test-set F1 drops when one feature's values are
                  shuffled. Slower, but it measures what the model actually
                  relies on when predicting unseen traffic.

Phase 16 uses SHAP, which answers a question neither of the above can: not
"which features matter overall" but "why was THIS flow called an attack".
Every prediction is decomposed into a contribution per feature, and those
contributions sum exactly to the model's output for that flow.

Both phases load the saved models from Phase 17 rather than refitting, so the
explanations describe the model that ships.

Usage (from the project root):
    python src/explainability.py
"""

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from data_loader import PROJECT_ROOT, RESULTS_DIR
from datasets import load_split
from eda import BLUE, GRID, INK, INK_SOFT, ORANGE, apply_style, save

MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
PERMUTATION_SAMPLE = 20_000   # permutation importance refits nothing but predicts a lot
SHAP_SAMPLE = 3_000           # beeswarm plots become unreadable well before this


def load_bundle(name: str) -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python src/save_model.py' first.")
    return joblib.load(path)


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int):
    """Subsample while keeping every class represented."""
    if len(X) <= n:
        return X, y
    rate = n / len(X)
    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for label in pd.unique(y):
        idx = np.flatnonzero((y == label).to_numpy())
        take = max(1, int(round(len(idx) * rate)))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep], y.iloc[keep]


# --------------------------------------------------------------------------
# Phase 15 - importance
# --------------------------------------------------------------------------

def compute_importance(bundle: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Gain-based and permutation importance for one model."""
    model, features = bundle["model"], bundle["features"]

    gain = pd.Series(model.feature_importances_, index=features)

    Xs, ys = stratified_sample(X_test[features], y_test, PERMUTATION_SAMPLE)
    if bundle["task"] == "multi":
        lookup = {name: i for i, name in enumerate(bundle["classes"])}
        ys = ys.map(lookup)
    scorer = "f1" if bundle["task"] == "binary" else "f1_macro"

    print(f"    permutation importance on {len(Xs):,} test rows ...", end="", flush=True)
    result = permutation_importance(
        model, Xs, ys, n_repeats=5, random_state=RANDOM_STATE,
        scoring=scorer, n_jobs=1,
    )
    print(" done")

    frame = pd.DataFrame({
        "gain": gain / gain.sum(),
        "permutation": result.importances_mean,
        "permutation_std": result.importances_std,
    }, index=features)
    frame["gain_rank"] = frame["gain"].rank(ascending=False)
    frame["permutation_rank"] = frame["permutation"].rank(ascending=False)
    frame["rank_shift"] = frame["gain_rank"] - frame["permutation_rank"]
    return frame.sort_values("permutation", ascending=False)


def plot_importance(frame: pd.DataFrame, task: str, filename: str) -> None:
    """
    The two importance measures side by side.

    Both are normalised to their own maximum, because gain is a share of total
    impurity reduction and permutation is a drop in F1 - different units that
    must not share an axis. What matters is the ordering, and where the two
    disagree.
    """
    top = frame.head(15).iloc[::-1]
    y = np.arange(len(top))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)

    ax1.barh(y, top["gain"] / top["gain"].max(), height=0.6, color=GRID)
    ax1.set_xlabel("Gain importance (relative)")
    ax1.set_title("Built from training data", fontsize=11)

    ax2.barh(y, top["permutation"] / top["permutation"].max(), height=0.6, color=BLUE)
    ax2.set_xlabel("Permutation importance (relative)")
    ax2.set_title("Measured on held-out test data", fontsize=11)

    for i, row in enumerate(top.itertuples()):
        shift = int(row.rank_shift)
        if abs(shift) >= 4:
            ax2.text(
                1.02, i,
                f"{'↑' if shift > 0 else '↓'}{abs(shift)} vs gain",
                va="center", fontsize=7.5,
                color=ORANGE if shift > 0 else INK_SOFT,
            )

    ax1.set_yticks(y, top.index.tolist(), fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    ax2.set_xlim(0, 1.24)

    name = "Experiment A — binary" if task == "binary" else "Experiment B — multi-class"
    fig.suptitle(f"{name}: what the model was built on vs what it relies on",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, -0.04,
             "Arrows mark features that move four or more places between the two "
             "measures. Gain describes construction; permutation describes behaviour.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    save(fig, filename)


# --------------------------------------------------------------------------
# Phase 16 - SHAP
# --------------------------------------------------------------------------

def shap_global(bundle: dict, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Global SHAP summary for the binary model."""
    model, features = bundle["model"], bundle["features"]
    Xs, _ = stratified_sample(X_test[features], y_test, SHAP_SAMPLE)

    print(f"    SHAP on {len(Xs):,} test flows ...", end="", flush=True)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(Xs)
    print(" done")

    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("Which features push a flow toward 'attack', and in which direction",
                 y=1.01, fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.03,
             "Each dot is one test flow. Position is that feature's contribution to the "
             "prediction;\ncolour is the feature's own value — red high, blue low.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout()
    save(fig, "19_shap_global.png")

    return explanation


def shap_local(bundle: dict, explanation, X_test: pd.DataFrame,
               y_test: pd.Series) -> None:
    """
    Explain one individual flow.

    A global summary says what matters on average. An analyst looking at a
    single alert needs the reason for THAT flow, which is what a waterfall
    plot gives: the contributions that moved this one prediction away from the
    model's baseline.
    """
    model, features = bundle["model"], bundle["features"]

    # Choose a confidently-detected attack rather than an arbitrary row, so the
    # explanation shows the mechanism clearly.
    proba = model.predict_proba(X_test[features])[:, 1]
    attacks = np.flatnonzero((y_test.to_numpy() == 1))
    chosen = attacks[np.argmax(proba[attacks])]

    explainer = shap.TreeExplainer(model)
    single = explainer(X_test[features].iloc[[chosen]])

    plt.figure(figsize=(9.5, 6))
    shap.plots.waterfall(single[0], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle("Why this individual flow was classified as an attack",
                 y=1.02, fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.04,
             f"Test flow #{chosen}, predicted attack with probability "
             f"{proba[chosen]:.4f}. Contributions sum to the model's output.",
             ha="center", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout()
    save(fig, "20_shap_local.png")

    contributions = pd.Series(single.values[0], index=features).sort_values(
        key=np.abs, ascending=False
    )
    print("\n    Top contributions for this flow:")
    for name, value in contributions.head(6).items():
        direction = "toward ATTACK" if value > 0 else "toward BENIGN"
        print(f"      {name:32} {value:+.4f}  {direction}")


def main() -> None:
    apply_style()
    split = load_split()
    X_test, y_binary, y_multi = split["X_test"], split["y_test_binary"], split["y_test_multi"]

    print("=" * 78)
    print("PHASE 15 - FEATURE IMPORTANCE")
    print("=" * 78)

    for task, filename, y in [
        ("binary", "18_feature_importance_binary.png", y_binary),
        ("multi", "18b_feature_importance_multi.png", y_multi),
    ]:
        bundle = load_bundle(
            "binary_model.pkl" if task == "binary" else "multiclass_model.pkl"
        )
        print(f"\n  {task}:")
        frame = compute_importance(bundle, X_test, y)
        frame.to_csv(TABLES_DIR / f"feature_importance_{task}.csv")
        plot_importance(frame, task, filename)

        print(f"\n    Top 8 by permutation importance:")
        for i, (name, row) in enumerate(frame.head(8).iterrows(), 1):
            print(f"      {i}. {name:32} {row['permutation']:.5f}")

        movers = frame[frame["rank_shift"].abs() >= 5]
        if not movers.empty:
            print(f"\n    Features whose rank shifts 5+ places between the two measures:")
            for name, row in movers.iterrows():
                print(f"      {name:32} gain #{int(row['gain_rank']):2} -> "
                      f"permutation #{int(row['permutation_rank']):2}")

    print("\n" + "=" * 78)
    print("PHASE 16 - SHAP EXPLANATIONS")
    print("=" * 78 + "\n")

    bundle = load_bundle("binary_model.pkl")
    explanation = shap_global(bundle, X_test, y_binary)
    shap_local(bundle, explanation, X_test, y_binary)


if __name__ == "__main__":
    main()
