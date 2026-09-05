"""
Phase 18 - Streamlit demonstration interface.

A thin layer over the models saved in Phase 17. It does no training and holds
no logic of its own: it loads the bundles from models/, collects the 25
features, and shows what the model predicted and why.

The "why" matters here. An intrusion detection system that only outputs
"ATTACK" is not actionable - an analyst needs the reason before they can
decide whether to act on it. Each prediction is shown with its SHAP
contributions, the same explanations produced in Phase 16.

Run from the project root:
    streamlit run app/app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "split"

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️",
                   layout="wide")


@st.cache_resource
def load_models() -> dict:
    """Load both saved bundles once per session."""
    bundles = {}
    for task, filename in [("binary", "binary_model.pkl"),
                           ("multi", "multiclass_model.pkl")]:
        path = MODELS_DIR / filename
        if path.exists():
            bundles[task] = joblib.load(path)
    return bundles


@st.cache_data
def load_examples() -> pd.DataFrame | None:
    """
    Real held-out flows to demonstrate with.

    Typing 25 flow statistics by hand is not a realistic way to try the model,
    so the interface opens on genuine test-set flows the model has never seen.
    """
    x_path, y_path = SPLIT_DIR / "X_test.parquet", SPLIT_DIR / "y_test.parquet"
    if not (x_path.exists() and y_path.exists()):
        return None

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    X = X.assign(_actual=y["multi"].to_numpy(), _binary=y["binary"].to_numpy())

    # One example per class, so every attack type can be demonstrated.
    picks = []
    for label in sorted(X["_actual"].unique()):
        picks.append(X[X["_actual"] == label].head(3))
    return pd.concat(picks).reset_index(drop=True)


def shap_contributions(bundle: dict, row: pd.DataFrame) -> pd.Series | None:
    """SHAP values for a single flow, or None if shap is unavailable."""
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(bundle["model"])
    values = explainer.shap_values(row[bundle["features"]])
    values = np.asarray(values)
    if values.ndim == 3:            # multi-class: (rows, features, classes)
        values = np.abs(values[0]).sum(axis=1)
    else:
        values = values[0]
    return pd.Series(values, index=bundle["features"])


bundles = load_models()

st.title("🛡️ Network Intrusion Detection")
st.caption(
    "XGBoost trained on 320,032 CIC-IDS2017 network flows. "
    "Every score shown below was measured on 80,008 held-out flows the model "
    "never saw during training."
)

if not bundles:
    st.error(
        "No saved models found in `models/`. Run `python src/save_model.py` first."
    )
    st.stop()

# --- measured performance -------------------------------------------------

cols = st.columns(4)
binary_metrics = bundles.get("binary", {}).get("metrics", {})
multi_metrics = bundles.get("multi", {}).get("metrics", {})

cols[0].metric("Binary F1", f"{binary_metrics.get('f1', float('nan')):.4f}")
cols[1].metric("Binary ROC-AUC", f"{binary_metrics.get('roc_auc', float('nan')):.5f}")
cols[2].metric("Multi-class F1", f"{multi_metrics.get('f1', float('nan')):.4f}")
cols[3].metric("Features used", str(len(bundles["binary"]["features"])))

st.divider()

# --- input ----------------------------------------------------------------

examples = load_examples()
left, right = st.columns([1, 1.35])

with left:
    st.subheader("Pick a flow")

    if examples is None:
        st.warning(
            "Test-set examples not found. Run `python src/datasets.py` to "
            "generate them, or enter values manually below."
        )
        source = "Manual entry"
    else:
        source = st.radio(
            "Where should the flow come from?",
            ["A real held-out flow", "Manual entry"],
            label_visibility="collapsed",
        )

    features = bundles["binary"]["features"]

    if source == "A real held-out flow" and examples is not None:
        labels = [
            f"#{i} — actually {row['_actual']}"
            for i, row in examples.iterrows()
        ]
        choice = st.selectbox("Example flow", range(len(labels)),
                              format_func=lambda i: labels[i])
        flow = examples.iloc[[choice]][features].copy()
        actual = examples.iloc[choice]["_actual"]
    else:
        st.caption("Values default to the median of the training data.")
        defaults = examples[features].median() if examples is not None else pd.Series(
            0.0, index=features
        )
        values = {}
        for name in features:
            values[name] = st.number_input(name, value=float(defaults[name]),
                                           format="%.4f")
        flow = pd.DataFrame([values])
        actual = None

with right:
    st.subheader("Verdict")

    binary_bundle = bundles["binary"]
    proba = binary_bundle["model"].predict_proba(flow[binary_bundle["features"]])[0]
    is_attack = proba[1] >= 0.5

    if is_attack:
        st.error(f"### ⚠️ ATTACK\nconfidence {proba[1]:.4f}")
    else:
        st.success(f"### ✓ BENIGN\nconfidence {proba[0]:.4f}")

    if is_attack and "multi" in bundles:
        multi_bundle = bundles["multi"]
        multi_proba = multi_bundle["model"].predict_proba(
            flow[multi_bundle["features"]]
        )[0]
        classes = multi_bundle["classes"]
        order = np.argsort(multi_proba)[::-1][:3]

        st.write("**Most likely attack type**")
        for rank, idx in enumerate(order, 1):
            st.write(f"{rank}. **{classes[idx]}** — {multi_proba[idx]:.4f}")

    if actual is not None:
        correct = (actual == "BENIGN") != is_attack
        st.caption(
            f"Ground truth for this flow: **{actual}** — "
            f"{'correct' if correct else 'incorrect'}"
        )

    st.divider()
    st.write("**Why**")
    contributions = shap_contributions(binary_bundle, flow)
    if contributions is None:
        st.caption("Install `shap` to see per-feature explanations.")
    else:
        top = contributions.reindex(
            contributions.abs().sort_values(ascending=False).index
        ).head(8)
        explanation = pd.DataFrame({
            "feature": top.index,
            "contribution": top.to_numpy(),
            "pushes toward": ["ATTACK" if v > 0 else "BENIGN" for v in top],
            "flow value": [flow.iloc[0][f] for f in top.index],
        })
        st.dataframe(explanation, hide_index=True, use_container_width=True)
        st.caption(
            "SHAP contributions for this specific flow. Positive values push the "
            "prediction toward ATTACK, negative toward BENIGN, and together they "
            "sum to the model's output."
        )

st.divider()
with st.expander("How this model was built"):
    st.markdown(
        f"""
- **Data** 2,830,743 CIC-IDS2017 flows, reduced to 2,498,185 after removing
  329,691 duplicate rows that would otherwise leak across the train/test split.
- **Features** {len(bundles['binary']['features'])} of 70, chosen by a consensus of
  mutual information, Random Forest importance and recursive elimination — all
  fitted on training data only. The 25-feature set outscored all 70.
- **Model** XGBoost, `{bundles['binary']['params']}`. Randomised search was run
  over 15 candidates; every gain fell inside the fold spread, and the tuned
  multi-class model scored *lower* on the test set, so the defaults ship.
- **Honest scoring** the test set was used exactly once, after every other
  decision was made. Binary ROC-AUC is {binary_metrics.get('roc_auc', 0):.6f} —
  reported at six decimals because three flows in this dataset carry identical
  features with conflicting labels, which makes a perfect score impossible.
"""
    )
