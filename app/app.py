"""
Phase 18 - Streamlit demonstration interface.

A thin layer over the models saved in Phase 17. It does no training and holds
no logic of its own: it loads the bundles from models/, collects the 25
features, and shows what the model predicted and why.

The "why" matters here. An intrusion detection system that only outputs
"ATTACK" is not actionable - an analyst needs the reason before they can
decide whether to act on it. Every prediction is shown with its SHAP
contributions, drawn as a diverging bar chart rather than a table so the
direction and relative weight of each feature read at a glance.

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
                   layout="wide", initial_sidebar_state="collapsed")

# Palette: a dark operations-console ground, with attack/benign carried by
# semantic colour and never by colour alone - every state also has a label.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --ground:    #0e1420;
  --surface:   #161d2b;
  --surface-2: #1e2738;
  --line:      #2a3547;
  --ink:       #e9eef6;
  --ink-2:     #9aa7bd;
  --ink-3:     #6b7891;
  --accent:    #4a92e8;
  --attack:    #e2564a;
  --benign:    #2fb886;
  --warn:      #d9a15c;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--ground); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.4rem 2.4rem 3rem; max-width: 1400px; }

/* ---------- masthead ---------- */
.mast {
  display: flex; align-items: center; gap: 16px;
  padding: 0 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 22px;
}
.mast .glyph {
  width: 46px; height: 46px; border-radius: 8px; flex: none;
  background: linear-gradient(145deg, #2a5ea8, #4a92e8);
  display: grid; place-items: center; font-size: 24px;
}
.mast h1 {
  font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: -0.01em;
  color: var(--ink); line-height: 1.2;
}
.mast p { margin: 3px 0 0; color: var(--ink-2); font-size: 0.85rem; max-width: 78ch; }
.pill {
  margin-left: auto; flex: none;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--benign); border: 1px solid var(--benign);
  padding: 5px 11px; border-radius: 100px; background: rgba(47,184,134,0.09);
}

/* ---------- metric strip ---------- */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 26px; }
.metric {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 8px; padding: 14px 16px 13px;
}
.metric .k {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3);
  display: block; margin-bottom: 5px;
}
.metric .v {
  font-size: 1.55rem; font-weight: 600; color: var(--ink);
  font-variant-numeric: tabular-nums; line-height: 1.1;
}
.metric .v span { font-size: 0.8rem; color: var(--ink-3); font-weight: 400; }

/* ---------- panels ---------- */
.panel-title {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3);
  margin: 0 0 12px; font-weight: 500;
}

/* ---------- verdict ---------- */
.verdict {
  border-radius: 10px; padding: 22px 24px; margin-bottom: 8px;
  border: 1px solid; position: relative; overflow: hidden;
}
.verdict.attack { background: rgba(226,86,74,0.10); border-color: rgba(226,86,74,0.45); }
.verdict.benign { background: rgba(47,184,134,0.10); border-color: rgba(47,184,134,0.42); }
.verdict .label {
  font-size: 1.85rem; font-weight: 700; letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 11px; line-height: 1;
}
.verdict.attack .label { color: var(--attack); }
.verdict.benign .label { color: var(--benign); }
.verdict .sub {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
  color: var(--ink-2); margin-top: 11px;
}
.meter { height: 5px; border-radius: 3px; background: var(--surface-2); margin-top: 13px; overflow: hidden; }
.meter i { display: block; height: 100%; border-radius: 3px; }
.verdict.attack .meter i { background: var(--attack); }
.verdict.benign .meter i { background: var(--benign); }

.truth {
  font-size: 0.8rem; color: var(--ink-3); margin: 10px 0 24px;
  font-family: 'IBM Plex Mono', monospace;
}
.truth b { color: var(--ink-2); font-weight: 500; }
.truth .ok  { color: var(--benign); }
.truth .bad { color: var(--attack); }

/* ---------- attack type ranking ---------- */
.rank { display: flex; flex-direction: column; gap: 9px; margin-bottom: 26px; }
.rank-row { display: grid; grid-template-columns: 20px 132px 1fr 62px; align-items: center; gap: 11px; }
.rank-row .n { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--ink-3); }
.rank-row .nm { font-size: 0.87rem; color: var(--ink); font-weight: 500; }
.rank-row .track { height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
.rank-row .track i { display: block; height: 100%; background: var(--accent); border-radius: 4px; }
.rank-row:first-child .track i { background: var(--attack); }
.rank-row .p {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
  color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums;
}

/* ---------- SHAP diverging bars ---------- */
.shap { display: flex; flex-direction: column; gap: 3px; }
.shap-row {
  display: grid; grid-template-columns: 200px 1fr 66px;
  align-items: center; gap: 12px; padding: 6px 0;
  border-bottom: 1px solid rgba(42,53,71,0.5);
}
.shap-row:last-child { border-bottom: 0; }
.shap-name { font-size: 0.8rem; color: var(--ink); line-height: 1.25; }
.shap-name em {
  display: block; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem; color: var(--ink-3); font-style: normal; margin-top: 1px;
}
.shap-track { position: relative; height: 17px; }
.shap-track::before {
  content: ''; position: absolute; left: 50%; top: 0; bottom: 0;
  width: 1px; background: var(--line);
}
.shap-bar { position: absolute; top: 3px; height: 11px; border-radius: 2px; }
.shap-bar.pos { left: 50%; background: var(--attack); }
.shap-bar.neg { right: 50%; background: var(--accent); }
.shap-val {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
  text-align: right; font-variant-numeric: tabular-nums;
}
.shap-val.pos { color: var(--attack); }
.shap-val.neg { color: var(--accent); }
.shap-key {
  display: flex; gap: 20px; margin-top: 14px;
  font-size: 0.74rem; color: var(--ink-3);
}
.shap-key span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

/* ---------- streamlit widget tweaks ---------- */
div[data-testid="stRadio"] label p { font-size: 0.85rem; }
.stSelectbox label, .stRadio > label { color: var(--ink-2) !important; font-size: 0.8rem !important; }
div[data-baseweb="select"] > div {
  background: var(--surface) !important; border-color: var(--line) !important;
}
.stExpander {
  border: 1px solid var(--line) !important; border-radius: 8px !important;
  background: var(--surface) !important;
}
hr { border-color: var(--line); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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
    so the interface opens on genuine test-set flows it has never seen.
    """
    x_path, y_path = SPLIT_DIR / "X_test.parquet", SPLIT_DIR / "y_test.parquet"
    if not (x_path.exists() and y_path.exists()):
        return None

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    X = X.assign(_actual=y["multi"].to_numpy(), _binary=y["binary"].to_numpy())

    picks = [X[X["_actual"] == label].head(3) for label in sorted(X["_actual"].unique())]
    return pd.concat(picks).reset_index(drop=True)


def shap_contributions(bundle: dict, row: pd.DataFrame) -> pd.Series | None:
    """SHAP values for a single flow, or None if shap is unavailable."""
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(bundle["model"])
    values = np.asarray(explainer.shap_values(row[bundle["features"]]))
    if values.ndim == 3:            # multi-class: (rows, features, classes)
        values = np.abs(values[0]).sum(axis=1)
    else:
        values = values[0]
    return pd.Series(values, index=bundle["features"])


def fmt(value: float) -> str:
    """Compact display for flow statistics that span many orders of magnitude."""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.2f}M"
    if abs(value) >= 10_000:
        return f"{value/1_000:,.1f}K"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


bundles = load_models()

if not bundles:
    st.error("No saved models found in `models/`. Run `python src/save_model.py` first.")
    st.stop()

binary_metrics = bundles.get("binary", {}).get("metrics", {})
multi_metrics = bundles.get("multi", {}).get("metrics", {})

st.markdown(
    """
    <div class="mast">
      <div class="glyph">🛡️</div>
      <div>
        <h1>Network Intrusion Detection</h1>
        <p>XGBoost over 25 flow features, trained on 320,032 CIC-IDS2017 flows.
           Every score below was measured on 80,008 held-out flows the model never saw.</p>
      </div>
      <div class="pill">model loaded</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metrics">
      <div class="metric"><span class="k">Binary F1</span>
        <div class="v">{binary_metrics.get('f1', float('nan')):.4f}</div></div>
      <div class="metric"><span class="k">Binary ROC-AUC</span>
        <div class="v">{binary_metrics.get('roc_auc', float('nan')):.6f}</div></div>
      <div class="metric"><span class="k">Multi-class F1</span>
        <div class="v">{multi_metrics.get('f1', float('nan')):.4f}</div></div>
      <div class="metric"><span class="k">Attack classes</span>
        <div class="v">{len(bundles['multi']['classes']) - 1} <span>+ benign</span></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

examples = load_examples()
left, right = st.columns([1, 1.5], gap="large")

with left:
    st.markdown('<p class="panel-title">Select a flow</p>', unsafe_allow_html=True)

    if examples is None:
        st.warning("Test-set examples not found. Run `python src/datasets.py`.")
        source = "Manual entry"
    else:
        source = st.radio("Source", ["A real held-out flow", "Manual entry"],
                          label_visibility="collapsed", horizontal=True)

    features = bundles["binary"]["features"]

    if source == "A real held-out flow" and examples is not None:
        labels = [f"#{i} — actually {row['_actual']}" for i, row in examples.iterrows()]
        choice = st.selectbox("Example flow", range(len(labels)),
                              format_func=lambda i: labels[i])
        flow = examples.iloc[[choice]][features].copy()
        actual = examples.iloc[choice]["_actual"]
    else:
        st.caption("Values default to the median of the test data.")
        defaults = (examples[features].median() if examples is not None
                    else pd.Series(0.0, index=features))
        values = {name: st.number_input(name, value=float(defaults[name]), format="%.4f")
                  for name in features}
        flow = pd.DataFrame([values])
        actual = None

with right:
    st.markdown('<p class="panel-title">Verdict</p>', unsafe_allow_html=True)

    binary_bundle = bundles["binary"]
    proba = binary_bundle["model"].predict_proba(flow[binary_bundle["features"]])[0]
    is_attack = proba[1] >= 0.5
    confidence = proba[1] if is_attack else proba[0]

    st.markdown(
        f"""
        <div class="verdict {'attack' if is_attack else 'benign'}">
          <div class="label">{'⚠' if is_attack else '✓'} {'ATTACK' if is_attack else 'BENIGN'}</div>
          <div class="sub">confidence {confidence:.4f}</div>
          <div class="meter"><i style="width:{confidence*100:.1f}%"></i></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if actual is not None:
        correct = (actual == "BENIGN") != is_attack
        st.markdown(
            f'<p class="truth">ground truth <b>{actual}</b> · '
            f'<span class="{"ok" if correct else "bad"}">'
            f'{"correct" if correct else "incorrect"}</span></p>',
            unsafe_allow_html=True,
        )

    if is_attack and "multi" in bundles:
        multi_bundle = bundles["multi"]
        multi_proba = multi_bundle["model"].predict_proba(
            flow[multi_bundle["features"]]
        )[0]
        classes = multi_bundle["classes"]
        order = np.argsort(multi_proba)[::-1][:3]

        st.markdown('<p class="panel-title">Most likely attack type</p>',
                    unsafe_allow_html=True)
        rows = "".join(
            f'<div class="rank-row"><span class="n">{rank}</span>'
            f'<span class="nm">{classes[i]}</span>'
            f'<span class="track"><i style="width:{multi_proba[i]*100:.1f}%"></i></span>'
            f'<span class="p">{multi_proba[i]:.4f}</span></div>'
            for rank, i in enumerate(order, 1)
        )
        st.markdown(f'<div class="rank">{rows}</div>', unsafe_allow_html=True)

    st.markdown('<p class="panel-title">Why — SHAP contributions</p>',
                unsafe_allow_html=True)

    contributions = shap_contributions(binary_bundle, flow)
    if contributions is None:
        st.caption("Install `shap` to see per-feature explanations.")
    else:
        top = contributions.reindex(
            contributions.abs().sort_values(ascending=False).index
        ).head(9)
        scale = float(top.abs().max()) or 1.0

        rows = ""
        for name, value in top.items():
            width = abs(value) / scale * 48
            sign = "pos" if value > 0 else "neg"
            rows += (
                f'<div class="shap-row">'
                f'<div class="shap-name">{name}<em>{fmt(float(flow.iloc[0][name]))}</em></div>'
                f'<div class="shap-track">'
                f'<div class="shap-bar {sign}" style="width:{width:.2f}%"></div></div>'
                f'<div class="shap-val {sign}">{value:+.3f}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div class="shap">{rows}</div>'
            f'<div class="shap-key">'
            f'<span><i class="swatch" style="background:var(--attack)"></i> pushes toward ATTACK</span>'
            f'<span><i class="swatch" style="background:var(--accent)"></i> pushes toward BENIGN</span>'
            f'<span>contributions sum to the model output</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("How this model was built"):
    st.markdown(
        f"""
- **Data** — 2,830,743 CIC-IDS2017 flows, reduced to 2,498,185 after removing
  329,691 duplicate rows that would otherwise leak across the train/test split.
- **Features** — {len(bundles['binary']['features'])} of 70, chosen by a consensus of
  mutual information, Random Forest importance and recursive elimination, all fitted
  on training data only. The 25-feature set outscored all 70 on both experiments.
- **Model** — XGBoost, `{bundles['binary']['params']}`. Randomised search ran over 15
  candidates; every gain fell inside the fold spread, and the tuned multi-class model
  scored *lower* on the test set, so the defaults ship.
- **Honest scoring** — the test set was used exactly once, after every other decision
  was made. Binary ROC-AUC is {binary_metrics.get('roc_auc', 0):.6f}, reported at six
  decimals because three flows in this dataset carry identical features with
  conflicting labels, which makes a perfect score impossible.
- **Known limits** — Infiltration recall is 0.714 and Bot F1 is 0.8136; these are the
  weakest classes and the ones to watch in any real deployment.
"""
    )
