"""
Generate the eight notebooks listed in the project document, Section 10.

The notebooks are built from this script rather than hand-edited so they stay
consistent with each other and can be regenerated whenever the pipeline
changes. Run from the notebooks/ folder:

    python _build_notebooks.py

Design
------
Each notebook imports the real pipeline from src/ and reads the artefacts the
pipeline produced. It does not re-implement anything, and it does not re-run
the expensive stages: training six models takes minutes and the randomised
search took over an hour, so those results are read from results/tables/ and
results/figures/ with the command that produced them stated in the notebook.

That keeps every notebook runnable in seconds, from a clone, without the
844 MB dataset - while still showing the code path that produced each number.
"""

import itertools
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent

PREAMBLE = """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from IPython.display import Image, display

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

def table(name, **kw):
    \"\"\"Read one of the pipeline's result tables.\"\"\"
    return pd.read_csv(TABLES / name, **kw)

def figure(name):
    \"\"\"Display one of the pipeline's figures.\"\"\"
    return Image(filename=str(FIGURES / name))

print(f"project root: {ROOT}")
"""


def _lines(text: str) -> list[str]:
    """
    Split into nbformat's source format: a list of lines, each KEEPING its
    trailing newline except the last. Splitting on "\\n" without keeping the
    separators concatenates every line into one, which turns each code cell
    into a single unparseable line.
    """
    return text.splitlines(keepends=True)


_counter = itertools.count(1)


def _cell_id() -> str:
    """nbformat 4.5 requires a unique id per cell."""
    return f"cell-{next(_counter):04d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": _lines(text.strip()),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text.strip("\n")),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def header(number: str, title: str, summary: str, produced_by: str) -> list[dict]:
    return [
        md(f"""
# {number} — {title}

{summary}

**Pipeline module:** `{produced_by}`

*Every number and figure below was produced by the pipeline in `src/`. This
notebook reads those results; it does not re-implement them.*
"""),
        code(PREAMBLE),
    ]


# --------------------------------------------------------------------------
NOTEBOOKS = {}

# ---- 01 -------------------------------------------------------------------
NOTEBOOKS["01_data_understanding.ipynb"] = notebook(
    header(
        "01", "Data understanding",
        "CIC-IDS2017 was captured over five working days in July 2017. Monday is "
        "benign traffic only; attacks were staged Tuesday through Friday. Eight CSV "
        "files, 80 columns each, 2,830,743 rows in total.",
        "src/data_loader.py",
    )
    + [
        md("""
## The capture files

Each file is one period of traffic. The misspelling in `Infilteration` is the
dataset authors' own — the filenames are left exactly as published.
"""),
        code("""
files = pd.DataFrame([
    ("Monday-WorkingHours",               529_918, "Benign only"),
    ("Tuesday-WorkingHours",              445_909, "FTP / SSH brute force"),
    ("Wednesday-workingHours",            692_703, "DoS variants, Heartbleed"),
    ("Thursday-Morning-WebAttacks",       170_366, "XSS, SQL injection, brute force"),
    ("Thursday-Afternoon-Infilteration",  288_602, "Infiltration"),
    ("Friday-Morning",                    191_033, "Botnet"),
    ("Friday-Afternoon-PortScan",         286_467, "Port scanning"),
    ("Friday-Afternoon-DDos",             225_745, "DDoS"),
], columns=["file", "rows", "contains"])

files.loc[len(files)] = ("TOTAL", files["rows"].sum(), "")
files
"""),
        md("""
## Memory: the first real constraint

This project was built on a machine with 7.8 GB of RAM. Loaded naively the merged
frame needs roughly 1.7 GB, Windows takes about 3 GB, and pandas makes copies during
a concatenation — so a straightforward merge exhausts memory.

`data_loader.py` loads and shrinks one file at a time, downcasting `float64 → float32`
and storing labels as categories. The merged dataset then occupies **686 MB**.
"""),
        code("""
import inspect
import data_loader

print(inspect.getsource(data_loader.downcast_numeric))
"""),
        md("""
## Running the profile

With the eight CSVs in `data/`, this reproduces the full profile — row and column
counts, dtypes, missing values, infinities, duplicates and the class distribution:

```bash
python src/data_loader.py
```

It takes a few minutes on the full 2.83 million rows, so the resulting class
distribution is read below rather than recomputed.
"""),
        code("""
raw_counts = table("class_distribution.csv", index_col=0)
raw_counts["percent"] = (raw_counts["count"] / raw_counts["count"].sum() * 100).round(4)
raw_counts
"""),
        md("""
## What the profile found

Four problems in the published data, each handled in notebook 03:

| Problem | Extent |
|---|---|
| Corrupted label text | 2,180 rows — `Web Attack` labels contain bytes `EF BF BD` where an en-dash belongs |
| Infinite values | `Flow Bytes/s` and `Flow Packets/s` divide by a zero-length flow duration |
| Duplicate rows | 11.66% of the dataset |
| Zero-variance columns | 8 columns CICFlowMeter never populated |

The class imbalance is the defining property of this dataset: **BENIGN is 80.3% of
raw rows, and Heartbleed has 11 rows in total** — a ratio of roughly 206,000:1.
"""),
    ]
)

# ---- 02 -------------------------------------------------------------------
NOTEBOOKS["02_eda.ipynb"] = notebook(
    header(
        "02", "Exploratory data analysis",
        "Class balance, feature distributions, correlation structure, and which "
        "single features separate benign traffic from attacks.",
        "src/eda.py",
    )
    + [
        md("""
## Class distribution after cleaning

A logarithmic axis is unavoidable here. On a linear scale every class below DoS would
be an invisible sliver against BENIGN's 2,072,444 rows.
"""),
        code('figure("01_class_distribution_multiclass.png")'),
        code("""
clean_counts = table("class_distribution_clean.csv", index_col=0)
clean_counts["percent"] = (clean_counts["count"] / clean_counts["count"].sum() * 100).round(4)
clean_counts
"""),
        md("""
## The binary target

82.96% of flows are benign. A model that predicts "BENIGN" for everything therefore
scores **82.96% accuracy while detecting nothing** — which is why this project reports
precision, recall, F1 and PR-AUC rather than accuracy alone.
"""),
        code('figure("02_class_distribution_binary.png")'),
        md("""
## Correlation structure

Two large blocks are visible: a packet-size cluster and a timing (inter-arrival)
cluster. Features inside each block carry nearly the same information — `Avg Bwd
Segment Size` and `Bwd Packet Length Mean` correlate at almost 1.0 because they are
the same quantity computed twice. This is the evidence for the feature selection in
notebook 03.
"""),
        code('figure("03_correlation_heatmap.png")'),
        md("""
## Which features separate attacks from benign traffic
"""),
        code("""
corr = table("feature_target_correlation.csv", index_col=0)
corr.head(12).round(4)
"""),
        code('figure("05_feature_separation.png")'),
        md("""
## Distributions, benign vs attack

The separation is real but not clean. Attack traffic spikes at the far right of
`Flow Duration` and `Flow IAT Mean` where benign traffic is thin — but the two
overlap heavily in the middle, which is why a threshold rule is not enough and a
model is needed.
"""),
        code('figure("04_feature_distributions.png")'),
    ]
)

# ---- 03 -------------------------------------------------------------------
NOTEBOOKS["03_preprocessing.ipynb"] = notebook(
    header(
        "03", "Preprocessing, feature selection and the split",
        "Cleaning decisions with before/after counts, the stratified sample and "
        "train/test split, the leakage check, and the four-method feature selection.",
        "src/preprocessing.py, src/datasets.py, src/feature_selection.py",
    )
    + [
        md("""
## Cleaning log

Every decision is recorded with the rows it removed. The project brief requires each
cleaning decision to be documented; this table is that record.
"""),
        code("""
log = table("cleaning_log.csv")
log[["step", "rows_before", "rows_after", "rows_removed", "percent_removed"]]
"""),
        md("""
### The duplicates are the dangerous one

**329,691 rows — 11.66% of the dataset.** Duplicates are compared *ignoring*
`source_file`, so identical flows appearing on different capture days are caught too.

If the same row sits in both training and test data, the model has already seen the
answer. It would score near-perfectly by memorisation. PortScan lost 43% of its rows
here — a scan fires thousands of near-identical probes, so many flows genuinely are
byte-for-byte identical.
"""),
        code("""
pairs = table("redundant_feature_pairs.csv")
print(f"{len(pairs)} feature pairs correlate above 0.95")
pairs.head(10)
"""),
        md("""
## Sampling

Training six algorithms on 2.5 million rows is not practical on four CPU cores, so
the model comparison runs on a 400,040-row stratified sample and only the final model
is refit on everything.

Classes are sampled proportionally **except** where a proportional share would fall
below 50 rows. At a 16.01% rate Heartbleed's share is 1.8 rows, which would leave the
test set with none at all.
"""),
        code("""
sampling = table("sampling_report.csv")
sampling
"""),
        md("""
## The train/test split

80/20, stratified on the 9-class label so one split serves both experiments and their
results stay comparable.
"""),
        code("""
balance = table("split_class_balance.csv", index_col=0)
balance
"""),
        md("""
### Leakage check

`datasets.py` separates two things that look identical but are not:

- **True leakage** — the same features *and* the same label on both sides. The model
  can memorise the answer. Found: **0 rows**.
- **Label conflict** — the same features with *different* labels. Not leakage; the
  model gains nothing because the copies disagree. Found: **3 rows**, benign in
  training and PortScan in test.

Those three rows put a hard ceiling on achievable accuracy. Whichever label a model
predicts, it is wrong about the other copy — so a reported 100% on this dataset would
be evidence of a bug rather than a good model.
"""),
        md("""
## Feature selection

Four methods, combined by average rank. Raw scores are not comparable across methods
— mutual information is in nats, forest importance sums to 1 — so each is converted
to a position before averaging.

The correlation filter is deliberately **not** a voter. It answers "is this feature a
duplicate?", not "is it useful?". In a first run it voted alongside the others and
four redundant pairs survived into the selected set, including two features
correlated at 1.0000. Used as a gate on the candidate pool instead, those slots went
to features carrying new information.

**Critically, all of this is fitted on the training split only.** Mutual information,
forest importance and RFE all learn from labels, so fitting them on the full dataset
would let test information influence which features are chosen.
"""),
        code("""
selected = table("selected_features.csv")
print(f"{len(selected)} features selected from 70\\n")
for i, name in enumerate(selected["feature"], 1):
    print(f"{i:2}. {name}")
"""),
        code('figure("06_feature_consensus.png")'),
        md("""
### Does dropping 45 features cost anything?

No — it helped, on both experiments, while training faster.
"""),
        code("""
comparison = table("feature_set_comparison.csv")
comparison
"""),
        code('figure("07_feature_set_comparison.png")'),
    ]
)

# ---- 04 -------------------------------------------------------------------
NOTEBOOKS["04_baseline_models.ipynb"] = notebook(
    header(
        "04", "Baseline models",
        "Six classical algorithms trained on both experiments and scored on the same "
        "held-out test set.",
        "src/train.py",
    )
    + [
        md("""
## The six algorithms

| Model | Type |
|---|---|
| Logistic Regression | linear baseline |
| KNN | distance-based |
| Decision Tree | single non-linear tree |
| Random Forest | tree ensemble (bagging) |
| SVM (RBF) | margin-based |
| XGBoost | tree ensemble (boosting) |

Scale-sensitive models (Logistic Regression, KNN, SVM) are wrapped in a `Pipeline`
with `StandardScaler`, so scaling is fitted on training folds only and never sees the
test set.

**Two models train on reduced sets**, recorded in the results table: SVM on 30,000
rows and KNN on 50,000, because their cost grows with sample size. Both still predict
on the full 80,008-row test set, so the comparison stays fair where it matters.

Reproduce with:

```bash
python src/train.py
```
"""),
        code("""
binary = table("baseline_models_binary.csv")
binary
"""),
        code("""
multi = table("baseline_models_multi.csv")
multi
"""),
        md("""
## Why SVM's AUC columns are blank

ROC-AUC and PR-AUC need a score per class that behaves like a probability. `SVC`
without `probability=True` produces one-vs-one voting margins, which do not sum to 1.
Computing an AUC from them would report a number the metric does not apply to, so the
columns are left empty and the reason documented. Enabling `probability=True` would
trigger an internal cross-validation that makes an already-slow model roughly five
times slower.

## What the results show

The tree family wins decisively on both experiments, and the reason is structural
rather than incidental. "Is this an attack?" is not a smooth boundary in feature
space — it is a set of conditions. *If packet sizes are unusually variable, and the
timing is machine-regular, and the destination port is unusual, then PortScan.*
Trees represent rules natively; Logistic Regression and SVM have to approximate them
with a surface.
"""),
    ]
)

# ---- 05 -------------------------------------------------------------------
NOTEBOOKS["05_model_comparison.ipynb"] = notebook(
    header(
        "05", "Model comparison, cross-validation and class imbalance",
        "Why accuracy is misleading here, whether the ranking survives "
        "cross-validation, and whether resampling helps.",
        "src/evaluate.py, src/cross_validation.py, src/imbalance.py",
    )
    + [
        md("""
## Accuracy hides a model that detects almost nothing

Because 82.96% of traffic is benign, every model here clears 0.92 accuracy. The gap
between accuracy and F1 is what separates them.
"""),
        code("""
allr = table("baseline_models_all.csv")
allr["accuracy_minus_f1"] = (allr["accuracy"] - allr["f1"]).round(4)
allr[["experiment", "model", "accuracy", "f1", "accuracy_minus_f1", "precision", "recall"]]
"""),
        code('figure("08_accuracy_vs_f1.png")'),
        md("""
**SVM on the multi-class task scores 0.9563 accuracy and 0.3844 F1** — a gap of 0.57.
It achieves that accuracy by predicting "BENIGN" for almost everything and being
rewarded by the class distribution. Deployed, attacks would walk straight past it.

XGBoost's gap is 0.04. High accuracy *and* high F1 — it is genuinely finding attacks.

## Cost is not symmetric
"""),
        code('figure("09_model_cost.png")'),
        md("""
KNN trains in 0.1 seconds and takes 8.1 seconds to predict. SVM takes 56.8 seconds to
predict. Trees and XGBoost predict in a fraction of a second. For a system watching
live traffic, **prediction** latency is the binding constraint — which disqualifies
KNN and SVM on architecture, not just accuracy.

## Cross-validation: does the ranking hold?

Every score above comes from a single split, which can be lucky or unlucky. Stratified
5-fold cross-validation refits each model five times.
"""),
        code("""
cv_binary = table("cross_validation_binary.csv")
cv_multi = table("cross_validation_multi.csv")
cols = ["model", "f1_mean", "f1_std", "accuracy_mean", "total_seconds"]
display(cv_binary[cols].round(5))
display(cv_multi[cols].round(5))
"""),
        md("""
**The binary ranking is stable** — a fold spread of 0.0016, and XGBoost leads while
training four times faster than Random Forest.

**The multi-class ranking is not.** It *reverses* under cross-validation: Decision
Tree now leads. Random Forest scored 0.9650 on one fold and 0.8830 on another, a
spread of 0.082 against model-to-model gaps of about 0.01. When the noise is eight
times the difference, the ranking is meaningless.
"""),
        code('figure("10_cross_validation.png")'),
        md("""
### Where the instability comes from

The obvious hypothesis — that Heartbleed's 9 rows destabilise the macro average — is
wrong. It does the opposite.
"""),
        code("""
stability = table("per_class_stability.csv", index_col=0)
stability.round(4)
"""),
        code('figure("11_per_class_stability.png")'),
        md("""
**Heartbleed scores 1.0000 on every fold from 9 training rows**, because its exploit
has an unmistakable traffic signature. **Infiltration**, with three times as many
rows, swings between 0.667 and 0.923 — it is an attacker already inside the network
behaving like an ordinary user, so there is genuinely little to separate. **Bot**,
with 250 rows, is the weakest class at 0.78.

What destabilises the score is a class being *hard to separate*, not merely small.

## Class imbalance

Four strategies, compared by cross-validation on the **training split** — choosing a
strategy by its test score would let the test set influence the model. SMOTE runs
inside an `imblearn` Pipeline so it is applied per training fold, never before the
split.
"""),
        code("""
imb_binary = table("imbalance_binary.csv")
imb_multi = table("imbalance_multi.csv")
cols = ["strategy", "f1_mean", "f1_std", "precision_mean", "recall_mean", "seconds"]
display(imb_binary[cols].round(5))
display(imb_multi[cols].round(5))
"""),
        code('figure("12_class_imbalance.png")'),
        md("""
### Two findings

**SMOTE cannot run on the multi-class task at all.** It interpolates between a rare
row and its k nearest same-class neighbours; Heartbleed has 9 training rows, which
drops to about 6 inside a fold — fewer than the algorithm needs. Forcing it with
`k_neighbors=1` would manufacture thousands of near-duplicates of two or three real
flows and teach the model those specific rows by heart.

**Configuration matters more than technique.** Undersampling appears twice on purpose.
`RandomUnderSampler`'s default reduces *every* class to the size of the smallest —
4 rows here — leaving about 36 training rows, and F1 collapses to 0.291. Capping only
the majority class instead gives **0.937**, the best multi-class result recorded and
the most stable of any strategy. Same technique, same data, one configuration line
between them.

On the binary task all five strategies sit within 0.0016 of each other, smaller than
the fold noise. The useful finding is the trade rather than the winner: baseline holds
the best precision and the worst recall, undersampling flips it exactly. **Catching
more attacks and raising more false alarms are the same dial.**
"""),
    ]
)

# ---- 06 -------------------------------------------------------------------
NOTEBOOKS["06_hyperparameter_tuning.ipynb"] = notebook(
    header(
        "06", "Hyperparameter optimisation",
        "Randomised search over Random Forest and XGBoost, and why every gain it "
        "found was inside the noise.",
        "src/tuning.py",
    )
    + [
        md("""
## Method

Randomised search rather than grid search: a full grid over these ranges is thousands
of combinations, and randomised search reaches a comparable result from a small sample
because only a few hyperparameters usually matter.

15 candidates × 3 folds per model, on a 150,000-row subsample of the training split.
Each tuned result is compared against the Phase 9 defaults **scored on the same
folds**, so the comparison isolates the effect of tuning.

```bash
python src/tuning.py    # about 64 minutes on 4 cores
```
"""),
        code("""
tune_binary = table("tuning_binary.csv")
tune_multi = table("tuning_multi.csv")
display(tune_binary.round(5))
display(tune_multi.round(5))
"""),
        code('figure("13_hyperparameter_tuning.png")'),
        md("""
## Every gain is inside the fold spread

| Experiment | Model | Gain | Combined spread | Verdict |
|---|---|---|---|---|
| A | Random Forest | +0.00053 | 0.00237 | not meaningful |
| A | XGBoost | +0.00012 | 0.00127 | not meaningful |
| B | Random Forest | +0.00835 | 0.01685 | not meaningful |
| B | XGBoost | +0.01176 | 0.03805 | not meaningful |

Each "improvement" is smaller than the range the same model produces just by being
trained on different slices of the same data.

One thing the mean hides: tuning **halved XGBoost's multi-class variance**, from
std 0.02597 to 0.01208. The average barely moved; the consistency genuinely improved.
"""),
        code("""
import itertools
import json
params = json.loads((TABLES / "best_params.json").read_text())
print(json.dumps(params, indent=2))
"""),
        md("""
## Where the gains actually came from

| Change | Multi-class F1 effect |
|---|---|
| Removing 329,691 duplicate rows | prevented an inflated score outright |
| Feature selection (70 → 25) | **+0.0165** |
| Hyperparameter tuning | +0.00012 (binary), and it *lost* on the test set |

**Data quality beat model configuration by two orders of magnitude.** That is the
opposite of where most effort usually goes, and notebook 07 confirms it on held-out
data: the tuned multi-class model scores *lower* than the defaults, so the defaults
ship.
"""),
    ]
)

# ---- 07 -------------------------------------------------------------------
NOTEBOOKS["07_final_evaluation.ipynb"] = notebook(
    header(
        "07", "Final evaluation on the held-out test set",
        "The only place the test set is scored. Every earlier choice was made on "
        "cross-validation over the training split.",
        "src/final_evaluation.py",
    )
    + [
        md("""
## The final model

XGBoost with default settings, refit on all 320,032 training rows, scored once on the
sealed 80,008-row test set.

Chosen on evidence rather than F1 alone: it leads binary cross-validation at
0.99678 ± 0.00059 and predicts 80,008 flows in 0.11 seconds against SVM's 56.8. For a
system watching live traffic, prediction latency is a requirement, not a tiebreak. On
the multi-class task the three tree models are statistically indistinguishable, so
cost and consistency decide it.
"""),
        code("""
final = table("final_evaluation.csv")
final.round(6)
"""),
        md("""
### Six decimal places, deliberately

At four, ROC-AUC reads `1.0000` — a perfect score. That is impossible here: three
flows carry identical features with conflicting labels, so no model can be right about
both. **0.999969** is the true figure and the defensible one.

### Cross-validation predicted this to four decimals

| | Binary F1 |
|---|---|
| Cross-validation estimate (training data only) | 0.99678 |
| Held-out test set (never seen) | 0.996997 |

A gap of 0.0002. That agreement is the real result — it is the evidence that the
deduplication worked, the stratified split held, feature selection never saw the test
data, and the model generalises.

### The tuned model lost

Multi-class F1 fell from 0.958281 to 0.957744 with tuned settings, and fit time rose
from 37.9s to 59.1s. Notebook 06 predicted exactly this. The defaults ship.

## Confusion matrices
"""),
        code('figure("14_confusion_binary.png")'),
        code('figure("15_confusion_multi.png")'),
        md("""
Nearly every error is a confusion with BENIGN in one direction or the other — 39
benign flows called PortScan, 14 Bot and 14 PortScan flows called benign. The model
almost never mistakes one attack type for another. **The hard problem is the benign
boundary, not telling attacks apart.**

## ROC and precision-recall curves
"""),
        code('figure("16_roc_pr_curves.png")'),
        md("""
The precision-recall curve is the more informative of the two on imbalanced data: its
baseline sits at the attack rate, not at 0.5.

## Per-class performance
"""),
        code("""
per_class = table("final_per_class_report.csv", index_col=0)
per_class.round(4)
"""),
        code('figure("17_final_per_class.png")'),
        md("""
## Honest limitations

- **Infiltration: precision 1.0000, recall 0.7143.** When the model says Infiltration
  it is always right, but it catches only 5 of 7. That is the dangerous direction of
  error — an attacker already inside the network, behaving like an ordinary user,
  walking past the detector.
- **Bot is the weakest class at 0.8136**, exactly as cross-validation predicted from
  training data alone, before the test set was ever opened.
- **Three flows carry conflicting labels**, so a perfect score is impossible.
- **This is 2017 traffic.** A model trained on it should not be assumed to detect
  attacks invented since.
"""),
    ]
)

# ---- 08 -------------------------------------------------------------------
NOTEBOOKS["08_explainability.ipynb"] = notebook(
    header(
        "08", "Feature importance and SHAP explanations",
        "What the model relies on, and why it called one specific flow an attack — "
        "including an assumption the data overturned.",
        "src/explainability.py",
    )
    + [
        md("""
## Two views of importance

They answer different questions and disagree sharply here.

| Measure | Question | Weakness |
|---|---|---|
| **Gain** | How much did this feature reduce impurity *while the model was built*? | Biased toward features with many distinct values; describes the **training** data |
| **Permutation** | How much does test F1 **drop** if this feature's values are shuffled? | Slower, but measures what the model relies on for **unseen** traffic |
"""),
        code("""
imp = table("feature_importance_binary.csv", index_col=0)
imp[["gain", "permutation", "gain_rank", "permutation_rank", "rank_shift"]].head(12).round(5)
"""),
        code('figure("18_feature_importance_binary.png")'),
        code('figure("18b_feature_importance_multi.png")'),
        md("""
### Destination Port: #10 by gain, #1 by permutation

It barely separates attack from benign — both have a median destination port of 80.
But it separates attack *types* decisively:

| Class | Median destination port | Service |
|---|---|---|
| Brute Force | 21 | FTP |
| BENIGN · DoS · DDoS · Web Attack | 80 | HTTP |
| Infiltration · Heartbleed | 444 | service port |
| PortScan | 3390 | beside RDP's 3389 |
| Bot | 8080 | HTTP alternate |

Which is why permutation importance ranks it first on the multi-class task at 0.29759,
more than double the runner-up, while gain placed it #13.

## Global SHAP
"""),
        code('figure("19_shap_global.png")'),
        md("""
### An assumption the data overturned

The intuitive story about attack traffic is that it is **uniform** — machine-generated
packets of consistent size, unlike messy human browsing. It reads well. It is wrong.

| | Packet Length Variance (median) |
|---|---|
| BENIGN | 945 |
| ATTACK | 2,101,995 |

A factor of roughly **2,200 in the opposite direction**. A flood mixes tiny control
packets with large payloads inside a single flow; ordinary browsing sits near a
consistent MTU. The *variance* is the signature, and the beeswarm above shows high
values pushing hard toward ATTACK.

This is worth stating plainly: the plausible explanation and the true one were not the
same, and only measurement separated them.

## Local SHAP — explaining one flow

A global summary says what matters on average. An analyst looking at a single alert
needs the reason for *that* flow.
"""),
        code('figure("20_shap_local.png")'),
        md("""
| Feature | Flow value | Contribution | Direction |
|---|---|---|---|
| Packet Length Variance | 2,927,710 | +6.22 | toward ATTACK |
| Init_Win_bytes_backward | 235 | +3.64 | toward ATTACK |
| Bwd Packet Length Max | 5,792 | +1.99 | toward ATTACK |
| Destination Port | 80 | +1.81 | toward ATTACK |
| Fwd Packet Length Max | 358 | −0.78 | toward BENIGN |

Starting from a baseline of −2.463, the contributions carry the output to +15.948 and
sum exactly to it.

This is the difference between an alert and an *actionable* alert: an analyst can see
that a packet-length variance of 2.9 million on port 80 is what triggered it. The same
explanation is shown for every prediction in the Streamlit demo:

```bash
streamlit run app/app.py
```
"""),
    ]
)


def main() -> None:
    for filename, content in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        n_cells = len(content["cells"])
        print(f"  wrote {filename:38} {n_cells:2} cells")
    print(f"\n{len(NOTEBOOKS)} notebooks written to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
