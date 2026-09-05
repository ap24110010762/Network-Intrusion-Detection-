# Explainable ML for Network Intrusion Detection

Machine-learning intrusion detection on **CIC-IDS2017**, answering two questions about
every network flow: is it benign or malicious, and if malicious, what kind of attack?

Measured on **80,008 held-out flows** the model never saw during training:

| | Binary (BENIGN vs ATTACK) | Multi-class (9 classes) |
|---|---|---|
| **F1** | **0.996997** | **0.958281** |
| Accuracy | 0.998975 | 0.998813 |
| Precision | 0.995976 | 0.977810 |
| Recall | 0.998021 | 0.944294 |
| ROC-AUC | 0.999969 | 0.999973 |
| PR-AUC | 0.999843 | 0.987474 |
| Prediction time | 0.11 s | 0.66 s |

ROC-AUC is reported at six decimals deliberately. At four it reads `1.0000`, and a
perfect score is impossible here — three flows in this dataset carry identical
features with conflicting labels.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
streamlit run app/app.py
```

The demo runs straight from the committed models — no dataset download needed. It
opens on real held-out flows and explains every verdict with SHAP contributions.

To reproduce the pipeline from scratch you need the dataset. Download
**MachineLearningCSV.zip** from
[the CIC-IDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html), extract the 8
CSVs into `data/`, then:

```bash
python src/data_loader.py        # profile the raw 2.83M rows
python src/preprocessing.py      # clean  -> data/processed/cleaned.parquet
python src/eda.py                # figures 01-05
python src/datasets.py           # sample + stratified split
python src/feature_selection.py  # 4 methods -> selected_features.csv
python src/train.py              # 6 models x 2 experiments
python src/evaluate.py           # figures 08-09
python src/cross_validation.py   # figures 10-11
python src/imbalance.py          # figure 12
python src/tuning.py             # figure 13
python src/final_evaluation.py   # score the test set, once
python src/save_model.py         # persist to models/
python src/explainability.py     # importance + SHAP
```

---

## What the pipeline does

| Stage | Module | Outcome |
|---|---|---|
| Integration | `data_loader.py` | Merges 8 CSVs into 2,830,743 rows. `float32` downcasting keeps the merge inside 686 MB rather than ~1.7 GB. |
| Cleaning | `preprocessing.py` | Repairs 2,180 corrupted labels, drops rows with infinities, removes **329,691 duplicates**, drops 8 zero-variance columns. Every decision logged to `results/tables/cleaning_log.csv`. |
| EDA | `eda.py` | Class distribution, correlation structure, feature separation. |
| Split | `datasets.py` | 400,040-row stratified sample, 80/20 split, leakage check. |
| Selection | `feature_selection.py` | Correlation gate + 3 ranking methods by consensus. **25 features beat all 70.** |
| Models | `train.py` | Logistic Regression, KNN, Decision Tree, Random Forest, SVM, XGBoost. |
| Validation | `cross_validation.py` | Stratified 5-fold, plus per-class stability. |
| Imbalance | `imbalance.py` | Baseline vs class weights vs SMOTE vs undersampling. |
| Tuning | `tuning.py` | Randomised search, 15 candidates × 3 folds. |
| Evaluation | `final_evaluation.py` | The test set, scored once. |
| Explainability | `explainability.py` | Gain vs permutation importance, global and local SHAP. |
| Deployment | `save_model.py`, `app/app.py` | Persisted bundles + Streamlit demo. |

---

## Findings worth more than the headline score

**Data quality beat hyperparameter tuning by two orders of magnitude.** Feature
selection was worth **+0.0165** multi-class F1. Sixty-four minutes of randomised
search was worth **+0.00012** on the binary task — and the tuned model then scored
*lower* on the test set, so the default settings ship.

**Removing duplicates mattered more than any model choice.** 329,691 rows (11.66%)
were exact duplicates. Left in, the same flow would appear in both training and test
data and every score would be inflated by memorisation.

**Imbalance handling was not needed, and mostly hurt.** Class weights *lowered*
multi-class F1 from 0.933 to 0.918. SMOTE could not run on the multi-class task at
all — Heartbleed has 9 training rows, fewer than the neighbours SMOTE interpolates
between. The one strategy that helped throws data away rather than inventing it.

**Configuration matters more than technique.** Undersampling scored **0.291 and
0.937** in the same experiment. `RandomUnderSampler`'s default cuts every class to
the smallest — 4 rows here — leaving ~36 training rows. Capping only the majority
class instead gave the best multi-class result recorded.

**Separability beats sample size.** Heartbleed, with 9 training rows, scores F1
**1.0000** on every cross-validation fold and on the test set; its traffic signature
is unmistakable. Bot, with 250 rows, scores **0.8136**. It is not rarity that hurts,
it is ambiguity.

**An assumption the data overturned.** Attack traffic does not have *uniform* packet
sizes, which is the intuitive story. It has extreme variance: `Packet Length
Variance` has a median of **945** for benign flows and **2,101,995** for attacks. A
flood mixes tiny control packets with large payloads inside one flow; ordinary
browsing sits near a consistent MTU.

**The number that validates the rest.** Cross-validation on training data estimated
binary F1 at **0.99678**. The test set, opened once at the very end, returned
**0.996997** — a gap of 0.0002. That agreement is the evidence the split was clean,
the duplicates were gone, and nothing leaked.

---

## Method notes

- **The test set was used exactly once.** Features, model, resampling strategy and
  hyperparameters were all chosen by cross-validation on the training split.
- **Phase 8 ran before Phase 7 deliberately.** Feature selection learns from labels,
  so the split has to come first or test information influences which features are
  chosen.
- **SVM and KNN trained on reduced sets** (30,000 and 50,000 rows) because their cost
  grows with sample size. Both still predict on the full test set, and the reduction
  is a column in the results table.
- **SVM's AUC columns are blank on purpose.** `SVC` without `probability=True`
  produces one-vs-one voting margins, not probabilities.

## Known limitations

- **Infiltration recall is 0.714** — 2 of 7 missed. Those are attackers already inside
  the network, behaving like ordinary users. The dangerous direction of error.
- **Bot F1 is 0.8136**, the weakest class.
- **Three flows carry identical features with conflicting labels**, so no model can be
  perfectly correct on this data. A reported 100% would indicate a bug.
- **This is 2017 traffic.** A model trained on it should not be assumed to detect
  attacks invented since.

---

## Layout

```
src/          pipeline modules, each runnable on its own
app/app.py    Streamlit demo
models/       trained bundles (model + feature order + class names + test scores)
results/
  figures/    21 figures
  tables/     the CSV behind every figure
data/         dataset goes here; not committed (see data/README.md)
```

Models are saved as a bundle rather than a bare estimator: a pickled estimator does
not record which columns it expects or in what order, and a mismatch there produces
silently wrong predictions instead of an error.

## Dataset

Canadian Institute for Cybersecurity, University of New Brunswick —
[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). The authors request
citation of their 2018 publication when the dataset is used.
