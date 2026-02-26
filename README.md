# 📊 Credit Scoring Machine Learning Pipeline

<p align="center">
  <a href="#-project-overview"><img src="https://img.shields.io/badge/🇺🇸_English-selected-blue?style=for-the-badge" alt="English"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/🇪🇸_Español-gray?style=for-the-badge" alt="Español"></a>
</p>

## 📌 Project Overview

This project implements an **end-to-end credit scoring machine learning pipeline**, designed to predict the probability of customer default using structured banking data.

The focus is not only on model performance, but also on **reproducibility, modularity, interpretability, and production readiness**, following best practices used in real-world ML systems.

The pipeline covers the full lifecycle:

- Data ingestion and validation
- Feature engineering and encoding
- Model training and evaluation
- Model selection
- Artifact versioning
- Inference simulation

> [!NOTE]
> The dataset used is the [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) from the UCI Machine Learning Repository. The `default` column (indicating whether a client has credit in default) is used as the target variable for credit risk modeling.

---

## 🎯 Problem Statement

Financial institutions need to assess credit risk to decide whether a client is likely to default on a financial obligation.

**Goals:**

- Predict the probability of default (`credit_score`)
- Handle imbalanced data effectively
- Build an interpretable and maintainable ML pipeline
- Ensure consistent behavior between training and inference

---

## 🧠 Dataset Characteristics

| Property | Description |
|---|---|
| **Target** | `default_binary` (binary: 0/1) |
| **Class distribution** | ~98.3% non-default / ~1.7% default |
| **Total samples** | 4,521 |
| **Feature types** | Mostly categorical (job, education, housing, loan, etc.) |
| **Numerical features** | Age, balance (binned during preprocessing) |

> [!WARNING]
> Because the minority class (default) represents only **~1.7%** of the data, metrics such as accuracy are **misleading**. This strongly influenced metric selection and modeling decisions.

---

## 🔍 Exploratory Data Analysis (EDA)

An initial EDA was conducted to:

- Understand class imbalance severity
- Inspect feature distributions
- Validate binning strategies
- Identify relationships between categorical variables and default behavior

**Key findings:**

- Strong class imbalance justified using **Precision-Recall based metrics**
- Some categorical variables showed monotonic relationships with default risk
- Supported the use of **WOE (Weight of Evidence)** encoding

EDA insights guided:

- Feature binning boundaries
- Encoding strategy decisions
- Metric selection priorities

---

## 🧩 Pipeline Architecture

```
data/dataset/bank.csv
       │
       ▼
┌─────────────────┐
│  1. Ingestion    │  Schema validation, type checking, early failure
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Target       │  Create binary target from 'default' column
│     Creation     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Stratified   │  70/30 split preserving class proportions
│     Split        │
└────────┬────────┘
    ┌────┴────┐
    ▼         ▼
  Train      Test
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Bin +  │ │ Bin +  │
│ WOE    │ │ Apply  │  ← Uses stored WOE mappings (no leakage)
│ Encode │ │ WOE    │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│  4. Training     │  Logistic Regression + Random Forest
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Evaluation   │  AP, ROC AUC, Gini, Brier, F1
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Selection    │  Filter by min ROC AUC → maximize AP
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Artifacts    │  Save model + WOE + features + metadata
└─────────────────┘
```

### 1. Data Ingestion & Validation

- Schema validation against a data contract
- Type checking (numeric vs string)
- Early failure on invalid inputs

### 2. Feature Engineering

- **Custom binning**: Age → `[young, adult, middle_age, senior]`, Balance → `[very_low, low, medium, high]`
- **Weight of Evidence (WOE) encoding** with Laplace smoothing
- Stored WOE mappings for consistent inference

### 3. Model Training

Models evaluated:

| Model | Configuration | Purpose |
|---|---|---|
| Logistic Regression | `class_weight="balanced"`, `solver="liblinear"` | Interpretable baseline |
| Random Forest | `class_weight="balanced"`, `n_estimators=200` | Non-linear baseline |

Both models use `class_weight="balanced"` to handle class imbalance by internally adjusting sample weights.

---

## 📈 Model Evaluation Strategy

Given the highly imbalanced nature of the problem, evaluation focused on **threshold-independent** metrics:

| Metric | Purpose |
|---|---|
| **Average Precision (AP)** | Precision-Recall performance |
| **ROC AUC** | Ranking/discrimination ability |
| **Gini Coefficient** | Credit risk industry standard (`2 × ROC AUC − 1`) |
| **Brier Score** | Probability calibration quality |
| **Max F1 + Threshold** | Decision trade-off analysis (diagnostic only) |

> [!IMPORTANT]
> Threshold optimization (Max F1) was analyzed but **not used for model selection**, as the decision threshold should be defined by business requirements, not by maximizing a metric.

---

## 🏆 Model Selection

A custom `select_best_model` module:

1. **Filters** models by a minimum ROC AUC threshold (≥ 0.75)
2. **Ranks** valid models by the primary metric (Average Precision)
3. **Breaks ties** using a secondary metric (Brier Score, lower is better)

**Final selection: Logistic Regression** outperformed Random Forest:

- Better ranking ability (ROC AUC & Gini)
- More stable under severe class imbalance
- Higher interpretability — critical in credit scoring for regulatory compliance

---

## 📦 Artifact Management

The final model is saved as a **versioned artifact** using `joblib`. Stored artifacts include:

- Trained model object
- WOE encoding mappings
- Feature list
- Model metadata and evaluation metrics
- Version identifier

```
artifacts/
└── model_v1/
    └── model.joblib
```

This ensures **reproducibility**, **traceability**, and supports safe future upgrades (`model_v2`, `model_v3`, etc.).

---

## 🔮 Inference Simulation

A standalone inference script (`run_inference.py`) demonstrates:

- Loading the saved artifact
- Applying **identical preprocessing** (binning + WOE with stored mappings)
- Generating default probabilities (`credit_score`)

**Output:** A continuous probability score per customer, suitable for batch or API-based deployment.

```bash
python run_inference.py
```

---

## 📁 Project Structure

```
credit_scoring/
├── main.py                          # Pipeline orchestrator
├── run_inference.py                 # Inference simulation script
├── README.md
├── README.es.md
├── data/
│   └── dataset/
│       └── bank.csv                 # Raw dataset
├── artifacts/
│   └── model_v1/
│       └── model.joblib             # Serialized model artifact
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
└── source/
    ├── __init__.py
    ├── ingestion/
    │   └── load_data.py             # Data loading & validation
    ├── preprocessing/
    │   └── feature_engineering.py   # Binning, WOE encoding
    ├── training/
    │   └── model_training.py        # Split, feature select, train
    ├── evaluation/
    │   ├── model_evaluation.py      # Metrics computation
    │   └── model_selection.py       # Best model selection
    └── artifacts/
        └── artifact_manager.py      # Save/load artifacts
```

---

## ⚠️ Challenges Encountered

- **Severe class imbalance** (~1.7% default rate) limited recall-based metrics and made Precision-Recall curves noisy
- **Environment compatibility issues** between NumPy versions and compiled libraries
- **Module resolution challenges** when structuring inference scripts as separate entry points

All issues were resolved using proper environment management, explicit artifact versioning, and modular pipeline design.

---

## 📌 Key Conclusions

1. **Class imbalance significantly limits achievable F1 and AP** — this is expected, not a failure
2. **ROC AUC and Gini are more stable** and reliable indicators for this scenario
3. **Logistic Regression remains a strong baseline** for credit scoring due to interpretability
4. **Artifact versioning is critical** for real-world ML systems
5. **A correct pipeline matters more** than chasing marginal metric gains

---

## 🚀 Future Work

- [ ] Add **XGBoost** with monotonic constraints
- [ ] Implement **cross-validation** for robustness testing
- [ ] Build **API deployment** (FastAPI)
- [ ] Add **batch scoring** interface
- [ ] Implement **monitoring & drift detection**