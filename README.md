# 📊 Credit Risk Scoring System with Modular ML Architecture

<p align="center">
  <a href="#-project-overview"><img src="https://img.shields.io/badge/🇺🇸_English-selected-blue?style=for-the-badge" alt="English"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/🇪🇸_Español-gray?style=for-the-badge" alt="Español"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white" alt="Matplotlib">
</p>

<p align="center">
  <img src="images/01_ss.png" alt="Credit Risk Scoring Simulator" width="800">
</p>

## 📌 Project Overview

This project implements an **end-to-end credit scoring machine learning pipeline**, designed to predict the probability of customer default using structured banking data.

The focus is not only on model performance, but also on **reproducibility, modularity, interpretability, and production readiness**, following best practices used in real-world ML systems.

The pipeline covers the full lifecycle:

- Data ingestion and validation
- Feature engineering (binning + WOE encoding)
- Model training and evaluation
- Automated model selection
- Artifact versioning
- Interactive scoring simulator (Streamlit)

> [!NOTE]
> The dataset used is the [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) from the UCI Machine Learning Repository. The `default` column (indicating whether a client has credit in default) is used as the target variable for credit risk modeling.

---

## 🎯 Problem Statement

Financial institutions need to assess credit risk to decide whether a client is likely to default on a financial obligation.

**Goals:**

- Predict the probability of default (`credit_score`)
- Handle highly imbalanced data (~1.7% positive class)
- Build an interpretable and maintainable ML pipeline
- Ensure consistent behavior between training and inference
- Provide an interactive tool for threshold analysis

---

## 🧠 Dataset Characteristics

| Property | Description |
|---|---|
| **Target** | `default_binary` (binary: 0/1) |
| **Class distribution** | ~98.3% non-default / ~1.7% default (76 out of 4,521) |
| **Total samples** | 4,521 |
| **Feature types** | Mostly categorical (job, education, housing, loan, etc.) |
| **Numerical features** | Age, balance (binned during preprocessing) |

> [!WARNING]
> Because the minority class (default) represents only **~1.7%** of the data, metrics such as accuracy are **misleading**. This strongly influenced metric selection and modeling decisions throughout the project.

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

> [!IMPORTANT]
> The train/test split is performed **before** any encoding to prevent data leakage. WOE mappings are computed exclusively on the training set and then applied to the test set using the stored mappings.

### Key Pipeline Steps

| Step | Description |
|---|---|
| **Ingestion** | Schema validation against a data contract + type checking |
| **Feature Engineering** | Custom binning (age, balance) + WOE encoding with Laplace smoothing |
| **Training** | Logistic Regression and Random Forest, both with `class_weight="balanced"` |
| **Evaluation** | Threshold-independent metrics (ROC AUC, Gini, AP, Brier) |
| **Selection** | Automated filtering → ranking → tie-breaking |
| **Artifacts** | Versioned model + preprocessing stored via `joblib` |

---

## 📊 Results & Model Performance

### Threshold-Independent Metrics (Test Set)

These metrics evaluate the model's **ranking ability and probability quality** regardless of any decision threshold:

<p align="center">
  <img src="images/06_ss.png" alt="Model Quality Metrics" width="800">
</p>

| Metric | Value | Interpretation |
|---|---|---|
| **ROC AUC** | **0.849** | The model correctly ranks a random default above a random non-default **84.9% of the time**. Values above 0.80 are considered good in credit scoring. |
| **Gini Coefficient** | **0.698** | Industry-standard metric (`2 × AUC − 1`). A Gini of 0.698 indicates **strong discriminatory power** — well above the 0.40 threshold typically expected in banking. |
| **Average Precision** | **0.075** | Low in absolute terms, but this is **expected** with only 1.7% positives. AP is heavily influenced by the base rate; a random classifier would score ~0.017. Our model achieves **~4.4× better than random**. |
| **Brier Score** | **0.160** | Measures probability calibration (lower is better, 0 is perfect). A baseline model predicting the base default rate (~1.7%) would achieve a Brier score close to 0.016. Our score of 0.160 reflects imperfect probability calibration and highlights room for improvement. |

### Score Distribution

The histogram below shows how the model distributes default probabilities across all customers:

<p align="center">
  <img src="images/03_ss.png" alt="Score Distribution" width="800">
</p>

**Key observations:**
- The model produces a **clear separation** between high-risk and low-risk customers
- The majority of customers are clustered at **low default probabilities** (< 0.1), as expected given the class distribution
- A distinct tail of **high-risk customers** appears above 0.6, indicating the model successfully identifies high-risk profiles
- The dashed line represents the adjustable **decision threshold** — all customers to the right are flagged as potential defaults

### Threshold-Dependent Metrics (Example at threshold = 0.60)

These metrics change based on the chosen decision threshold and represent the **operational trade-off**:

<p align="center">
  <img src="images/05_ss.png" alt="Decision Performance and Scored Data" width="800">
</p>

| Metric | Value | Interpretation |
|---|---|---|
| **Accuracy** | 0.774 | Misleading in imbalanced settings — a naive "predict all non-default" classifier achieves 98.3% |
| **Precision** | 0.061 | Of customers flagged as default, 6.1% actually are. Low due to extreme class imbalance |
| **Recall** | 0.868 | The model catches **86.8% of actual defaults** — critical for risk management |
| **F1 Score** | 0.114 | Harmonic mean of precision and recall. Low F1 is expected and **not a failure** — it reflects the mathematical reality of imbalanced data |

> [!IMPORTANT]
> **Why is precision so low?** With only 76 defaults in 4,521 records (1.7%), even a good model will flag many non-defaults. This is the **precision-recall trade-off** inherent to imbalanced problems. In credit scoring, **high recall** (catching most defaults) is often prioritized over high precision, as the cost of missing a default far outweighs the cost of extra reviews.

### Approval Summary

<p align="center">
  <img src="images/04_ss.png" alt="Approval Summary" width="800">
</p>

At threshold = 0.60: **3,444 approved (76.2%)** | **1,077 rejected (23.8%)**

---

## 🖥️ Interactive Scoring Simulator

The project includes a **Streamlit web application** for interactive credit risk scoring:

<p align="center">
  <img src="images/02_ss.png" alt="Dataset Preview" width="800">
</p>

**Features:**
- 📂 Upload any CSV file with the bank data schema
- 🎯 Adjust the decision threshold in real-time
- 📊 Visualize score distributions with color-coded histograms
- ✅ See approval/rejection summaries
- 🎯 Analyze threshold-dependent metrics (when ground truth is available)
- 📈 View official model quality metrics

```bash
streamlit run app.py
```

---

## 🏆 Model Selection

A custom `select_best_model` module automates model selection:

1. **Filters** models by a minimum ROC AUC threshold (≥ 0.75)
2. **Ranks** valid models by the primary metric (Average Precision)
3. **Breaks ties** using a secondary metric (Brier Score, lower is better)

**Final selection: Logistic Regression** outperformed Random Forest:

- Better ranking ability (ROC AUC & Gini)
- More stable under severe class imbalance
- Higher interpretability — critical in credit scoring for regulatory compliance

---

## 📦 Artifact Management

The final model is saved as a **versioned artifact** using `joblib`:

```
artifacts/
└── model_v1/
    └── model.joblib    # model + WOE mappings + features + metadata + metrics
```

This ensures **reproducibility**, **traceability**, and supports safe future upgrades (`model_v2`, `model_v3`, etc.).

---

## � Project Structure

```
credit_scoring/
├── main.py                          # Pipeline orchestrator
├── app.py                           # Streamlit scoring simulator
├── run_inference.py                 # CLI inference script
├── requirements.txt
├── README.md                        # Documentation (English)
├── README.es.md                     # Documentation (Español)
├── data/
│   └── dataset/
│       └── bank.csv
├── artifacts/
│   └── model_v1/
│       └── model.joblib
├── images/                          # Screenshots for documentation
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
└── source/
    ├── ingestion/
    │   └── load_data.py
    ├── preprocessing/
    │   └── feature_engineering.py
    ├── training/
    │   └── model_training.py
    ├── evaluation/
    │   ├── model_evaluation.py
    │   └── model_selection.py
    ├── inference/
    │   └── inference_pipeline.py
    └── artifacts/
        └── artifact_manager.py
```

---

## 📌 Key Conclusions

1. **ROC AUC of 0.849 and Gini of 0.698** demonstrate the model has strong discriminatory power for separating defaulters from non-defaulters
2. **Low AP and F1 are expected**, not a failure — they reflect the mathematical reality of extreme class imbalance (1.7% positives)
3. **Logistic Regression outperformed Random Forest**, confirming that simpler, interpretable models can excel in credit scoring
4. **The pipeline design matters more** than chasing marginal metric gains — proper split before encoding, artifact versioning, and inference consistency are key
5. **An interactive simulator** enables stakeholders to explore threshold trade-offs without touching code

---

## 🚀 Future Work

- [ ] Add **XGBoost** with monotonic constraints
- [ ] Implement **cross-validation** for robustness testing
- [ ] Build **API deployment** (FastAPI)
- [ ] Add **batch scoring** interface
- [ ] Implement **monitoring & drift detection**
- [ ] Improve **probability calibration** (Platt scaling / isotonic regression)

---

## ⚙️ Setup & Usage

```bash
# Create environment
conda create -n credit_scoring python=3.11 numpy=1.26 pandas scipy scikit-learn matplotlib seaborn joblib streamlit

# Activate environment
conda activate credit_scoring

# Train pipeline
python main.py

# Run Streamlit app
streamlit run app.py

# Run CLI inference
python run_inference.py
```