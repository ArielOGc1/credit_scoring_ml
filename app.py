import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from source.artifacts.artifact_manager import load_artifact
from source.inference.inference_pipeline import run_inference_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8ecf8 50%, #f5f3ff 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef2ff 0%, #e0e7ff 100%);
        border-right: 1px solid #c7d2fe;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #4f46e5 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1e1b4b !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ── Headers ── */
    h1 {
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #312e81 !important;
        font-weight: 600 !important;
    }
    h4 {
        color: #3730a3 !important;
    }

    /* ── General text ── */
    p, span, li, label, .stMarkdown {
        color: #1e1b4b;
    }

    /* ── Dataframe container ── */
    .stDataFrame {
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── File uploader ── */
    section[data-testid="stFileUploader"] {
        border: 2px dashed #a5b4fc;
        border-radius: 12px;
        padding: 8px;
    }

    /* ── Divider ── */
    hr {
        border-color: #c7d2fe !important;
    }

    /* ── Alert boxes ── */
    .stAlert {
        border-radius: 10px !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Color palette for charts
# ─────────────────────────────────────────────
CHART_COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#a78bfa",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "bg": "#fafaff",
    "text": "#1e1b4b",
    "grid": "#e0e7ff",
    "spine": "#c7d2fe",
}

def style_chart(fig, ax, title=""):
    """Apply consistent light theme styling to matplotlib charts."""
    fig.patch.set_facecolor(CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])
    ax.set_title(title, color=CHART_COLORS["text"], fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors=CHART_COLORS["text"], labelsize=10)
    ax.xaxis.label.set_color(CHART_COLORS["text"])
    ax.yaxis.label.set_color(CHART_COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(CHART_COLORS["spine"])
    ax.grid(axis="y", color=CHART_COLORS["grid"], linestyle="--", linewidth=0.5)
    fig.tight_layout()


# ─────────────────────────────────────────────
# Load artifact
# ─────────────────────────────────────────────
ARTIFACT_PATH = "artifacts/model_v1/model.joblib"

@st.cache_resource
def get_artifact():
    return load_artifact(ARTIFACT_PATH)

artifact = get_artifact()
official_metrics = artifact["metrics"]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload CSV file",
        type=["csv"],
        help="Upload a semicolon-separated CSV file with the same schema as the training data."
    )

    st.markdown("")
    threshold = st.slider(
        "🎯 Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability threshold for classifying a customer as default risk."
    )
    st.caption("⬇ Lower → more risk detection (↑ recall, ↓ precision)")
    st.caption("⬆ Higher → fewer false alarms (↓ recall, ↑ precision)")

    st.markdown("---")
    run_button = st.button("🚀 Run Scoring", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Model:** `{artifact.get('model_name', 'N/A')}`")
    st.markdown(f"**Version:** `{artifact.get('version', 'N/A')}`")
    st.markdown(f"**Features:** `{len(artifact.get('features', []))}`")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 🏦 Credit Risk Scoring Simulator")
st.markdown(
    "<p style='color: #4338ca; font-size: 1.1rem; margin-top: -10px;'>"
    "Simulate credit approval decisions using a trained ML model. "
    "Upload a dataset, adjust the threshold, and explore the results."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
if uploaded_file is None:
    # ── Empty state ──
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 80px 20px;
        ">
            <p style="font-size: 4rem; margin-bottom: 10px;">📂</p>
            <p style="font-size: 1.3rem; color: #4338ca; font-weight: 600;">
                Upload a CSV file to get started
            </p>
            <p style="font-size: 0.95rem; color: #6b7280; max-width: 500px; margin: 10px auto;">
                Use the sidebar to upload a semicolon-separated CSV file with the same schema
                as the training dataset (bank.csv).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    df = pd.read_csv(uploaded_file, sep=";")

    st.markdown("### 🗂️ Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"📋 {df.shape[0]:,} rows × {df.shape[1]} columns")

    if run_button:
        try:
            with st.spinner("🔄 Running inference pipeline..."):
                results = run_inference_pipeline(df, ARTIFACT_PATH)
                results["prediction"] = (results["credit_score"] >= threshold).astype(int)

            st.success("✅ Scoring completed successfully!")
            st.markdown("---")

            # ─── Score Distribution ───
            st.markdown("### 📊 Score Distribution")

            fig, ax = plt.subplots(figsize=(10, 4))
            n, bins_arr, patches = ax.hist(
                results["credit_score"],
                bins=40,
                edgecolor="none",
                alpha=0.85
            )
            # Color bars by threshold
            for patch, b in zip(patches, bins_arr):
                if b < threshold:
                    patch.set_facecolor(CHART_COLORS["success"])
                else:
                    patch.set_facecolor(CHART_COLORS["danger"])

            ax.axvline(
                threshold, color=CHART_COLORS["warning"],
                linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}"
            )
            ax.legend(
                fontsize=11, facecolor=CHART_COLORS["bg"],
                edgecolor=CHART_COLORS["spine"], labelcolor=CHART_COLORS["text"]
            )
            ax.set_xlabel("Credit Score (Default Probability)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            style_chart(fig, ax, "Distribution of Predicted Default Probabilities")
            st.pyplot(fig)

            st.markdown("---")

            # ─── Approval Summary ───
            st.markdown("### ✅ Approval Summary")

            approval_counts = results["prediction"].value_counts()
            total = len(results)
            approved = int(approval_counts.get(0, 0))
            rejected = int(approval_counts.get(1, 0))

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Scored", f"{total:,}")
            col2.metric("✅ Approved", f"{approved:,}", delta=f"{approved/total*100:.1f}%")
            col3.metric("❌ Rejected", f"{rejected:,}", delta=f"{rejected/total*100:.1f}%", delta_color="inverse")

            # ─── Approval pie chart ───
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                [approved, rejected],
                labels=["Approved", "Rejected"],
                autopct="%1.1f%%",
                colors=[CHART_COLORS["success"], CHART_COLORS["danger"]],
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"color": CHART_COLORS["text"], "fontsize": 12}
            )
            for autotext in autotexts:
                autotext.set_fontweight("bold")
                autotext.set_fontsize(13)
            fig.patch.set_facecolor(CHART_COLORS["bg"])
            ax.set_title(
                "Approval Breakdown",
                color=CHART_COLORS["text"], fontsize=14, fontweight="bold", pad=15
            )
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("---")

            # ─── Scored Data Preview ───
            st.markdown("### 📋 Scored Data")

            display_cols = [c for c in results.columns if c not in ["prediction"]]
            display_cols.append("prediction")
            st.dataframe(
                results[display_cols].head(20),
                use_container_width=True
            )

            # ─── Performance section (only if ground truth exists) ───
            if "default" in results.columns:
                st.markdown("---")
                st.markdown("### 🎯 Decision Performance (Threshold-Dependent)")
                st.caption(f"Evaluated at threshold = **{threshold:.2f}**")

                y_true = results["default"].map({"no": 0, "yes": 1})
                y_pred = results["prediction"]

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Precision", f"{prec:.3f}")
                c3.metric("Recall", f"{rec:.3f}")
                c4.metric("F1 Score", f"{f1:.3f}")

                st.markdown("")

                # ─── Confusion Matrix ───
                st.markdown("#### 🔢 Confusion Matrix")

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d",
                    cmap="PuBuGn",
                    ax=ax,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 16, "fontweight": "bold"},
                    linewidths=2,
                    linecolor=CHART_COLORS["bg"],
                    square=True
                )
                ax.set_xlabel("Predicted", fontsize=12)
                ax.set_ylabel("Actual", fontsize=12)
                ax.set_xticklabels(["Non-Default", "Default"], fontsize=10)
                ax.set_yticklabels(["Non-Default", "Default"], fontsize=10, rotation=0)
                style_chart(fig, ax, "Confusion Matrix")
                ax.grid(False)
                st.pyplot(fig)

                st.info(
                    "💡 **Tip:** Adjust the threshold in the sidebar to explore the "
                    "precision-recall trade-off. Lower thresholds catch more defaults "
                    "but increase false positives."
                )

            # ─── Official Model Metrics (at the end) ───
            st.markdown("---")
            st.markdown("### 📈 Model Quality Metrics (Test Set)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROC AUC", f"{official_metrics['roc_auc']:.3f}")
            m2.metric("Gini", f"{official_metrics['gini']:.3f}")
            m3.metric("Avg Precision", f"{official_metrics['ap']:.3f}")
            m4.metric("Brier Score", f"{official_metrics['brier']:.3f}")

        except Exception as e:
            st.error(f"❌ **Pipeline failed:** {str(e)}")
            st.exception(e)
