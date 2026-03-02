
import pandas as pd

from source.ingestion.load_data import validate_schema, validate_types
from source.preprocessing.feature_engineering import (
    bin_age,
    bin_balance,
    apply_woe_encoding
)
from source.artifacts.artifact_manager import load_artifact


def run_inference_pipeline(df, artifact_path):
    """
    Runs the full inference pipeline using a saved model artifact.

    Steps:
    ------
    1. Validate schema and data types
    2. Apply feature engineering (binning)
    3. Apply stored WOE encodings
    4. Generate probability scores

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset for scoring.
    artifact_path : str
        Path to the saved model artifact (.joblib).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added `credit_score` column.

    Raises
    ------
    ValueError
        If validation or preprocessing fails.
    """

    # -------------------------
    # Load artifact
    # -------------------------
    artifact = load_artifact(artifact_path)

    model = artifact["model"]
    woe_store = artifact["woe_store"]
    features = artifact["features"]

    # -------------------------
    # Validation
    # -------------------------
    df = validate_schema(df)
    df = validate_types(df)

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # -------------------------
    # Feature engineering
    # -------------------------
    df = bin_age(df)
    df = bin_balance(df)

    df = apply_woe_encoding(df, woe_store)

    # -------------------------
    # Feature selection
    # -------------------------
    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing required features after preprocessing: {missing_features}"
        )

    X = df[features]

    # -------------------------
    # Scoring
    # -------------------------
    scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["credit_score"] = scores

    return df