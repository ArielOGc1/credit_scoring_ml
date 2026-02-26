from source.artifacts.artifact_manager import load_artifact
from source.preprocessing.feature_engineering import bin_age, bin_balance, apply_woe_encoding
from source.ingestion.load_data import load_raw_data

def run_inference():
    # Load artifact
    artifact = load_artifact("artifacts/model_v1/model.joblib")

    model = artifact["model"]
    woe_store = artifact["woe_store"]
    features = artifact["features"]

    # Load new data (simulation)
    sample_df = load_raw_data("data/raw/bank.csv").sample(5, random_state=42)

    # Apply preprocessing
    sample_df = bin_age(sample_df)
    sample_df = bin_balance(sample_df)
    sample_df = apply_woe_encoding(sample_df, woe_store)

    # Select features
    X_sample = sample_df[features]

    # Inference
    scores = model.predict_proba(X_sample)[:, 1]
    sample_df["credit_score"] = scores

    print("Inference results:")
    print(sample_df[["credit_score"]])


if __name__ == "__main__":
    run_inference()