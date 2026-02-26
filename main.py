# Import modules and libraries
from source.ingestion.load_data import load_raw_data, validate_schema, validate_types
from source.preprocessing.feature_engineering import bin_age, bin_balance,create_target, encode_categoricals, apply_woe_encoding
from source.training.model_training import split_dataframe, select_features, train_model
from source.evaluation.model_evaluation import evaluate_model
from source.evaluation.model_selection import select_best_model
from source.artifacts.artifact_manager import save_artifact, load_artifact
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models= {
    'LogisticRegression': LogisticRegression(class_weight= 'balanced', solver= "liblinear", max_iter= 1000, random_state= 12345),
    'RandomForestClassifier': RandomForestClassifier(class_weight= "balanced", n_estimators= 200, random_state= 12345),
    
}
# Load data and validate
df= load_raw_data('data/dataset/bank.csv')
df= validate_schema(df)
df= validate_types(df)
# Target create
df= create_target(df)
# Split
train_df, test_df= split_dataframe(df, target_col= 'default_binary')
# Feature engineering
CATEGORICAL_COLUMNS= [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "age_bin",
    "balance_bin"
]
# Train
train_df= bin_age(train_df)
train_df= bin_balance(train_df)

train_df, woe_store= encode_categoricals(train_df, features= CATEGORICAL_COLUMNS)

# Test
test_df= bin_age(test_df)
test_df= bin_balance(test_df)

test_df= apply_woe_encoding(test_df, woe_store=woe_store)

# Feature selection
FEATURES= [
    "age_bin_woe",
    "balance_bin_woe",
    "job_woe",
    "marital_woe",
    "education_woe",
    "housing_woe",
    "loan_woe"
]

# Training
X_train, y_train= select_features(train_df, features= FEATURES)
X_test, y_test= select_features(test_df, features= FEATURES)
#model= train_model(X_train, y_train)

# Evaluation
model_results= {}
for name, model in models.items():
    trained_model= train_model(model, X_train, y_train)
    metrics= evaluate_model(trained_model, X= X_test, y= y_test)
    
    model_results[name]= {
        "model": trained_model,
        "metrics": metrics
    }

best_model_info= select_best_model(model_results)
print("Best model selected:")
print(best_model_info["model_name"])
print(best_model_info["metrics"])

# Build final artifact
artifact= {
    "model_name": best_model_info["model_name"],
    "model": best_model_info["model"],
    "metrics": best_model_info["metrics"],
    "features": FEATURES,
    "woe_store": woe_store,
    "version": "v1"
}
artifact_path= "artifacts/model_v1/model.joblib"

save_artifact(artifact= artifact, path= artifact_path)
loaded_artifact= load_artifact(artifact_path)

assert loaded_artifact["model_name"] == artifact["model_name"]
