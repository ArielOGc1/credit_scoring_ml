"""
Model Selection Module

This module contains utilities for selecting the best trained model
based on evaluation metrics. The selection logic is designed for
highly imbalanced binary classification problems

Responsibilities:
- Apply a minimum quality filter using ROC AUC.
- Among the models that pass the filter, optimize a primary metric (e.g., Average Precision).
- Optionally use a secondary metric (e.g., Brier score) to break ties.

This module does not train or evaluate models; it only compares
already evaluated models and selects the best candidate for
production use.
"""

def select_best_model(model_results, min_roc_auc=0.75, primary_metric="ap", secondary_metric="brier"):
    """
    Select the best model based on predefined evaluation rules.

    Parameters
    ----------
    model_results : dict
        Dictionary with trained models and their metrics.
    min_roc_auc : float, optional
        Minimum ROC AUC required to consider a model valid.
    primary_metric : str, optional
        Metric to maximize (default: average precision).
    secondary_metric : str, optional
        Metric to minimize for tie-breaking (default: brier score).

    Returns
    -------
    dict
        Dictionary containing the selected model, its name, and metrics.

    Raises
    ------
    ValueError
        If no model satisfies the minimum quality criteria.
    """
    # Filter valid models
    valid_models= []
    for model_name, content in model_results.items():
        model_metrics= content["metrics"]

        if model_metrics["roc_auc"] >= min_roc_auc:
            valid_models.append({
                "model_name": model_name,
                "model": content["model"],
                "metrics": model_metrics
            })

    if not valid_models:
        raise ValueError("No model satisfies the minimum ROC AUC threshold.")
    
    # Sort by primary metric
    valid_models= sorted(valid_models, key= lambda x:x["metrics"][primary_metric], reverse= True)

    best_primary_value= valid_models[0]["metrics"][primary_metric]
    top_models= [m for m in valid_models if m["metrics"][primary_metric] == best_primary_value]

    if len(top_models) > 1:
        top_models= sorted(top_models, key= lambda x:x["metrics"][secondary_metric])
    
    # Winner
    best_model= top_models[0]
    return best_model