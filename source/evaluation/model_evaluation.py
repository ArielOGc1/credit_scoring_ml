"""
Model Evaluation Module

Responsibilities:
- Evaluate model
- Check metrics
- Save results

This module does NOT:
- Perform feature engineering
- Load raw data
- Train models
"""

# Import libraries and model trained
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, brier_score_loss

def evaluate_model(model, X, y):
    """
    Evaluate Model Module
    
    Parameters
    ----------
    model: object.self
        Object self contain the trained model

    X : pandas.DataFrame
        Dataframe contain the features.
    
    y : pandas.Series
        Series contain the target.
    
    Returns
    -------
    results : dict containing evaluation metrics

    Notes
    -----
    - Metrics such as ROC AUC, Average Precision, Gini coefficient and
    Brier score are independent of the decision threshold and are used
    for model comparison and selection.
    - The maximum F1-score and its associated threshold are provided
    strictly for diagnostic purposes and are NOT used for model
    selection.
    - The decision threshold should be defined separately according to
    business or operational requirements.
    """
  
    y_prob= model.predict_proba(X)[:, 1]

    precision, recall, thresholds= precision_recall_curve(y, y_prob)

    ap= average_precision_score(y, y_prob)

    roc= roc_auc_score(y, y_prob)

    gini= 2 * roc - 1 # Gini's coefficient

    f1= 2 * (precision * recall) / (precision + recall + 1e-10)

    brier= brier_score_loss(y, y_prob)

    f1_thresholds= f1[:-1]
    idx= np.argmax(f1_thresholds)
    best_f1= f1_thresholds[idx]
    best_threshold= thresholds[idx]

    results= {
        "ap": ap,
        "roc_auc": roc,
        "gini": gini,
        "brier": brier,
        "max_f1": float(best_f1),
        "f1_threshold": float(best_threshold) # This threshold is not used for model selection.
    }
    return results
    
    