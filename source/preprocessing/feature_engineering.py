'''
Feature engineering Module

Responsibility:
- Binning age and balance
- Encoding categorical variables
- Return a clean reliable DataFrame for downstream training

This module does NOT perform:
- Model training
- Target transformation
- Model evaluating

'''
import pandas as pd
import numpy as np
import math

def bin_age(df):
    """
    This function will bin 'age'
    
    :param df: pd.Dataframe
    :return --> pd.Dataframe ('age' binned)
    """
    bins= [20, 30, 45, 60, 100]
    labels= ['young', 'adult', 'middle_age', 'senior']
    df= df.copy()

    df['age_bin']= pd.cut(
        df['age'],
        bins= bins,
        labels= labels,
        right= False
    )

    return df

def bin_balance(df):
    """
    This function will bin 'balance'
    
    :param df --> pd.Dataframe

    :return --> pd.Dataframe ('balance' binned)
    """
    bins= [-math.inf, 69, 444, 1480, math.inf]
    labels= ['very_low_balance', 'low_balance', 'medium_balance', 'high_balance']
    df= df.copy()

    df['balance_bin']= pd.cut(
        df['balance'],
        bins= bins,
        labels= labels,
        right= False
    )
    
    return df

def create_target(df, target_col= 'default'):
    """
    Create binary target for credit risk modeling.
    """
    df= df.copy()
    df['default_binary']= df[target_col].map({'yes': 1, 'no': 0})
    return df

CATEGORICAL_COLUMNS=['job', 'marital', 'education', 'housing', 'loan', 'age_bin', 'balance_bin']
def encode_categoricals(df, features, target_col= 'default_binary'):
    """
    This function assumes presence of both classes.
    Encodes categorical features using Weight of Evidence (WoE).

    For each categorical column:
    - Computes the distribution of default and non-default events.
    - Applies Laplace smoothing to avoid division by zero.
    - Calculates WoE values per category.
    - Creates a mapping dictionary for production usage.
    - Encodes the column into a numeric WoE feature.
    - Unseen categories are assigned the mean WoE of the feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the categorical features and the target column 'default'.

    features : list[str]
        List of categorical column names to encode.

    target_col: str: default_binary
        Target column
    Returns
    -------
    df_encoded : pandas.DataFrame
        DataFrame with new WoE-encoded columns appended.

    woe_store : dict
        Dictionary containing, for each feature:
        {
            "mapping": {category: woe_value},
            "default_woe": float
        }
        Used to apply the same encoding during inference/production.
    """
    df= df.copy()
    if target_col not in df.columns:
        raise ValueError(f'{target_col} not exist')
    # Global event counts (used for WoE denominator)
    total_defaults_global= df[target_col].sum()
    total_non_defaults_global= len(df) - total_defaults_global

    woe_store= {}
    for col in features:

        # Group statistics per category
        feature_grouped= (df.groupby(col, observed= False).agg(default= (target_col, "sum"), total= (target_col, "count")))
        non_default= (feature_grouped['total'] - feature_grouped['default'])

        # Laplace Smoothing to avoid zero divisions
        dist_bad= (feature_grouped['default'] + 0.5) / (total_defaults_global + 0.5) # Smoothing: 0.5
        dist_good= (non_default + 0.5) / (total_non_defaults_global + 0.5) # Smoothing: 0.5

        # Store mapping and default WoE for production inference
        woe= np.log(dist_good/dist_bad)
        mapping= woe.to_dict()
        default_woe= woe.mean()
        woe_store[col]= {
            "mapping": mapping,
            "default_woe": default_woe
        }
        df[f'{col}_woe']= df[col].map(mapping).astype(float).fillna(default_woe)

    return df, woe_store 
    
def apply_woe_encoding(df, woe_store):
    """
    Applies pre-trained Weight of Evidence (WoE) encoding to categorical features.

    This function should be used during validation or production inference.
    It uses the mappings learned during training and handles unseen categories
    by assigning a default WoE value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing the original categorical columns.

    woe_store : dict
        Dictionary containing WoE mappings and default values for each feature.
        Format:
        {
            "feature_name": {
                "mapping": {category: woe_value, ...},
                "default_woe": float
            },
            ...
        }

    Returns
    -------
    pandas.DataFrame
        Dataset with new columns '<feature>_woe' added.
    """

    df= df.copy()

    for col, storage in woe_store.items():
        mapping= storage['mapping']
        default_value= storage['default_woe']

        df[f'{col}_woe']= df[col].map(mapping).astype(float).fillna(default_value)
    return df

def preprocessing_pipeline(df, categorical_columns, is_training= True, woe_store= None):
    """
    Complete feature engineering pipeline.

    Steps:
    - Bin age
    - Bin balance
    - Encode categorical variables using WoE

    Parameters
    ----------
    df : pandas.DataFrame
        Raw validated dataset.

    categorical_features : list
        List of categorical columns to encode.

    is_training : bool, default=True
        If True, computes WoE mappings (training mode).
        If False, applies existing WoE mappings (inference mode).

    woe_store : dict, optional
        Required when training=False.
        Pre-trained WoE mappings.

    Returns
    -------
    pandas.DataFrame
        Processed dataset ready for modeling.

    dict (optional)
        WoE mappings (only returned in training mode).
    """

    df= df.copy()
    df= bin_age(df)
    df= bin_balance(df)

    if is_training:
        df, woe_store= encode_categoricals(df, categorical_columns)
        return df, woe_store
    else:
        if woe_store is None:
            raise ValueError("woe_store must be provided when is_training= False")

        df= apply_woe_encoding(df, woe_store)

    return df