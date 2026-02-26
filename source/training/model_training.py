"""
Model Training Module

Responsibilities:
- Split dataset
- Train logistic regression model
- Evaluate performance
- Save trained model and artifacts

This module does NOT:
- Perform feature engineering
- Load raw data
"""

# Import libraries
from sklearn.model_selection import train_test_split

FEATURES=[
    'age_bin',
    'balance_bin',
    'job_woe',
    'marital_woe',
    'education_woe',
    'housing_woe',
    'loan_woe'
]
def select_features(df, features):
    """
    Select the correct features
    

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the categorical features and the target column 'default'.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame contain the features for the model.

    y : pandas.Series
        Series contain the target for the model.
    """
    y= df['default_binary']
    X= df[features]

    return X, y

def split_dataframe(df, target_col, test_size= 0.3, random_state= 12345):
    """
    Split raw dataframe into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset including features and target.

    target_col : str
        Name of the target column.

    test_size : float
        Proportion of test data.

    random_state : int
        Random seed.

    Returns
    -------
    train_df : pandas.DataFrame
    test_df : pandas.DataFrame
    """
    train_df, test_df= train_test_split(df, test_size= test_size, random_state=random_state, stratify=df[target_col])
    return train_df, test_df

def split_data(X, y, test_size= 0.3, random_state= 12345):
    """
    Split data
    

    Parameters
    ----------
    X : pandas.DataFrame
        Input dataframe contain the features for the model.
    
    y : pandas.Series
        Input series contain the target for the model.
    
    test_size : float
        Input float will proportion to split data.
    
    random_state : int
        Input int will be the seed for the train_test_split.

    Returns
    -------
    X_train : pandas.DataFrame
        DataFrame contain the features for the train distribution.

    X_test : pandas.DataFrame
        DataFrame contain the features for the test distribution.

    y_train : pandas.Series
        Series contain the target for the train distribution.
    
    y_test : pandas.Series
        Series contain the target for the test distribution.
    """
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= test_size, random_state= random_state, stratify= y)
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train):
    """
    Train model
    
    Parameters
    ----------
    model : Model for scikit-learn
        Input model by scikit-learn.

    X_train : pandas.DataFrame
        Input dataframe contain the features train for the model.
    
    y_train : pandas.Series
        Series contain the target for the train distribution.
    
    Returns
    -------
    model : Model trained
    """
    model.fit(X_train, y_train)
    return model

