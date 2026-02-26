"""
Data Ingestion Module

Responsibility:
- Load raw data from external sources (CSV, Excel, etc.)
- Validate dataset structure and required schema
- Perform basic integrity checks
- Return a raw but structurally reliable DataFrame for downstream processing

This module does NOT perform:
- Data cleaning
- Feature engineering
- Target transformation
- Any business logic
"""
# Import libraries
import pandas as pd
import os
from pandas.api.types import is_numeric_dtype, is_string_dtype

# Schema/Data Contract
REQUIRED_COLUMNS= {
    'age': int,
    'job': str,
    'marital': str,
    'education': str,
    "default": str,
    "balance": int,
    "housing": str,
    "loan": str
}
# Functions
def load_raw_data(path, sep= ';'):
    '''
    Loads raw dataset and validates schema.
    Returns: pandas Dataframe

    :param path --> str

    Raises:
        FileNotFoundError
        ValueError
    '''

    if os.path.exists(path):
        df= pd.read_csv(path, sep= sep)
    else:
        raise FileNotFoundError(f"File not found at path: {path}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def validate_schema(df):
    '''
    Validate dataset structure and required schema
    Returns: pandas Dataframe
    
    :param df --> pandas.Dataframe

    Raises:
        ValueError
    '''
    expected_columns= set(REQUIRED_COLUMNS.keys())
    df_columns= set(df.columns)
    missing_columns= expected_columns - df_columns

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df

def validate_types(df):
    '''
    Checks the correct datatype of the columns
    Returns: pandas Dataframe
    
    :param df --> pandas.Dataframe

    Raises:
        ValueError
    '''

    errors= []

    for col, expected_type in REQUIRED_COLUMNS.items():

        actual_type= df[col].dtype
        # Numeric types
        if expected_type == int:
            if not is_numeric_dtype(df[col]): # is_numeric_dtype its input must a Series
                errors.append(f"{col}: expected numeric, got {actual_type}")
        
        # Strings / categorical types
        elif expected_type == str:
            if not is_string_dtype(df[col]):
                errors.append(f"{col}: expected str, got {actual_type}")
            
    
    if errors:
        raise ValueError("Invalid column types:\n" + "\n".join(errors))
    
    return df

