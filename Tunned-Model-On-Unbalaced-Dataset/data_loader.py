import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import os

# Define schema for validation
schema = pa.DataFrameSchema({
    "Time": pa.Column(float, checks=pa.Check.ge(0)),
    "Amount": pa.Column(float, checks=pa.Check.ge(0)),
    "Class": pa.Column(int, checks=pa.Check.isin([0, 1])),
})

def load_data(filepath: str) -> pd.DataFrame:
    """Loads credit card data from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates data against schema."""
    try:
        validated_df = schema.validate(df)
        print("Data validation passed.")
        return validated_df
    except pa.errors.SchemaError as e:
        print(f"Data validation failed: {e}")
        raise e

def split_features_target(df: pd.DataFrame, target_col: str = 'Class'):
    """Splits dataframe into X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
