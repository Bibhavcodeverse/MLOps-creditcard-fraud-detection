import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transforms specified columns."""
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns:
            for col in self.columns:
                # Add small constant to avoid log(0)
                X_copy[col] = np.log1p(X_copy[col]) 
        return X_copy

def build_preprocessing_pipeline():
    """Builds the feature engineering pipeline."""
    
    # RobustScaler is good for outliers which are common in fraud data
    robust_scaler = RobustScaler()
    
    # We want to log transform 'Amount' as it is usually right-skewed
    log_transformer = LogTransformer(columns=['Amount'])
    
    # We can perform LogTransform then Scale
    # But Scaler should apply to all numerical features
    # V1-V28 are already PCA transformed, so maybe just scaling is enough for them?
    # Actually, V features are already scaled around 0 usually.
    # But RobustScaler won't hurt.
    
    pipeline = Pipeline([
        ('log_transform', log_transformer),
        ('scaler', robust_scaler)
    ])
    
    return pipeline
