import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.high_amount_threshold = None
        self.required_columns = ['Time', 'Amount']

    def fit(self, df):
        """Learns scaling parameters and thresholds from training data."""
        
        self.high_amount_threshold = df['Amount'].quantile(0.95)
        
        
        self.scaler.fit(df[['Amount']])
        print("Feature Engineering: Fitted scaler and learned threshold.")
        return self

    def transform(self, df):
        """Applies feature transformations to the data."""
        df = df.copy()
        
        # 1. log_amount
        df['log_amount'] = np.log1p(df['Amount'])
        
        # 2. amount_zscore
        if hasattr(self.scaler, 'mean_'): # Check if fitted
            df['amount_zscore'] = self.scaler.transform(df[['Amount']])
        else:
            raise ValueError("FeatureEngineering instance is not fitted yet.")

        # 3. is_high_amount
        if self.high_amount_threshold is not None:
             df['is_high_amount'] = (df['Amount'] > self.high_amount_threshold).astype(int)
        else:
             raise ValueError("FeatureEngineering instance is not fitted yet.")

        # 4. amount_per_time
        df['amount_per_time'] = df['Amount'] / (df['Time'] + 1)
        
        return df

    def save(self, filepath="models/feature_engineer.pkl"):
        """Saves the fitted object."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Feature Engineering artifacts saved to {filepath}")

    @staticmethod
    def load(filepath="models/feature_engineer.pkl"):
        """Loads a fitted object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature Engineering artifacts not found at {filepath}")
        return joblib.load(filepath)

if __name__ == "__main__":
    # Test script compatibility or for quick validation
    try:
        df = pd.read_csv('data/processed/train.csv')
        fe = FeatureEngineering()
        fe.fit(df)
        df_transformed = fe.transform(df)
        print("Transformation successful.")
        print(df_transformed[['log_amount', 'amount_zscore', 'is_high_amount', 'amount_per_time']].head())
        fe.save()
    except Exception as e:
        print(f"Feature Engineering Test Failed: {e}")
