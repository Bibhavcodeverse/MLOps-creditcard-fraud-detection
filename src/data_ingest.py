import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataIngestion:
    def __init__(self, file_path, output_dir="data/processed"):
        self.file_path = file_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Loads data from CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        print(f"Loading data from {self.file_path}...")
        return pd.read_csv(self.file_path)

    def validate_data(self, df):
        """Validates that required columns exist and data types are correct."""
        required_columns = ['Time', 'Amount', 'Class']
        # V1-V28 are also required but we can check them dynamically or assume they exist if 'Class' matches context
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check basic stats
        print(f"Data validated. Shape: {df.shape}")
        return True

    def split_and_save(self, df):
        """Splits data into train/test and saves to processed folder."""
        X = df.drop("Class", axis=1)
        y = df["Class"]

        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Saving
        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        
        # Recombine for saving (easier for next steps to just load one file if needed, or we can save X/y separately)
        # Standard practice: save train and test sets separately
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Train data saved to {train_path} ({train_df.shape})")
        print(f"Test data saved to {test_path} ({test_df.shape})")
        
        return train_path, test_path

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion(file_path="data/raw/creditcard.csv")
    try:
        df = ingestion.load_data()
        ingestion.validate_data(df)
        ingestion.split_and_save(df)
    except Exception as e:
        print(f"Data Ingestion Failed: {e}")
