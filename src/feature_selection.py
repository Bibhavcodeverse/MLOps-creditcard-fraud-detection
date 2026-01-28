import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Paths
# Assuming script is run from project root 'credit-card-fraud-ml'
RAW_DATA_PATH = r'data/raw/creditcard.csv'
PROCESSED_DATA_DIR = r'data/processed'

def prepare_data():
    print("Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: File not found at {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Feature Engineering
    print("Engineering features...")
    df['log_amount'] = np.log1p(df['Amount'])
    df['amount_per_time'] = df['Amount'] / (df['Time'] + 1)
    
    threshold_95 = df['Amount'].quantile(0.95)
    df['is_high_amount'] = (df['Amount'] > threshold_95).astype(int)
    
    scaler = StandardScaler()
    df['amount_zscore'] = scaler.fit_transform(df[['Amount']])

    # 2. Strict Feature Selection
    # Top 20 Features based on importance analysis
    top_features = [
        'V17', 'V12', 'V14', 'V10', 'V11', 'V16', 'V7', 'V9', 'V18', 'V21', 
        'V4', 'V26', 'V3', 'V27', 'V2', 'V8', 'V1', 'V19', 'V20', 'V5',
        'log_amount', 'amount_per_time', 'amount_zscore', 'is_high_amount', 
        'Class' # Keep target
    ]
    
    print(f"Selecting features: {len(top_features) - 1} predictors")
    df_selected = df[top_features]

    X = df_selected.drop("Class", axis=1)
    y = df_selected["Class"]

    # 3. Train-Test Split (Stratified)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Apply SMOTE (Only on Training Data)
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 5. Save Data
    print("Saving processed data...")
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    X_train_resampled.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    y_train_resampled.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)

    print("Data preparation complete.")
    print(f"Train shape: {X_train_resampled.shape}")
    print(f"Test shape: {X_test.shape}")

if __name__ == "__main__":
    prepare_data()
