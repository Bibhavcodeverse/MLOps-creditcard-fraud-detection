import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
# Assuming running from 'credit-card-fraud-ml'
PROCESSED_DATA_DIR = r'data/processed'
MODELS_DIR = r'models'

def load_data():
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()
    return X_test, y_test

def test_model(model_name):
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}")
        return

    print(f"\nLoading {model_name}...")
    model = joblib.load(model_path)
    
    X_test, y_test = load_data()
    
    print(f"Running predictions with {model_name}...")
    y_pred = model.predict(X_test)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Inference Demo
    print(f"\n--- Single Prediction Demo ({model_name}) ---")
    sample_row = X_test.iloc[0].values.reshape(1, -1)
    prediction = model.predict(sample_row)[0]
    true_label = y_test[0]
    print(f"Sample Features (First Row): {sample_row}")
    print(f"Prediction: {prediction} ({'Fraud' if prediction==1 else 'Legit'})")
    print(f"True Label: {true_label}")

if __name__ == "__main__":
    # Test all available models
    models = ['random_forest', 'xgboost', 'ensemble_model']
    for model in models:
        test_model(model)
