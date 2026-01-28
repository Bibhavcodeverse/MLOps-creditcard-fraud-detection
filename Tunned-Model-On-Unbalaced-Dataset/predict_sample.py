import pandas as pd
import joblib
import numpy as np
from data_loader import load_data, split_features_target

def predict_sample():
    """Loads model and predicts on a random sample from the dataset."""
    print("Loading artifacts...")
    try:
        pipeline = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/model.pkl')
    except FileNotFoundError:
        print("Error: Model artifacts not found. Please run 'src/pipeline.py' first.")
        return

    # Load a small sample of data to simulate new inputs
    # In a real API, this would come from a request
    print("Loading sample data...")
    df = load_data('data/raw/creditcard.csv')
    
    # Get a positive and negative sample for demonstration
    fraud_sample = df[df['Class'] == 1].sample(1)
    normal_sample = df[df['Class'] == 0].sample(1)
    
    samples = pd.concat([fraud_sample, normal_sample])
    X_sample, y_true = split_features_target(samples)
    
    # Preprocess
    print("\nPreprocessing samples...")
    X_transformed = pipeline.transform(X_sample)
    
    # Predict
    print("Predicting...")
    # Calculate probabilities
    probs = model.predict_proba(X_transformed)[:, 1]
    
    # Apply threshold (optional, using 0.5 default here or the one found in tuning)
    # Let's say we use a conservative 0.5 for now, or the optimized one if we had saved it.
    predictions = (probs >= 0.5).astype(int)
    
    print("\n--- Prediction Results ---")
    for i, (true, pred, prob) in enumerate(zip(y_true, predictions, probs)):
        type_str = "FRAUD" if true == 1 else "NORMAL"
        print(f"Sample {i+1} (Actual: {type_str}):")
        print(f"  Probability of Fraud: {prob:.4f}")
        print(f"  Predicted Class: {pred} ({'FRAUD' if pred == 1 else 'NORMAL'})")
        print("-" * 30)

if __name__ == "__main__":
    predict_sample()
