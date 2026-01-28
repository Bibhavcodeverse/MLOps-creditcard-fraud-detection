import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from data_loader import load_data, split_features_target, validate_data
from sklearn.model_selection import train_test_split

def optimize_threshold(model, X_test, y_test):
    """Finds the optimal threshold for F1 score."""
    print("Optimizing Threshold...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # Remove NaNs
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    # Compare with default
    y_pred_default = (y_prob >= 0.5).astype(int)
    default_f1 = f1_score(y_test, y_pred_default)
    print(f"Default (0.5) F1 Score: {default_f1:.4f}")
    
    improvement = (best_f1 - default_f1) / default_f1 * 100
    print(f"Improvement: {improvement:.2f}%")
    
    return best_threshold

def main():
    # Load data
    df = load_data('data/raw/creditcard.csv')
    validate_data(df)
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load pipeline
    try:
        pipeline = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/model.pkl')
    except FileNotFoundError:
        print("Models not found. Please run pipeline.py first.")
        return
        
    X_test_transformed = pipeline.transform(X_test)
    
    # 1. Optimize Threshold
    best_thresh = optimize_threshold(model, X_test_transformed, y_test)
    
    # 2. Evaluate with new threshold
    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    y_pred_new = (y_prob >= best_thresh).astype(int)
    
    print("\n--- Final Evaluation with Optimal Threshold ---")
    print(classification_report(y_test, y_pred_new))

if __name__ == "__main__":
    main()
