import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp

def load_reference_data():
    """Loads the training data as baseline."""
    path = "data/processed/train.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Reference data not found. Run data_ingest.py first.")
    return pd.read_csv(path)

def check_drift(current_data_path, threshold=0.05):
    """
    Checks for drift between reference data and current data.
    Uses KS test for continuous variables.
    """
    print("Loading reference and current data...")
    ref_df = load_reference_data()
    
    if not os.path.exists(current_data_path):
         raise FileNotFoundError(f"Current data not found: {current_data_path}")
    curr_df = pd.read_csv(current_data_path)
    
    # Key features to monitor (can be all, but selecting key ones for speed/relevance)
    # V14, V12, V17 are often top features in Fraud Detection
    features_to_monitor = ['V14', 'V12', 'V17', 'Amount']
    
    drift_report = {}
    drift_detected = False
    
    print("\n--- Drift Monitoring Report ---")
    
    for feature in features_to_monitor:
        if feature in ref_df.columns and feature in curr_df.columns:
            # KS Test
            stat, p_value = ks_2samp(ref_df[feature], curr_df[feature])
            
            # If p_value < threshold, distributions are different -> Drift
            is_drift = p_value < threshold
            drift_report[feature] = {
                "ks_stat": stat,
                "p_value": p_value,
                "drift_detected": is_drift
            }
            
            status = "DRIFT DETECTED" if is_drift else "No Drift"
            print(f"Feature {feature}: p-value={p_value:.5f} -> {status}")
            
            if is_drift:
                drift_detected = True
        else:
            print(f"Skipping {feature} (missing in data)")
            
    return drift_detected, drift_report

if __name__ == "__main__":
    # Example: Check drift against the test set (which should match, but let's see)
    # real world: current_data_path would be a batch of new data
    try:
        drift, report = check_drift("data/processed/test.csv")
        
        if drift:
            print("\nWARNING: Data Drift Detected! Triggering retraining recommended.")
            # In a real system, this would emit an event or call retrain directly
        else:
            print("\nSystem status: Healthy (No significant drift).")
            
    except Exception as e:
        print(f"Monitoring Failed: {e}")
