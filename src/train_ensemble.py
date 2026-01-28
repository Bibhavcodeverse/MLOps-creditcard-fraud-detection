import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import joblib
import os

# Paths
# Assuming script is run from project root 'credit-card-fraud-ml'
PROCESSED_DATA_DIR = r'data/processed'
MODELS_DIR = r'models'
METRICS_DIR = r'metrics'

def train_ensemble():
    print("Loading processed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()

    # Define Base Estimators
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        scale_pos_weight=1,
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )

    # Train Ensemble
    print("Training Voting Ensemble (RF + XGB)...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft' 
    )
    ensemble.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    # Metrics
    metrics_report = f"""--- Ensemble Evaluation ---
Precision: {precision_score(y_test, y_pred):.4f}
Recall: {recall_score(y_test, y_pred):.4f}
F1 Score: {f1_score(y_test, y_pred):.4f}
ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}

Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred)}
"""
    print(metrics_report)

    # Save Metrics
    if not os.path.exists(METRICS_DIR):
        os.makedirs(METRICS_DIR)
    
    with open(os.path.join(METRICS_DIR, 'ensemble_metrics.txt'), 'w') as f:
        f.write(metrics_report)

    # Save Model
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    joblib.dump(ensemble, os.path.join(MODELS_DIR, 'ensemble_model.pkl'))
    print("Model and metrics saved.")

if __name__ == "__main__":
    train_ensemble()
