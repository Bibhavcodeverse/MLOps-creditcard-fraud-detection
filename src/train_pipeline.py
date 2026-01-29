import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from feature_engineering import FeatureEngineering

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def load_data():
    print("Loading processed data...")
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed data not found. Run data_ingest.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def train():
    #  Load Data
    train_df, test_df = load_data()
    
    # Feature Engineering
    print("Applying Feature Engineering...")
    fe = FeatureEngineering()
    fe.fit(train_df)
    fe.save("models/feature_engineer.pkl")
    
    train_df = fe.transform(train_df)
    test_df = fe.transform(test_df)
    
    # Prepare X and y
    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]
    
    #  SMOTE Balancing
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    #  Model Tuning & Training
    print("Starting Model Training...")
    
    # Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42)
    print("Tuning Random Forest...")
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=2, scoring='f1', cv=2, verbose=1, n_jobs=-1, random_state=42)
    rf_search.fit(X_train_resampled, y_train_resampled)
    best_rf = rf_search.best_estimator_
    
    # XGBoost
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 10]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    print("Tuning XGBoost...")
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=2, scoring='f1', cv=2, verbose=1, n_jobs=-1, random_state=42)
    xgb_search.fit(X_train_resampled, y_train_resampled)
    best_xgb = xgb_search.best_estimator_
    
    # Voting Ensemble
    print("Training Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[('rf', best_rf), ('xgb', best_xgb)],
        voting='soft'
    )
    voting_clf.fit(X_train_resampled, y_train_resampled)
    
    #  Evaluation
    print("Evaluating Model...")
    y_pred = voting_clf.predict(X_test)
    y_prob = voting_clf.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("ROC-AUC:", auc)
    
    #  Save Artifacts
    joblib.dump(voting_clf, "models/model.pkl")
    
    metrics = {
        "classification_report": report,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "best_params": {
            "rf": rf_search.best_params_,
            "xgb": xgb_search.best_params_
        }
    }
    
    with open("metrics/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Training completed. Model saved to models/model.pkl and metrics to metrics/evaluation.json")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
