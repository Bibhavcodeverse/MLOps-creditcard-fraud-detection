import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_model(X_train, y_train, random_state=42):
    """Trains a Random Forest model."""
    print("Training Random Forest Model...")
    # Justification: Random Forest is robust to outliers (which we have), 
    # handles non-linear relationships, and class_weight='balanced' helps with imbalance.
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_state, 
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def cross_validate_model(model, X, y, cv=5):
    """Performs Stratified K-Fold Cross Validation."""
    print(f"Running {cv}-Fold Stratified Cross Validation...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean F1 Score: {scores.mean():.4f}")
    return scores

def save_model(model, output_dir='models'):
    """Saves the trained model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, 'model.pkl')
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
