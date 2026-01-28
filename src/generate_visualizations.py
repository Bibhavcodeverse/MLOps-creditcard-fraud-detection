import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

# Paths
PROCESSED_DATA_DIR = r'data/processed'
MODELS_DIR = r'models'
PLOTS_DIR = r'plots'

def load_data():
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()
    return X_test, y_test

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, f'cm_{model_name}.png')
    plt.savefig(save_path)
    print(f"Saved Confusion Matrix to {save_path}")
    plt.close()

def plot_pr_curve(y_true, y_prob, model_name, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        save_path = os.path.join(PLOTS_DIR, f'pr_{model_name}.png')
        plt.savefig(save_path)
        print(f"Saved PR Curve to {save_path}")
        plt.close()
    else:
        ax.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')

def generate_visualizations():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    X_test, y_test = load_data()
    models = ['random_forest', 'xgboost', 'ensemble_model']
    
    # Combined PR Curve Plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    for model_name in models:
        model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found.")
            continue
            
        print(f"Processing {model_name}...")
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Individual CM
        plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Add to combined PR Curve
        plot_pr_curve(y_test, y_prob, model_name, ax=ax)
    
    # Finalize Combined PR Curve
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend(loc="lower left")
    ax.grid(True)
    
    combined_pr_path = os.path.join(PLOTS_DIR, 'pr_curve_comparison.png')
    plt.savefig(combined_pr_path)
    print(f"Saved Combined PR Curve to {combined_pr_path}")
    plt.close()

if __name__ == "__main__":
    generate_visualizations()
