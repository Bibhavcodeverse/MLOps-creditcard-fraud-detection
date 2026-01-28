import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    average_precision_score, 
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

def evaluate_model(model, X_test, y_test, output_dir='metrics'):
    """Evaluates the model and saves metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'auprc': average_precision_score(y_test, y_prob),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics
    with open(os.path.join(output_dir, 'scores.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

    # Save detailed evaluation report to text file
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Evaluation Report\n")
        f.write("=================\n\n")
        
        f.write("Scalar Metrics:\n")
        f.write("---------------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        f.write("----------------------\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-----------------\n")
        f.write(str(cm))
        f.write("\n\n")
        
        f.write("Precision-Recall Curve Data:\n")
        f.write("----------------------------\n")
        f.write("Recall,Precision,Threshold\n")
        # thresholds is shorter than precision/recall by 1
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
            f.write(f"{r:.4f},{p:.4f},{t:.4f}\n")
        # Add the last point (probability 1, usually)
        f.write(f"{recall[-1]:.4f},{precision[-1]:.4f},N/A\n")
            
    print(f"Detailed evaluation report saved to {report_path}")
    
    return metrics
