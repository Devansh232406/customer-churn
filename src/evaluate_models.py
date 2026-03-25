import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os
import joblib

def evaluate_model(model, X_test, y_test, model_name, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    
    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{model_name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
        plt.close()
        
    return metrics

def evaluate_all_models(models_dict, X_test, y_test):
    print("Evaluating models...")
    all_metrics = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        
    metrics_df = pd.DataFrame(all_metrics)
    print("\n--- Model Evaluation Results ---")
    print(metrics_df.to_string(index=False))
    
    os.makedirs('outputs', exist_ok=True)
    metrics_df.to_csv('outputs/model_metrics.csv', index=False)
    print("\nMetrics saved to outputs/model_metrics.csv")
    print("Plots saved to outputs/")
    return metrics_df
