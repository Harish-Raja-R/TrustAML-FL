import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef

def evaluate_predictions(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Check if there's only one class present
    if len(np.unique(y_true)) == 1:
        # If only negative class, PR-AUC and ROC-AUC are not well defined
        # Return fallback metrics
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "fpr": 0.0,
            "fnr": 0.0
        }
    
    pr_auc = average_precision_score(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "fpr": float(fpr),
        "fnr": float(fnr)
    }
