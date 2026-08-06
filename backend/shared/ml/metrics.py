import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, List

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    
    # Calculate precision, recall, f1
    # For multi-class, default to macro average
    unique_labels = np.unique(y_true)
    is_multiclass = len(unique_labels) > 2
    
    average_type = "macro" if is_multiclass else "binary"
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average_type, zero_division=0)
    
    # Calculate ROC-AUC
    roc_auc = None
    if y_prob is not None:
        try:
            if is_multiclass:
                # y_prob should be of shape (n_samples, n_classes)
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            else:
                # y_prob can be probability of positive class
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5  # fallback if roc_auc calculation fails (e.g. only one class present in split)
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }
