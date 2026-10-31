import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from config.logging_config import get_logger

logger = get_logger("ModelEvaluator")

def evaluate_classification_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test"
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    y_pred = model.predict(X)
    
    # Probabilities for ROC-AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    
    roc_auc = float(roc_auc_score(y, y_proba)) if y_proba is not None else 0.0
    cm = confusion_matrix(y, y_pred).tolist()
    
    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)

    metrics = {
        f"{split_name}_accuracy": round(acc, 4),
        f"{split_name}_precision": round(prec, 4),
        f"{split_name}_recall": round(rec, 4),
        f"{split_name}_f1": round(f1, 4),
        f"{split_name}_roc_auc": round(roc_auc, 4),
        f"{split_name}_confusion_matrix": cm,
        "classification_report": report_dict
    }

    logger.info(
        f"[{split_name.upper()}] Acc: {acc:.4f} | Prec: {prec:.4f} | "
        f"Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}"
    )
    return metrics

def extract_feature_importance(model: Any, feature_names: List[str], top_n: int = 15) -> List[Dict[str, Any]]:
    """Extracts top feature importances or coefficients for model interpretability."""
    importances = []
    
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        for name, val in zip(feature_names, raw_imp):
            importances.append({"feature": name, "importance": round(float(val), 5)})
    elif hasattr(model, "coef_"):
        raw_imp = np.abs(model.coef_[0])
        for name, val in zip(feature_names, raw_imp):
            importances.append({"feature": name, "importance": round(float(val), 5)})
    else:
        return []

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances[:top_n]
