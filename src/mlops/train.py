import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

from config.config import (
    ANALYTICAL_DATA_DIR,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_SEED
)
from config.logging_config import get_logger
from src.mlops.feature_store import AcademicFeatureStore
from src.mlops.evaluate import evaluate_classification_model, extract_feature_importance

logger = get_logger("ModelTraining")

def train_all_models(mart_file: Path = None) -> Dict[str, Any]:
    """
    Trains baseline and advanced models (Logistic Regression, Random Forest, XGBoost).
    Tracks all experiments, metrics, parameters, and artifacts with MLflow.
    Identifies and registers the champion model.
    """
    if mart_file is None:
        mart_file = ANALYTICAL_DATA_DIR / "student_analytical_mart.csv"

    logger.info(f"Loading analytical mart from {mart_file}...")
    if not mart_file.exists():
        err_msg = f"Analytical mart file not found at {mart_file}. Please run the ETL pipeline first."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    df = pd.read_csv(mart_file)

    # Prepare features and partitions
    store = AcademicFeatureStore()
    X_train, y_train, X_val, y_val, X_test, y_test = store.prepare_data(df)
    feature_names = store.get_feature_names()

    # Configure MLflow tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow tracking configured at: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"Could not connect to external MLflow URI, using local fallback: {e}")

    # Model definitions
    candidate_models = {
        "LogisticRegression": {
            "model": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            "params": {"C": 1.0, "penalty": "l2", "class_weight": "balanced"}
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            "params": {"n_estimators": 150, "max_depth": 12, "min_samples_split": 4}
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=RANDOM_SEED
            ),
            "params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.85}
        }
    }

    model_eval_results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None

    for name, item in candidate_models.items():
        logger.info(f"--- Training Model: {name} ---")
        clf = item["model"]
        params = item["params"]

        # MLflow run tracking
        try:
            with mlflow.start_run(run_name=f"model_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params(params)
                mlflow.log_param("model_type", name)
                mlflow.log_param("train_samples", len(X_train))

                # Train
                clf.fit(X_train, y_train)

                # Evaluate on Validation set
                val_metrics = evaluate_classification_model(clf, X_val, y_val, split_name="val")
                # Evaluate on Test set
                test_metrics = evaluate_classification_model(clf, X_test, y_test, split_name="test")

                # Log metrics to MLflow
                for k, v in {**val_metrics, **test_metrics}.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                # Feature importances
                top_features = extract_feature_importance(clf, feature_names)

                model_eval_results[name] = {
                    "model_name": name,
                    "params": params,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "top_features": top_features
                }

                # Log model artifact with MLflow
                try:
                    mlflow.sklearn.log_model(clf, artifact_path=name)
                except Exception as log_err:
                    logger.warning(f"Could not serialize model to MLflow artifact repo: {log_err}")

        except Exception as run_err:
            logger.warning(f"MLflow run failed for {name}, executing local fallback: {run_err}")
            clf.fit(X_train, y_train)
            val_metrics = evaluate_classification_model(clf, X_val, y_val, split_name="val")
            test_metrics = evaluate_classification_model(clf, X_test, y_test, split_name="test")
            top_features = extract_feature_importance(clf, feature_names)
            model_eval_results[name] = {
                "model_name": name,
                "params": params,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "top_features": top_features
            }

        # Select champion model based on Test F1-Score
        current_test_f1 = model_eval_results[name]["test_metrics"]["test_f1"]
        if current_test_f1 > best_f1:
            best_f1 = current_test_f1
            best_model_name = name
            best_model_obj = clf

    # Save champion model locally
    best_model_file = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model_obj, best_model_file)
    logger.info(f"Champion Model [{best_model_name}] saved to {best_model_file} with Test F1: {best_f1:.4f}")

    # Generate metadata card
    champion_metadata = {
        "model_name": best_model_name,
        "trained_at": datetime.now().isoformat(),
        "framework": best_model_obj.__class__.__module__,
        "champion_test_metrics": model_eval_results[best_model_name]["test_metrics"],
        "champion_val_metrics": model_eval_results[best_model_name]["val_metrics"],
        "hyperparameters": model_eval_results[best_model_name]["params"],
        "top_features": model_eval_results[best_model_name]["top_features"],
        "all_model_comparison": {
            m: {
                "test_accuracy": res["test_metrics"]["test_accuracy"],
                "test_precision": res["test_metrics"]["test_precision"],
                "test_recall": res["test_metrics"]["test_recall"],
                "test_f1": res["test_metrics"]["test_f1"],
                "test_roc_auc": res["test_metrics"]["test_roc_auc"]
            }
            for m, res in model_eval_results.items()
        }
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(champion_metadata, f, indent=4)

    logger.info(f"Model metadata card written to {metadata_path}")
    return {
        "best_model_name": best_model_name,
        "best_model_metrics": model_eval_results[best_model_name]["test_metrics"],
        "model_comparison": champion_metadata["all_model_comparison"]
    }

if __name__ == "__main__":
    train_all_models()
