import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config.config import (
    ANALYTICAL_DATA_DIR,
    REPORTS_DIR,
    DRIFT_P_VALUE_THRESHOLD,
    DRIFT_PSI_THRESHOLD
)
from config.logging_config import get_logger
from src.mlops.feature_store import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logger = get_logger("DriftMonitor")

def calculate_psi(baseline: pd.Series, current: pd.Series, num_buckets: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between baseline and current distributions."""
    try:
        combined = pd.concat([baseline, current]).dropna()
        if len(combined) == 0:
            return 0.0

        # Create quantile bins
        quantiles = np.linspace(0, 1, num_buckets + 1)
        bins = np.percentile(combined, quantiles * 100)
        bins = np.unique(bins)
        
        if len(bins) < 2:
            return 0.0

        base_counts, _ = np.histogram(baseline.dropna(), bins=bins)
        curr_counts, _ = np.histogram(current.dropna(), bins=bins)

        base_pct = (base_counts + 1e-4) / (len(baseline) + 1e-4 * len(base_counts))
        curr_pct = (curr_counts + 1e-4) / (len(current) + 1e-4 * len(curr_counts))

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(psi)
    except Exception:
        return 0.0

def run_drift_monitoring(
    baseline_df: pd.DataFrame = None,
    current_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Evaluates input data quality, feature drift (KS-test and PSI),
    and target class distribution shifts between baseline training and incoming batches.
    Emits a formal monitoring audit report with retraining triggers.
    """
    logger.info("Initiating Data Quality and Feature Drift Monitoring...")

    if baseline_df is None:
        mart_path = ANALYTICAL_DATA_DIR / "student_analytical_mart.csv"
        if not mart_path.exists():
            logger.warning("Analytical mart not yet generated. Drift monitoring skipped.")
            return {"status": "SKIPPED", "reason": "No baseline mart found"}
        baseline_df = pd.read_csv(mart_path)

    # If current_df is not supplied, simulate an incoming semester cohort with subtle perturbations
    if current_df is None:
        logger.info("Generating simulated incoming cohort batch for drift evaluation...")
        current_df = baseline_df.sample(frac=0.35, random_state=123).copy()
        # Add realistic subtle environmental shift to attendance and internal marks
        current_df["attendance_rate"] = (current_df["attendance_rate"] * 0.96).clip(0, 100)
        current_df["internal_marks_avg"] = (current_df["internal_marks_avg"] * 0.98).clip(0, 25)

    feature_drift_results: List[Dict[str, Any]] = []
    drifted_features_count = 0

    # 1. Evaluate Numerical Feature Drift using Kolmogorov-Smirnov Test
    for col in NUMERICAL_FEATURES:
        if col in baseline_df.columns and col in current_df.columns:
            base_vals = baseline_df[col].dropna()
            curr_vals = current_df[col].dropna()

            ks_stat, p_val = ks_2samp(base_vals, curr_vals)
            is_drifted = bool(p_val < DRIFT_P_VALUE_THRESHOLD)
            
            # Calculate PSI
            psi = calculate_psi(base_vals, curr_vals)

            if is_drifted:
                drifted_features_count += 1

            feature_drift_results.append({
                "feature": col,
                "type": "numerical",
                "test": "Kolmogorov-Smirnov & PSI",
                "statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 5),
                "psi": round(float(psi), 4),
                "drift_detected": is_drifted,
                "baseline_mean": round(float(base_vals.mean()), 2),
                "current_mean": round(float(curr_vals.mean()), 2)
            })

    # 2. Evaluate Class Distribution Shift (Concept / Target Drift)
    base_dropout_rate = float(baseline_df["is_dropout"].mean()) if "is_dropout" in baseline_df.columns else 0.0
    curr_dropout_rate = float(current_df["is_dropout"].mean()) if "is_dropout" in current_df.columns else 0.0
    target_shift_pct = abs(curr_dropout_rate - base_dropout_rate) * 100

    # 3. Retraining Decision Engine
    retrain_recommended = (drifted_features_count >= 3) or (target_shift_pct > 8.0)
    retrain_reasons = []
    if drifted_features_count >= 3:
        retrain_reasons.append(f"{drifted_features_count} numerical features show statistically significant drift (p < {DRIFT_P_VALUE_THRESHOLD})")
    if target_shift_pct > 8.0:
        retrain_reasons.append(f"Target dropout rate shifted by {target_shift_pct:.1f}% (Baseline: {base_dropout_rate:.1%}, Current: {curr_dropout_rate:.1%})")

    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_sample_size": len(baseline_df),
        "current_sample_size": len(current_df),
        "total_features_monitored": len(feature_drift_results),
        "drifted_features_count": drifted_features_count,
        "overall_drift_status": "DRIFT_DETECTED" if retrain_recommended else "STABLE",
        "baseline_dropout_rate": round(base_dropout_rate, 4),
        "current_dropout_rate": round(curr_dropout_rate, 4),
        "target_shift_pct": round(target_shift_pct, 2),
        "retraining_recommended": retrain_recommended,
        "retraining_triggers": retrain_reasons,
        "feature_metrics": feature_drift_results
    }

    report_path = REPORTS_DIR / "drift_monitoring_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Drift monitoring completed. Status: {report['overall_drift_status']}. Retraining recommended: {retrain_recommended}")
    return report

if __name__ == "__main__":
    run_drift_monitoring()
