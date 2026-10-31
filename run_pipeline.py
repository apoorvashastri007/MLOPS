"""
End-to-End Orchestrator for Student Performance and Dropout-Risk Analytics System.
Executes complete Part 1 (Data Engineering) and Part 2 (MLOps) pipelines.
"""

import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.ingestion_pipeline import run_ingestion_pipeline
from src.etl.etl_pipeline import run_etl_pipeline
from src.mlops.train import train_all_models
from src.mlops.monitoring import run_drift_monitoring
from config.logging_config import get_logger

logger = get_logger("Orchestrator")

def run_full_system_pipeline():
    print("\n" + "="*70)
    print(" [START] STUDENT ANALYTICS & MLOps END-TO-END PIPELINE ")
    print("="*70)
    total_start = time.time()

    # PHASE 1: Multi-Source Data Ingestion
    print("\n[PHASE 1/4] Ingesting Multi-Source Academic Records (UCI, ERP, LMS)...")
    t1 = time.time()
    uci_df, erp_df, lms_df, ingest_summary = run_ingestion_pipeline()
    print(f" -> Ingestion Complete in {time.time()-t1:.2f}s:")
    print(f"    - UCI Historical Records: {len(uci_df):,}")
    print(f"    - College ERP Records:    {len(erp_df):,}")
    print(f"    - Moodle LMS Activity:    {len(lms_df):,}")

    # PHASE 2: ETL, Data Quality Validation & Database Warehouse Loading
    print("\n[PHASE 2/4] Executing ETL, Data Quality Gates & Star Schema Warehouse Load...")
    t2 = time.time()
    etl_summary = run_etl_pipeline()
    print(f" -> ETL Pipeline Complete in {time.time()-t2:.2f}s:")
    print(f"    - Clean Records Passed:   {etl_summary['clean_count']:,}")
    print(f"    - Rejected/Failed Logs:   {etl_summary['rejected_count']:,}")
    print("    - Populated Warehouse Tables:")
    for tbl, cnt in etl_summary["database_table_counts"].items():
        print(f"      * {tbl:<26}: {cnt:,} rows")

    # PHASE 3: MLOps Model Training, Experiment Tracking & Champion Selection
    print("\n[PHASE 3/4] Training Baseline & Advanced ML Models (Logistic Regression, RF, XGBoost)...")
    t3 = time.time()
    train_results = train_all_models()
    print(f" -> Model Training Complete in {time.time()-t3:.2f}s:")
    print(f"    - Champion Model:         {train_results['best_model_name']}")
    print(f"    - Champion Test Accuracy: {train_results['best_model_metrics']['test_accuracy']:.2%}")
    print(f"    - Champion Test F1-Score: {train_results['best_model_metrics']['test_f1']:.4f}")
    print(f"    - Champion Test ROC-AUC:  {train_results['best_model_metrics']['test_roc_auc']:.4f}")
    print("    - Model Benchmark Summary:")
    for model_name, metrics in train_results["model_comparison"].items():
        print(f"      * {model_name:<20}: Acc={metrics['test_accuracy']:.2%}, F1={metrics['test_f1']:.4f}, AUC={metrics['test_roc_auc']:.4f}")

    # PHASE 4: Data Drift & Model Monitoring
    print("\n[PHASE 4/4] Executing Kolmogorov-Smirnov & PSI Feature Drift Monitoring...")
    t4 = time.time()
    drift_report = run_drift_monitoring()
    print(f" -> Drift Monitoring Complete in {time.time()-t4:.2f}s:")
    print(f"    - Drift Status:           {drift_report['overall_drift_status']}")
    print(f"    - Drifted Features:       {drift_report['drifted_features_count']} / {drift_report['total_features_monitored']}")
    print(f"    - Retraining Recommended: {'YES [WARNING]' if drift_report['retraining_recommended'] else 'NO [HEALTHY]'}")

    total_time = time.time() - total_start
    print("\n" + "="*70)
    print(f" [SUCCESS] PIPELINE EXECUTION COMPLETED IN {total_time:.2f} SECONDS")
    print("="*70)
    print(" Deliverables Generated:")
    print("  1. Raw Ingestion Layer:      data/raw/")
    print("  2. Cleaned & Error Logs:     data/cleaned/")
    print("  3. Star Schema Data Mart:    data/analytical/")
    print("  4. Relational Database:      database/student_warehouse.db")
    print("  5. Serialized Champion Model:models/best_model.joblib")
    print("  6. Drift Monitoring Report:  reports/drift_monitoring_report.json")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_full_system_pipeline()
