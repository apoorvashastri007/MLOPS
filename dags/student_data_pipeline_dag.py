"""
Apache Airflow DAG for Student Performance and Dropout-Risk Analytics System.
Defines scheduled, orchestrated data engineering and MLOps tasks.
"""

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to sys.path for task execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Safe import of Airflow components with fallback runner for non-Airflow environments
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

default_args = {
    "owner": "DataEngineering_MLOps_Team",
    "depends_on_past": False,
    "start_date": datetime(2026, 9, 1),
    "email": ["academic.analytics@university.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Task execution wrappers
def task_ingest_uci():
    from src.ingestion.ingest_uci import ingest_uci_dataset
    print("Executing Task: Ingest UCI Dataset...")
    df = ingest_uci_dataset()
    print(f"Ingested {len(df)} records from UCI source.")

def task_ingest_erp():
    from src.ingestion.ingest_erp import generate_and_ingest_erp
    print("Executing Task: Ingest College ERP Records...")
    df = generate_and_ingest_erp()
    print(f"Ingested {len(df)} ERP records.")

def task_ingest_lms():
    from src.ingestion.ingest_lms import generate_and_ingest_lms
    print("Executing Task: Ingest Moodle LMS Logs...")
    df = generate_and_ingest_lms()
    print(f"Ingested {len(df)} LMS log records.")

def task_run_etl():
    from src.etl.etl_pipeline import run_etl_pipeline
    print("Executing Task: Run Staging, Validation, Star Schema Transformation & DB Loading...")
    summary = run_etl_pipeline()
    print(f"ETL completed: {summary['clean_count']} records loaded into database warehouse.")

def task_train_mlops():
    from src.mlops.train import train_all_models
    print("Executing Task: Train Machine Learning Baseline and Advanced Models...")
    results = train_all_models()
    print(f"MLOps Training Completed. Best Model: {results.get('best_model_name')}")

def task_monitor_drift():
    from src.mlops.monitoring import run_drift_monitoring
    print("Executing Task: Run Feature Drift & Data Quality Monitoring...")
    drift_report = run_drift_monitoring()
    print(f"Drift Monitoring Completed. Status: {drift_report.get('overall_drift_status')}")

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="student_performance_data_pipeline",
        default_args=default_args,
        description="End-to-End Ingestion, ETL, Warehouse Load & MLOps Pipeline for Academic Risk",
        schedule_interval="@weekly",
        catchup=False,
        tags=["academic_analytics", "etl", "mlops", "student_retention"]
    ) as dag:
        
        start = EmptyOperator(task_id="start_pipeline")
        
        ingest_uci = PythonOperator(
            task_id="ingest_uci_dataset",
            python_callable=task_ingest_uci
        )
        
        ingest_erp = PythonOperator(
            task_id="ingest_erp_records",
            python_callable=task_ingest_erp
        )
        
        ingest_lms = PythonOperator(
            task_id="ingest_lms_logs",
            python_callable=task_ingest_lms
        )
        
        etl_and_load = PythonOperator(
            task_id="etl_transform_and_load_warehouse",
            python_callable=task_run_etl
        )
        
        train_models = PythonOperator(
            task_id="train_mlops_models",
            python_callable=task_train_mlops
        )
        
        check_drift = PythonOperator(
            task_id="monitor_data_and_concept_drift",
            python_callable=task_monitor_drift
        )
        
        end = EmptyOperator(task_id="end_pipeline")
        
        # Pipeline DAG dependencies
        start >> [ingest_uci, ingest_erp, ingest_lms] >> etl_and_load >> train_models >> check_drift >> end

def execute_standalone():
    """Fallback orchestrator for environments where Apache Airflow server is not configured."""
    print("Running Pipeline in Standalone Airflow-Compatible Execution Mode...")
    task_ingest_uci()
    task_ingest_erp()
    task_ingest_lms()
    task_run_etl()
    task_train_mlops()
    task_monitor_drift()
    print("Standalone pipeline execution finished successfully.")

if __name__ == "__main__":
    execute_standalone()
