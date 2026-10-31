import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from config.config import (
    RAW_DATA_DIR,
    STAGING_DATA_DIR,
    CLEANED_DATA_DIR,
    ANALYTICAL_DATA_DIR
)
from config.logging_config import get_logger
from src.ingestion.ingestion_pipeline import run_ingestion_pipeline
from src.etl.validator import DataValidator
from src.etl.transformer import DataTransformer
from database.db_manager import DatabaseManager

logger = get_logger("ETLPipeline")

def run_etl_pipeline():
    """
    Executes the full End-to-End ETL and Data Engineering pipeline:
    1. Multi-source Ingestion (Raw Layer)
    2. Data Standardization & Merging (Staging Layer)
    3. Quality Validation, Deduplication & Error Logging (Cleaned Layer)
    4. Transformation, Feature Engineering & Dimensional Partitioning (Analytical Layer)
    5. Loading Star Schema & Analytical Mart into Database (Storage Layer)
    """
    logger.info("==================================================")
    logger.info("   STUDENT ANALYTICS ETL & DATA PIPELINE START    ")
    logger.info("==================================================")
    start_time = datetime.now()

    # Step 1: Ensure Raw Ingestion
    uci_raw_file = RAW_DATA_DIR / "uci_student_raw.csv"
    if not uci_raw_file.exists():
        logger.info("Raw ingested data not found. Running Ingestion Pipeline...")
        uci_df, erp_df, lms_df, _ = run_ingestion_pipeline()
    else:
        logger.info("Loading existing raw ingested datasets...")
        uci_df = pd.read_csv(uci_raw_file, sep=";")
        uci_df.columns = [c.strip().replace('"', '').replace('\t', '') for c in uci_df.columns]
        
        erp_raw_file = RAW_DATA_DIR / "erp_records_raw.csv"
        erp_df = pd.read_csv(erp_raw_file) if erp_raw_file.exists() else None
        
        lms_raw_file = RAW_DATA_DIR / "lms_logs_raw.csv"
        lms_df = pd.read_csv(lms_raw_file) if lms_raw_file.exists() else None

    # Step 2: Staging Layer - Standardize Identifiers and Join Sources
    logger.info("Assembling Staging Layer with standardized student identifiers...")
    if "student_id" not in uci_df.columns:
        uci_df.insert(0, "student_id", [f"STU_{10000 + i}" for i in range(len(uci_df))])

    staged_df = uci_df.copy()
    if erp_df is not None:
        staged_df = staged_df.merge(
            erp_df[["student_id", "erp_attendance_pct", "erp_internal_marks", "erp_assignment_completion_pct"]],
            on="student_id",
            how="left"
        )
    if lms_df is not None:
        staged_df = staged_df.merge(
            lms_df[["student_id", "lms_login_count", "lms_quiz_attempts", "lms_activity_score"]],
            on="student_id",
            how="left"
        )

    staging_file = STAGING_DATA_DIR / "staged_student_data.csv"
    staged_df.to_csv(staging_file, index=False)
    logger.info(f"Staged {len(staged_df)} records saved to {staging_file}")

    # Step 3: Cleaned Layer - Quality Checks and Validation
    logger.info("Running Data Quality Checks and Validation Rules...")
    validator = DataValidator()
    clean_df, rejected_df, val_report = validator.validate(staged_df)

    clean_file = CLEANED_DATA_DIR / "cleaned_students.csv"
    clean_df.to_csv(clean_file, index=False)
    logger.info(f"Cleaned dataset saved: {len(clean_df)} records passed quality gates.")

    # Step 4: Analytical Layer - Feature Engineering & Star Schema
    logger.info("Transforming into Star Schema and Analytical Mart...")
    transformer = DataTransformer()
    schema_tables, student_analytical_mart = transformer.transform(clean_df)

    # Step 5: Database Loading
    logger.info("Loading Star Schema and Analytical Mart into Database...")
    db = DatabaseManager()
    db.initialize_schema()
    for table_name, df_table in schema_tables.items():
        db.load_table(df_table, table_name, if_exists="replace")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    db_counts = db.get_table_counts()

    pipeline_summary = {
        "pipeline": "Student Performance ETL Pipeline",
        "timestamp": end_time.isoformat(),
        "duration_seconds": round(duration, 2),
        "status": "COMPLETED_SUCCESSFULLY",
        "staged_count": len(staged_df),
        "clean_count": len(clean_df),
        "rejected_count": len(rejected_df),
        "database_table_counts": db_counts
    }

    summary_path = CLEANED_DATA_DIR / "etl_execution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=4)

    logger.info("==================================================")
    logger.info(f"   ETL PIPELINE COMPLETED IN {duration:.2f} SECONDS    ")
    logger.info(f"   Populated tables: {db_counts}")
    logger.info("==================================================")
    return pipeline_summary

if __name__ == "__main__":
    run_etl_pipeline()
