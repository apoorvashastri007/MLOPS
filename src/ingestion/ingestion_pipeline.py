import json
from datetime import datetime
from config.config import RAW_DATA_DIR
from config.logging_config import get_logger
from src.ingestion.ingest_uci import ingest_uci_dataset
from src.ingestion.ingest_erp import generate_and_ingest_erp
from src.ingestion.ingest_lms import generate_and_ingest_lms

logger = get_logger("IngestionPipeline")

def run_ingestion_pipeline():
    """
    Executes the complete multi-source ingestion workflow:
    1. Ingests primary UCI Student Performance dataset
    2. Ingests College ERP academic records (attendance, internal marks)
    3. Ingests LMS engagement logs (Moodle activity, quiz attempts)
    4. Generates an aggregated audit summary
    """
    logger.info("=== Starting Multi-Source Ingestion Pipeline ===")
    start_time = datetime.now()

    # Step 1: Ingest primary historical dataset
    uci_df = ingest_uci_dataset()

    # Step 2: Ingest ERP records
    erp_df = generate_and_ingest_erp(num_students=len(uci_df))

    # Step 3: Ingest LMS records
    lms_df = generate_and_ingest_lms(num_students=len(uci_df))

    end_time = datetime.now()
    duration_secs = (end_time - start_time).total_seconds()

    summary = {
        "pipeline_name": "Multi-Source Student Data Ingestion",
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "duration_seconds": round(duration_secs, 2),
        "status": "SUCCESS",
        "sources_ingested": [
            {
                "source": "UCI Student Performance and Dropout Dataset",
                "records": len(uci_df),
                "features": len(uci_df.columns),
                "target_file": "data/raw/uci_student_raw.csv"
            },
            {
                "source": "College ERP Export",
                "records": len(erp_df),
                "features": len(erp_df.columns),
                "target_file": "data/raw/erp_records_raw.csv"
            },
            {
                "source": "Moodle LMS Activity Export",
                "records": len(lms_df),
                "features": len(lms_df.columns),
                "target_file": "data/raw/lms_logs_raw.csv"
            }
        ],
        "total_records_ingested": len(uci_df)
    }

    summary_file = RAW_DATA_DIR / "ingestion_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Ingestion completed in {duration_secs:.2f}s. Summary logged to {summary_file}")
    return uci_df, erp_df, lms_df, summary

if __name__ == "__main__":
    run_ingestion_pipeline()
