import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
from config.config import SOURCE_CSV, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger("IngestUCI")

def compute_checksum(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for data integrity tracking."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def ingest_uci_dataset(source_path: Path = SOURCE_CSV) -> pd.DataFrame:
    """
    Ingests the primary UCI Student Performance & Dropout dataset.
    - Preserves pristine raw copy in data/raw/
    - Records extraction date, status, checksum, and row count
    - Handles formatting artifacts (semicolon separators, tab characters)
    """
    logger.info(f"Starting ingestion from primary source: {source_path}")
    if not source_path.exists():
        err_msg = f"Source dataset file not found at: {source_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Compute raw checksum
    checksum = compute_checksum(source_path)

    # Copy pristine raw file to raw data directory
    raw_target = RAW_DATA_DIR / "uci_student_raw.csv"
    shutil.copy2(source_path, raw_target)

    # Ingest using Pandas
    try:
        # File is semicolon-delimited
        df = pd.read_csv(raw_target, sep=";")
        
        # Clean column names (strip whitespace and rogue tab characters)
        df.columns = [col.strip().replace('"', '').replace('\t', '') for col in df.columns]
        
        row_count = len(df)
        col_count = len(df.columns)
        logger.info(f"Successfully ingested {row_count} rows and {col_count} columns from UCI source.")

        # Record extraction metadata
        metadata = {
            "source": "UCI Student Performance and Dropout Dataset",
            "file_name": source_path.name,
            "extraction_timestamp": datetime.now().isoformat(),
            "status": "SUCCESS",
            "row_count": row_count,
            "col_count": col_count,
            "sha256": checksum,
            "columns": list(df.columns)
        }
        
        meta_file = RAW_DATA_DIR / "extraction_metadata_uci.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Extraction metadata saved to {meta_file}")
        return df

    except Exception as e:
        logger.error(f"Failed to ingest UCI dataset: {e}")
        # Log failure metadata
        fail_meta = {
            "source": "UCI Student Performance and Dropout Dataset",
            "extraction_timestamp": datetime.now().isoformat(),
            "status": "FAILED",
            "error": str(e)
        }
        with open(RAW_DATA_DIR / "extraction_metadata_uci_error.json", "w", encoding="utf-8") as f:
            json.dump(fail_meta, f, indent=4)
        raise e

if __name__ == "__main__":
    ingest_uci_dataset()
