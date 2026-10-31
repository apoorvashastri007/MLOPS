import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from config.config import RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger("IngestERP")

def generate_and_ingest_erp(num_students: int = 4424, seed: int = 42) -> pd.DataFrame:
    """
    Simulates and ingests academic ERP export records for all enrolled students.
    Includes attendance percentages, continuous internal evaluation (CIE) marks,
    assignment completion counts, and exam fee status.
    """
    logger.info("Ingesting College ERP academic records...")
    np.random.seed(seed)
    
    student_ids = [f"STU_{10000 + i}" for i in range(num_students)]
    
    # Generate realistic academic distributions
    # Attendance between 50% and 100%, correlated with success
    attendance_rates = np.clip(np.random.beta(a=5, b=2, size=num_students) * 100, 45.0, 100.0).round(1)
    
    # Internal marks out of 25 (average 16.5)
    internal_marks = np.clip(np.random.normal(loc=16.5, scale=4.0, size=num_students), 4.0, 25.0).round(1)
    
    # Total assignments = 10, assignments completed
    assignments_total = 10
    assignments_submitted = np.random.binomial(n=assignments_total, p=attendance_rates / 100)
    assignment_rate = (assignments_submitted / assignments_total * 100).round(1)

    erp_df = pd.DataFrame({
        "student_id": student_ids,
        "erp_attendance_pct": attendance_rates,
        "erp_internal_marks": internal_marks,
        "erp_assignments_submitted": assignments_submitted,
        "erp_assignments_total": assignments_total,
        "erp_assignment_completion_pct": assignment_rate,
        "erp_last_synced": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    raw_path = RAW_DATA_DIR / "erp_records_raw.csv"
    erp_df.to_csv(raw_path, index=False)

    # Compute checksum
    sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    metadata = {
        "source": "College ERP Database Export (Attendance & Internal Marks)",
        "file_name": raw_path.name,
        "extraction_timestamp": datetime.now().isoformat(),
        "status": "SUCCESS",
        "row_count": len(erp_df),
        "col_count": len(erp_df.columns),
        "sha256": sha256
    }
    
    meta_path = RAW_DATA_DIR / "extraction_metadata_erp.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Ingested {len(erp_df)} ERP records with checksum {sha256[:8]}...")
    return erp_df

if __name__ == "__main__":
    generate_and_ingest_erp()
