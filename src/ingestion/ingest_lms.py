import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from config.config import RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger("IngestLMS")

def generate_and_ingest_lms(num_students: int = 4424, seed: int = 101) -> pd.DataFrame:
    """
    Simulates and ingests Moodle/Canvas Learning Management System (LMS) activity logs.
    Captures portal logins, quiz completion counts, resource downloads, and active engagement score.
    """
    logger.info("Ingesting LMS engagement and activity logs...")
    np.random.seed(seed)
    
    student_ids = [f"STU_{10000 + i}" for i in range(num_students)]
    
    # Portal logins in past semester (10 to 250 logins)
    login_counts = np.random.poisson(lam=75, size=num_students)
    
    # Quiz attempts (total 15 quizzes)
    quiz_attempts = np.random.binomial(n=15, p=0.8, size=num_students)
    
    # Discussion forum postings
    forum_posts = np.random.poisson(lam=4, size=num_students)
    
    # Composite LMS activity engagement score (0 to 100)
    activity_score = np.clip(
        (login_counts * 0.4) + (quiz_attempts * 3.5) + (forum_posts * 2.0),
        10.0,
        100.0
    ).round(1)

    lms_df = pd.DataFrame({
        "student_id": student_ids,
        "lms_login_count": login_counts,
        "lms_quiz_attempts": quiz_attempts,
        "lms_forum_posts": forum_posts,
        "lms_activity_score": activity_score,
        "lms_source_platform": "Moodle LMS v4.2",
        "lms_extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    raw_path = RAW_DATA_DIR / "lms_logs_raw.csv"
    lms_df.to_csv(raw_path, index=False)

    # Checksum & Metadata
    sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    metadata = {
        "source": "Moodle LMS Activity Export",
        "file_name": raw_path.name,
        "extraction_timestamp": datetime.now().isoformat(),
        "status": "SUCCESS",
        "row_count": len(lms_df),
        "col_count": len(lms_df.columns),
        "sha256": sha256
    }
    
    meta_path = RAW_DATA_DIR / "extraction_metadata_lms.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Ingested {len(lms_df)} LMS records with checksum {sha256[:8]}...")
    return lms_df

if __name__ == "__main__":
    generate_and_ingest_lms()
