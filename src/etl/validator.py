import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from config.config import CLEANED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger("DataValidator")

class DataValidator:
    """
    Enforces academic data quality checks and validation rules:
    - Verifies valid ranges for marks (0-20 for European curricular units, 0-200 for admission grade)
    - Validates age, attendance percentages (0-100%), and enrollment counts
    - Removes duplicates and records with critical structural errors
    - Emits a clean dataset, a rejected-records dataset, and a validation audit report.
    """

    def __init__(self):
        self.rules = [
            ("student_id_not_null", "Student ID must not be null"),
            ("age_valid", "Age at enrollment must be between 16 and 80"),
            ("admission_grade_valid", "Admission grade must be between 0 and 200"),
            ("sem1_grade_valid", "Curricular units 1st sem grade must be between 0 and 20"),
            ("sem2_grade_valid", "Curricular units 2nd sem grade must be between 0 and 20"),
            ("attendance_valid", "Attendance rate must be between 0% and 100%"),
            ("target_valid", "Target must be one of: Dropout, Graduate, Enrolled"),
        ]

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        logger.info(f"Initiating validation checks on {len(df)} incoming staged records...")
        initial_count = len(df)
        
        # Track rejection reasons
        rejection_reasons: Dict[int, List[str]] = {}

        for idx, row in df.iterrows():
            reasons = []
            
            # Check 1: Student ID
            if pd.isna(row.get("student_id")):
                reasons.append("Missing Student ID")
                
            # Check 2: Age range
            age = row.get("Age at enrollment", 20)
            if age < 16 or age > 80:
                reasons.append(f"Invalid age: {age}")
                
            # Check 3: Admission Grade
            adm_grade = row.get("Admission grade", 100)
            if adm_grade < 0 or adm_grade > 200:
                reasons.append(f"Invalid admission grade: {adm_grade}")
                
            # Check 4: Sem 1 Grade
            sem1_grade = row.get("Curricular units 1st sem (grade)", 10)
            if sem1_grade < 0 or sem1_grade > 20:
                reasons.append(f"Invalid sem 1 grade: {sem1_grade}")
                
            # Check 5: Sem 2 Grade
            sem2_grade = row.get("Curricular units 2nd sem (grade)", 10)
            if sem2_grade < 0 or sem2_grade > 20:
                reasons.append(f"Invalid sem 2 grade: {sem2_grade}")
                
            # Check 6: Attendance
            att = row.get("erp_attendance_pct", 80)
            if att < 0 or att > 100:
                reasons.append(f"Invalid attendance percentage: {att}")
                
            # Check 7: Target
            tgt = str(row.get("Target", "")).strip()
            if tgt not in ["Dropout", "Graduate", "Enrolled"]:
                reasons.append(f"Invalid target category: {tgt}")

            if reasons:
                rejection_reasons[idx] = reasons

        # Split into Valid and Rejected
        rejected_indices = list(rejection_reasons.keys())
        rejected_df = df.loc[rejected_indices].copy()
        if not rejected_df.empty:
            rejected_df["rejection_reason"] = ["; ".join(rejection_reasons[i]) for i in rejected_indices]
            
        clean_df = df.drop(index=rejected_indices).copy()

        # Check and eliminate duplicate records
        duplicates_mask = clean_df.duplicated(subset=["student_id"], keep="first")
        num_duplicates = int(duplicates_mask.sum())
        clean_df = clean_df[~duplicates_mask]

        # Save files
        rejected_path = CLEANED_DATA_DIR / "rejected_records.csv"
        rejected_df.to_csv(rejected_path, index=False)
        
        valid_count = len(clean_df)
        rejected_count = len(rejected_df)

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_records_evaluated": initial_count,
            "passed_records": valid_count,
            "rejected_records": rejected_count,
            "duplicates_removed": num_duplicates,
            "quality_pass_rate_pct": round((valid_count / initial_count) * 100, 2) if initial_count else 0,
            "violation_summary": {
                f"Rule: {rule[0]}": sum(1 for r_list in rejection_reasons.values() if any(rule[0] in r.lower() or rule[1].lower() in r.lower() for r in r_list))
                for rule in self.rules
            }
        }

        report_path = CLEANED_DATA_DIR / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Validation finished: {valid_count} passed ({report['quality_pass_rate_pct']}%), {rejected_count} rejected. Report logged to {report_path}")
        return clean_df, rejected_df, report

if __name__ == "__main__":
    validator = DataValidator()
