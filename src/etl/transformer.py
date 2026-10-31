import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from config.config import ANALYTICAL_DATA_DIR, CLEANED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger("Transformer")

# Official Course Code to Degree Name Mapping (Polytechnic Institute of Portalegre)
COURSE_MAPPING = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service (Evening)",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening)"
}

# Qualification and Occupation Mappings for Demographics
QUALIFICATION_MAPPING = {
    1: "Secondary Education - 12th Year",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year of Schooling",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year)",
    37: "Basic Education 1st Cycle (4th/5th Year)",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year)"
}

class DataTransformer:
    """
    Transforms cleansed data into star-schema dimension tables, fact tables,
    and a unified analytical data mart with advanced academic feature engineering.
    """

    def __init__(self):
        pass

    def transform(self, clean_df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        logger.info("Starting transformation and feature engineering pipeline...")
        df = clean_df.copy()

        # 1. Standardize and engineer features
        # Normalize course names
        df["course_name"] = df["Course"].map(lambda c: COURSE_MAPPING.get(c, f"Course Code {c}"))
        df["attendance_mode"] = df["Daytime/evening attendance"].map(lambda x: "Daytime" if x == 1 else "Evening")
        df["gender_label"] = df["Gender"].map(lambda x: "Male" if x == 1 else "Female")
        
        # Flags
        df["debtor_flag"] = df["Debtor"].map(lambda x: "Yes" if x == 1 else "No")
        df["tuition_fees_flag"] = df["Tuition fees up to date"].map(lambda x: "Yes" if x == 1 else "No")
        df["scholarship_flag"] = df["Scholarship holder"].map(lambda x: "Yes" if x == 1 else "No")
        df["international_flag"] = df["International"].map(lambda x: "Yes" if x == 1 else "No")
        df["displaced_flag"] = df["Displaced"].map(lambda x: "Yes" if x == 1 else "No")
        df["special_needs_flag"] = df["Educational special needs"].map(lambda x: "Yes" if x == 1 else "No")

        # Academic aggregations & features
        sem1_enrolled = df["Curricular units 1st sem (enrolled)"].clip(lower=0)
        sem1_approved = df["Curricular units 1st sem (approved)"].clip(lower=0)
        sem2_enrolled = df["Curricular units 2nd sem (enrolled)"].clip(lower=0)
        sem2_approved = df["Curricular units 2nd sem (approved)"].clip(lower=0)

        total_enrolled = sem1_enrolled + sem2_enrolled
        total_approved = sem1_approved + sem2_approved

        df["sem1_approval_rate"] = np.where(sem1_enrolled > 0, sem1_approved / sem1_enrolled, 0.0).round(3)
        df["sem2_approval_rate"] = np.where(sem2_enrolled > 0, sem2_approved / sem2_enrolled, 0.0).round(3)
        df["approval_rate_overall"] = np.where(total_enrolled > 0, total_approved / total_enrolled, 0.0).round(3)

        df["grade_progression"] = (
            df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
        ).round(2)

        df["financial_stress_flag"] = (
            (df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0)
        ).astype(int)

        # ERP and LMS fields
        df["attendance_rate"] = df.get("erp_attendance_pct", 85.0).round(1)
        df["internal_marks_avg"] = df.get("erp_internal_marks", 15.0).round(1)
        df["assignment_completion_rate"] = df.get("erp_assignment_completion_pct", 80.0).round(1)
        df["lms_activity_score"] = df.get("lms_activity_score", 70.0).round(1)

        # Target classification
        df["is_dropout"] = (df["Target"].str.strip() == "Dropout").astype(int)

        # Multi-factorial risk level assignment
        def assign_risk_level(row):
            score = 0
            if row["Target"] == "Dropout":
                score += 3
            if row["approval_rate_overall"] < 0.50:
                score += 2
            elif row["approval_rate_overall"] < 0.75:
                score += 1
            if row["attendance_rate"] < 75.0:
                score += 2
            if row["financial_stress_flag"] == 1:
                score += 1
            if row["grade_progression"] < -2.0:
                score += 1
            
            if score >= 4:
                return "High Risk"
            elif score >= 2:
                return "Moderate Risk"
            return "Low Risk"

        df["risk_level"] = df.apply(assign_risk_level, axis=1)

        # 2. Extract Star Schema Dimension Tables
        # Dim Student
        dim_student = pd.DataFrame({
            "student_id": df["student_id"],
            "gender": df["gender_label"],
            "age_at_enrollment": df["Age at enrollment"],
            "international": df["international_flag"],
            "displaced": df["displaced_flag"],
            "special_needs": df["special_needs_flag"],
            "debtor": df["debtor_flag"],
            "tuition_fees_up_to_date": df["tuition_fees_flag"],
            "scholarship_holder": df["scholarship_flag"]
        }).drop_duplicates(subset=["student_id"])

        # Dim Course
        dim_course = pd.DataFrame({
            "course_id": df["Course"],
            "course_code": df["Course"].astype(str),
            "course_name": df["course_name"],
            "attendance_mode": df["attendance_mode"]
        }).drop_duplicates(subset=["course_id"]).sort_values("course_id")

        # Dim Demographics
        dim_demographics = pd.DataFrame({
            "student_id": df["student_id"],
            "marital_status": df["Marital status"].astype(str),
            "nationality": df["Nacionality"].astype(str),
            "mother_qualification": df["Mother's qualification"].map(lambda q: QUALIFICATION_MAPPING.get(q, f"Qualification {q}")),
            "father_qualification": df["Father's qualification"].map(lambda q: QUALIFICATION_MAPPING.get(q, f"Qualification {q}")),
            "mother_occupation": df["Mother's occupation"].astype(str),
            "father_occupation": df["Father's occupation"].astype(str)
        }).drop_duplicates(subset=["student_id"])

        # Dim Macro Economy
        dim_macro_economy = pd.DataFrame({
            "student_id": df["student_id"],
            "unemployment_rate": df["Unemployment rate"],
            "inflation_rate": df["Inflation rate"],
            "gdp_growth": df["GDP"]
        }).drop_duplicates(subset=["student_id"])

        # Fact Student Performance
        fact_student_performance = pd.DataFrame({
            "student_id": df["student_id"],
            "course_id": df["Course"],
            "admission_grade": df["Admission grade"],
            "previous_qualification_grade": df["Previous qualification (grade)"],
            "sem1_enrolled": df["Curricular units 1st sem (enrolled)"],
            "sem1_evaluations": df["Curricular units 1st sem (evaluations)"],
            "sem1_approved": df["Curricular units 1st sem (approved)"],
            "sem1_grade": df["Curricular units 1st sem (grade)"],
            "sem2_enrolled": df["Curricular units 2nd sem (enrolled)"],
            "sem2_evaluations": df["Curricular units 2nd sem (evaluations)"],
            "sem2_approved": df["Curricular units 2nd sem (approved)"],
            "sem2_grade": df["Curricular units 2nd sem (grade)"],
            "attendance_rate": df["attendance_rate"],
            "internal_marks_avg": df["internal_marks_avg"],
            "assignment_completion_rate": df["assignment_completion_rate"],
            "lms_activity_score": df["lms_activity_score"],
            "target": df["Target"],
            "is_dropout": df["is_dropout"]
        })

        # 3. Unified Analytical Data Mart
        student_analytical_mart = pd.DataFrame({
            "student_id": df["student_id"],
            "gender": df["gender_label"],
            "age_at_enrollment": df["Age at enrollment"],
            "debtor": df["debtor_flag"],
            "tuition_fees_up_to_date": df["tuition_fees_flag"],
            "scholarship_holder": df["scholarship_flag"],
            "course_name": df["course_name"],
            "attendance_mode": df["attendance_mode"],
            "admission_grade": df["Admission grade"],
            "sem1_enrolled": df["Curricular units 1st sem (enrolled)"],
            "sem1_approved": df["Curricular units 1st sem (approved)"],
            "sem1_grade": df["Curricular units 1st sem (grade)"],
            "sem2_enrolled": df["Curricular units 2nd sem (enrolled)"],
            "sem2_approved": df["Curricular units 2nd sem (approved)"],
            "sem2_grade": df["Curricular units 2nd sem (grade)"],
            "attendance_rate": df["attendance_rate"],
            "internal_marks_avg": df["internal_marks_avg"],
            "assignment_completion_rate": df["assignment_completion_rate"],
            "lms_activity_score": df["lms_activity_score"],
            "approval_rate_overall": df["approval_rate_overall"],
            "grade_progression": df["grade_progression"],
            "financial_stress_flag": df["financial_stress_flag"],
            "target": df["Target"],
            "is_dropout": df["is_dropout"],
            "risk_level": df["risk_level"]
        })

        # Save analytical outputs to disk
        dim_student.to_csv(ANALYTICAL_DATA_DIR / "dim_student.csv", index=False)
        dim_course.to_csv(ANALYTICAL_DATA_DIR / "dim_course.csv", index=False)
        dim_demographics.to_csv(ANALYTICAL_DATA_DIR / "dim_demographics.csv", index=False)
        dim_macro_economy.to_csv(ANALYTICAL_DATA_DIR / "dim_macro_economy.csv", index=False)
        fact_student_performance.to_csv(ANALYTICAL_DATA_DIR / "fact_student_performance.csv", index=False)
        student_analytical_mart.to_csv(ANALYTICAL_DATA_DIR / "student_analytical_mart.csv", index=False)

        schema_tables = {
            "dim_student": dim_student,
            "dim_course": dim_course,
            "dim_demographics": dim_demographics,
            "dim_macro_economy": dim_macro_economy,
            "fact_student_performance": fact_student_performance,
            "student_analytical_mart": student_analytical_mart
        }

        logger.info("Successfully generated dimensional tables and student analytical mart.")
        return schema_tables, student_analytical_mart

if __name__ == "__main__":
    transformer = DataTransformer()
