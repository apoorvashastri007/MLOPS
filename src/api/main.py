import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config.config import MODELS_DIR, REPORTS_DIR
from config.logging_config import get_logger
from src.api.schemas import (
    StudentPredictionInput,
    StudentPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)

logger = get_logger("FastAPI_Service")

app = FastAPI(
    title="Student Performance & Dropout-Risk Prediction API",
    description="Production-ready MLOps prediction service providing real-time inference, risk stratification, and academic intervention triggers.",
    version="1.0.0"
)

# Enable CORS for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
PREPROCESSOR = None
METADATA = None

def load_inference_artifacts():
    """Loads the serialized model, preprocessor, and metadata from disk."""
    global MODEL, PREPROCESSOR, METADATA
    model_path = MODELS_DIR / "best_model.joblib"
    preproc_path = MODELS_DIR / "preprocessor.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    if model_path.exists() and preproc_path.exists():
        try:
            MODEL = joblib.load(model_path)
            PREPROCESSOR = joblib.load(preproc_path)
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    METADATA = json.load(f)
            logger.info("MLOps model and preprocessor artifacts loaded successfully into API memory.")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
    else:
        logger.warning("Model artifacts not yet found on disk. Run model training pipeline.")

@app.on_event("startup")
def startup_event():
    load_inference_artifacts()

def generate_interventions(prob: float, input_data: StudentPredictionInput) -> List[str]:
    """Generates rule-based proactive academic interventions based on risk factors."""
    interventions = []
    if input_data.attendance_rate < 75.0:
        interventions.append("Attendance Warning: Schedule mandatory counseling and attendance contract.")
    if input_data.sem1_approved < input_data.sem1_enrolled * 0.6:
        interventions.append("Remedial Tutoring: Enroll in remedial coursework for failed 1st semester subjects.")
    if input_data.internal_marks_avg < 12.0:
        interventions.append("Continuous Assessment Support: Assign faculty mentor for internal evaluations.")
    if input_data.debtor == "Yes" or input_data.tuition_fees_up_to_date == "No":
        interventions.append("Financial Aid Advisory: Connect student with university hardship fund / installment scheme.")
    if input_data.lms_activity_score < 40.0:
        interventions.append("LMS Engagement Trigger: Automated learning advisor check-in for digital portal activity.")
    
    if prob >= 0.70:
        interventions.append("High-Priority Intervention: Dean of Students review and comprehensive retention plan.")
    elif not interventions:
        interventions.append("Standard Monitoring: Student in good academic standing. Continue periodic progress reviews.")
        
    return interventions

def transform_single_student(student: StudentPredictionInput) -> pd.DataFrame:
    """Converts student schema into DataFrame matching the feature pipeline."""
    # Derived features
    total_enrolled = student.sem1_enrolled + student.sem2_enrolled
    total_approved = student.sem1_approved + student.sem2_approved
    approval_rate = round(total_approved / max(total_enrolled, 1), 3)
    progression = round(student.sem2_grade - student.sem1_grade, 2)

    data = {
        "age_at_enrollment": [student.age_at_enrollment],
        "admission_grade": [student.admission_grade],
        "sem1_enrolled": [student.sem1_enrolled],
        "sem1_approved": [student.sem1_approved],
        "sem1_grade": [student.sem1_grade],
        "sem2_enrolled": [student.sem2_enrolled],
        "sem2_approved": [student.sem2_approved],
        "sem2_grade": [student.sem2_grade],
        "attendance_rate": [student.attendance_rate],
        "internal_marks_avg": [student.internal_marks_avg],
        "assignment_completion_rate": [student.assignment_completion_rate],
        "lms_activity_score": [student.lms_activity_score],
        "approval_rate_overall": [approval_rate],
        "grade_progression": [progression],
        "gender": [student.gender],
        "debtor": [student.debtor],
        "tuition_fees_up_to_date": [student.tuition_fees_up_to_date],
        "scholarship_holder": [student.scholarship_holder],
        "attendance_mode": [student.attendance_mode],
        "course_name": [student.course_name]
    }
    return pd.DataFrame(data)

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Student Performance & Dropout-Risk Prediction System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "status": "ONLINE",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["General"])
def health_check():
    global MODEL, PREPROCESSOR
    if MODEL is None or PREPROCESSOR is None:
        load_inference_artifacts()
    return {
        "status": "healthy" if MODEL is not None else "model_pending",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_info", response_model=ModelInfoResponse, tags=["MLOps"])
def get_model_info():
    global METADATA
    if METADATA is None:
        load_inference_artifacts()
    if METADATA is None:
        raise HTTPException(status_code=404, detail="Model metadata not found. Train the model first.")
    return METADATA

@app.post("/predict", response_model=StudentPredictionResponse, tags=["Inference"])
def predict_student_risk(student: StudentPredictionInput):
    global MODEL, PREPROCESSOR, METADATA
    if MODEL is None or PREPROCESSOR is None:
        load_inference_artifacts()
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifacts not yet loaded.")

    try:
        df_input = transform_single_student(student)
        X_trans = PREPROCESSOR.transform(df_input)

        # Predict probability
        if hasattr(MODEL, "predict_proba"):
            prob = float(MODEL.predict_proba(X_trans)[0, 1])
        else:
            prob = float(MODEL.predict(X_trans)[0])

        prob = round(prob, 4)
        is_at_risk = prob >= 0.50

        if prob >= 0.65:
            risk_level = "High Risk"
            predicted_status = "Dropout Likely"
        elif prob >= 0.35:
            risk_level = "Moderate Risk"
            predicted_status = "Academic Struggle / Enrolled"
        else:
            risk_level = "Low Risk"
            predicted_status = "Graduate Likely"

        interventions = generate_interventions(prob, student)
        model_name = METADATA.get("model_name", "Trained_Classifier") if METADATA else "Champion_Model"

        return StudentPredictionResponse(
            student_id=student.student_id or "STU_ANON",
            dropout_probability=prob,
            predicted_status=predicted_status,
            risk_level=risk_level,
            is_at_risk=is_at_risk,
            recommended_interventions=interventions,
            model_version=model_name
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_risk(batch: BatchPredictionRequest):
    predictions = [predict_student_risk(s) for s in batch.students]
    high = sum(1 for p in predictions if p.risk_level == "High Risk")
    mod = sum(1 for p in predictions if p.risk_level == "Moderate Risk")
    low = sum(1 for p in predictions if p.risk_level == "Low Risk")

    return BatchPredictionResponse(
        total_students=len(predictions),
        high_risk_count=high,
        moderate_risk_count=mod,
        low_risk_count=low,
        predictions=predictions
    )

@app.get("/drift/report", tags=["MLOps"])
def get_drift_report():
    report_file = REPORTS_DIR / "drift_monitoring_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Drift report not found. Execute drift monitoring task first.")
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)
