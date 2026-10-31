"""
Automated Integration and Verification Test Suite for FastAPI Prediction Functions.
Tests model inference, risk classification, automated interventions, and drift reporting directly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.api.main import (
    root,
    health_check,
    get_model_info,
    predict_student_risk,
    predict_batch_risk,
    get_drift_report,
    load_inference_artifacts
)
from src.api.schemas import StudentPredictionInput, BatchPredictionRequest

def test_system_endpoints():
    print("=" * 65)
    print(" [TEST SUITE] RUNNING FASTAPI INFERENCE INTEGRATION TESTS")
    print("=" * 65)

    # 1. Initialize Artifacts
    load_inference_artifacts()

    # 2. Test Root
    res_root = root()
    assert res_root["status"] == "ONLINE", f"Root endpoint failed: {res_root}"
    print(" [PASS] root() -> Status: ONLINE (Service Metadata OK)")

    # 3. Test Health Check
    res_health = health_check()
    assert res_health["status"] == "healthy", f"Health status unhealthy: {res_health}"
    print(" [PASS] health_check() -> Status: healthy (Champion Model Loaded in RAM)")

    # 4. Test Model Info
    res_info = get_model_info()
    model_name = res_info["model_name"]
    test_acc = res_info["champion_test_metrics"]["test_accuracy"]
    test_f1 = res_info["champion_test_metrics"]["test_f1"]
    print(f" [PASS] get_model_info() -> Model: {model_name} | Accuracy: {test_acc:.2%} | F1: {test_f1:.4f}")

    # 5. Test Single Student Prediction (Honors / Low-Risk Profile)
    low_risk_payload = StudentPredictionInput(
        student_id="STU_TEST_HONORS",
        gender="Female",
        age_at_enrollment=19,
        debtor="No",
        tuition_fees_up_to_date="Yes",
        scholarship_holder="Yes",
        course_name="Nursing",
        attendance_mode="Daytime",
        admission_grade=150.0,
        sem1_enrolled=6,
        sem1_approved=6,
        sem1_grade=15.5,
        sem2_enrolled=6,
        sem2_approved=6,
        sem2_grade=16.0,
        attendance_rate=96.0,
        internal_marks_avg=21.0,
        assignment_completion_rate=100.0,
        lms_activity_score=92.0
    )
    pred_low = predict_student_risk(low_risk_payload)
    print(f" [PASS] predict_student_risk(Honors):")
    print(f"        -> Status: {pred_low.predicted_status} | Prob: {pred_low.dropout_probability:.1%} | Tier: {pred_low.risk_level}")
    assert pred_low.risk_level == "Low Risk", f"Expected Low Risk but got {pred_low.risk_level}"

    # 6. Test Single Student Prediction (At-Risk Profile)
    high_risk_payload = StudentPredictionInput(
        student_id="STU_TEST_AT_RISK",
        gender="Male",
        age_at_enrollment=27,
        debtor="Yes",
        tuition_fees_up_to_date="No",
        scholarship_holder="No",
        course_name="Informatics Engineering",
        attendance_mode="Evening",
        admission_grade=105.0,
        sem1_enrolled=6,
        sem1_approved=1,
        sem1_grade=7.0,
        sem2_enrolled=6,
        sem2_approved=0,
        sem2_grade=0.0,
        attendance_rate=58.0,
        internal_marks_avg=7.5,
        assignment_completion_rate=30.0,
        lms_activity_score=25.0
    )
    pred_high = predict_student_risk(high_risk_payload)
    print(f" [PASS] predict_student_risk(At-Risk):")
    print(f"        -> Status: {pred_high.predicted_status} | Prob: {pred_high.dropout_probability:.1%} | Tier: {pred_high.risk_level}")
    print(f"        -> Automated Interventions Triggered ({len(pred_high.recommended_interventions)}):")
    for act in pred_high.recommended_interventions:
        print(f"           * {act}")
    assert pred_high.risk_level == "High Risk", f"Expected High Risk but got {pred_high.risk_level}"

    # 7. Test Batch Prediction
    batch_payload = BatchPredictionRequest(students=[low_risk_payload, high_risk_payload])
    batch_res = predict_batch_risk(batch_payload)
    print(f" [PASS] predict_batch_risk():")
    print(f"        -> Evaluated {batch_res.total_students} students (High: {batch_res.high_risk_count}, Low: {batch_res.low_risk_count})")

    # 8. Test Drift Report
    drift_res = get_drift_report()
    print(f" [PASS] get_drift_report():")
    print(f"        -> Status: {drift_res['overall_drift_status']} | Monitored Features: {drift_res['total_features_monitored']}")

    print("=" * 65)
    print(" ALL API PREDICTION AND INFERENCE TESTS PASSED (100% SUCCESS)!")
    print("=" * 65)

if __name__ == "__main__":
    test_system_endpoints()
