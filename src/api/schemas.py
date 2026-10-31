from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class StudentPredictionInput(BaseModel):
    """Schema for individual student academic and demographic profile."""
    student_id: Optional[str] = Field("STU_NEW", description="Student unique identifier")
    gender: str = Field("Male", description="Gender: Male or Female")
    age_at_enrollment: int = Field(20, ge=16, le=80, description="Age when enrolled")
    debtor: str = Field("No", description="Has overdue tuition or library debt: Yes or No")
    tuition_fees_up_to_date: str = Field("Yes", description="Tuition fees current: Yes or No")
    scholarship_holder: str = Field("No", description="Scholarship recipient: Yes or No")
    course_name: str = Field("Nursing", description="Degree program name")
    attendance_mode: str = Field("Daytime", description="Attendance schedule: Daytime or Evening")
    admission_grade: float = Field(125.0, ge=0.0, le=200.0, description="Entrance exam mark (0-200)")
    
    # Semester 1 Performance
    sem1_enrolled: int = Field(6, ge=0, description="Units enrolled in 1st semester")
    sem1_approved: int = Field(5, ge=0, description="Units approved in 1st semester")
    sem1_grade: float = Field(13.5, ge=0.0, le=20.0, description="Average grade in 1st semester (0-20)")
    
    # Semester 2 Performance
    sem2_enrolled: int = Field(6, ge=0, description="Units enrolled in 2nd semester")
    sem2_approved: int = Field(5, ge=0, description="Units approved in 2nd semester")
    sem2_grade: float = Field(13.0, ge=0.0, le=20.0, description="Average grade in 2nd semester (0-20)")
    
    # ERP & LMS Features
    attendance_rate: float = Field(88.0, ge=0.0, le=100.0, description="ERP attendance percentage")
    internal_marks_avg: float = Field(16.0, ge=0.0, le=25.0, description="ERP CIE internal marks average (0-25)")
    assignment_completion_rate: float = Field(85.0, ge=0.0, le=100.0, description="Assignment completion percentage")
    lms_activity_score: float = Field(75.0, ge=0.0, le=100.0, description="Moodle LMS activity engagement score")

class StudentPredictionResponse(BaseModel):
    """Prediction result and recommended academic intervention."""
    student_id: str
    dropout_probability: float
    predicted_status: str
    risk_level: str
    is_at_risk: bool
    recommended_interventions: List[str]
    model_version: str

class BatchPredictionRequest(BaseModel):
    students: List[StudentPredictionInput]

class BatchPredictionResponse(BaseModel):
    total_students: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    predictions: List[StudentPredictionResponse]

class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    framework: str
    champion_test_metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    top_features: List[Dict[str, Any]]
