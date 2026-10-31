# Academic Data Dictionary & Validation Specification

## 1. Project Overview
The **Student Performance and Dropout-Risk Prediction System** synthesizes student demographic, academic, behavioral, and macroeconomic features across historical and real-time administrative systems (UCI Student Success dataset, College ERP exports, and Moodle LMS logs).

---

## 2. Ingestion & Raw Data Sources

| Source | Platform / Origin | File Format | Extraction Tool | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Dataset** | UCI Machine Learning Repository | CSV (Semicolon Delimited) | Python / Pandas | Historical student demographic, admission, and curricular unit records. |
| **Institutional ERP** | College Administrative ERP | CSV / SQL | Pandas / DB Connector | Continuous internal evaluation (CIE) marks, classroom attendance percentages, assignment counts. |
| **Learning Management**| Moodle LMS v4.2 | CSV / REST API | Python / Pandas | Portal login events, quiz attempts, forum discussion posts, and digital activity score. |

---

## 3. Relational Star Schema Data Dictionary

### Table 1: `dim_student` (Dimension Table)
*Grain: One record per enrolled student.*

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | VARCHAR(32) | PRIMARY KEY | Standardized institutional identifier (e.g., `STU_10001`). |
| `gender` | VARCHAR(10) | NOT NULL | Student gender (`Male` or `Female`). |
| `age_at_enrollment` | INTEGER | CHECK (16-80) | Age of student at time of degree enrollment. |
| `international` | VARCHAR(5) | NOT NULL | International student indicator (`Yes` or `No`). |
| `displaced` | VARCHAR(5) | NOT NULL | Whether the student relocated to attend university (`Yes` or `No`). |
| `special_needs` | VARCHAR(5) | NOT NULL | Documented educational special needs (`Yes` or `No`). |
| `debtor` | VARCHAR(5) | NOT NULL | Unresolved institutional/library fees (`Yes` or `No`). |
| `tuition_fees_up_to_date`| VARCHAR(5) | NOT NULL | Current tuition payment status (`Yes` or `No`). |
| `scholarship_holder` | VARCHAR(5) | NOT NULL | State or institutional scholarship recipient (`Yes` or `No`). |

---

### Table 2: `dim_course` (Dimension Table)
*Grain: One record per academic program.*

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `course_id` | INTEGER | PRIMARY KEY | Institutional degree identifier code. |
| `course_code` | VARCHAR(32) | NOT NULL | Official course catalog code. |
| `course_name` | VARCHAR(128)| NOT NULL | Full degree title (e.g., `Nursing`, `Informatics Engineering`, `Management`). |
| `attendance_mode` | VARCHAR(32) | NOT NULL | Academic timetable (`Daytime` or `Evening`). |

---

### Table 3: `dim_demographics` (Dimension Table)
*Grain: One record per student.*

| Column | Type | Description |
| :--- | :--- | :--- |
| `student_id` | VARCHAR(32) | Foreign Key referencing `dim_student(student_id)`. |
| `marital_status` | VARCHAR(64) | Marital status category (Single, Married, Divorced, etc.). |
| `nationality` | VARCHAR(64) | Nationality country code. |
| `mother_qualification` | VARCHAR(128)| Highest educational level achieved by mother. |
| `father_qualification` | VARCHAR(128)| Highest educational level achieved by father. |
| `mother_occupation` | VARCHAR(128)| Professional occupation code of mother. |
| `father_occupation` | VARCHAR(128)| Professional occupation code of father. |

---

### Table 4: `dim_macro_economy` (Dimension Table)
*Grain: One record per cohort enrollment year.*

| Column | Type | Description |
| :--- | :--- | :--- |
| `student_id` | VARCHAR(32) | Foreign Key referencing `dim_student(student_id)`. |
| `unemployment_rate` | REAL | Regional unemployment percentage at enrollment. |
| `inflation_rate` | REAL | National annual consumer inflation rate. |
| `gdp_growth` | REAL | National gross domestic product (GDP) percentage change. |

---

### Table 5: `fact_student_performance` (Fact Table)
*Grain: Semester performance per student per enrolled program.*

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `fact_id` | INTEGER | PRIMARY KEY | Auto-incrementing surrogate primary key. |
| `student_id` | VARCHAR(32) | FK | References `dim_student(student_id)`. |
| `course_id` | INTEGER | FK | References `dim_course(course_id)`. |
| `admission_grade` | REAL | CHECK (0-200) | Secondary school entrance qualification mark. |
| `previous_qualification_grade` | REAL | CHECK (0-200) | Grade achieved in preceding diploma/certificate. |
| `sem1_enrolled` | INTEGER | >= 0 | Number of curricular units enrolled in 1st semester. |
| `sem1_evaluations` | INTEGER | >= 0 | Unit exam/project evaluations attempted in 1st semester. |
| `sem1_approved` | INTEGER | >= 0 | Curricular units successfully cleared in 1st semester. |
| `sem1_grade` | REAL | CHECK (0-20) | Weighted mean grade across 1st semester units. |
| `sem2_enrolled` | INTEGER | >= 0 | Number of curricular units enrolled in 2nd semester. |
| `sem2_evaluations` | INTEGER | >= 0 | Unit evaluations attempted in 2nd semester. |
| `sem2_approved` | INTEGER | >= 0 | Curricular units successfully cleared in 2nd semester. |
| `sem2_grade` | REAL | CHECK (0-20) | Weighted mean grade across 2nd semester units. |
| `attendance_rate` | REAL | CHECK (0-100) | Continuous lecture and lab attendance percentage (ERP). |
| `internal_marks_avg` | REAL | CHECK (0-25) | Continuous internal evaluation (CIE) score average out of 25. |
| `assignment_completion_rate` | REAL | CHECK (0-100) | Timely assignment submission rate percentage. |
| `lms_activity_score` | REAL | CHECK (0-100) | Telemetry engagement score derived from Moodle activity. |
| `target` | VARCHAR(16) | NOT NULL | Academic status: `Dropout`, `Enrolled`, `Graduate`. |
| `is_dropout` | INTEGER | CHECK (0 or 1)| Binary classification target (1 = Dropout, 0 = Non-dropout). |

---

### Table 6: `student_analytical_mart` (Unified Analytical Data Mart)
*Grain: Comprehensive student profile joining dimensions, facts, and engineered features.*

| Feature Name | Type | Derived Formula / Source | Purpose |
| :--- | :--- | :--- | :--- |
| `approval_rate_overall` | REAL | `(sem1_approved + sem2_approved) / total_enrolled` | Core metric of academic progress. |
| `grade_progression` | REAL | `sem2_grade - sem1_grade` | Trajectory indicator (positive = improving, negative = decline). |
| `financial_stress_flag`| INTEGER | `1 if (debtor == 'Yes' or tuition_fees == 'No') else 0` | Captures acute socio-economic vulnerability. |
| `risk_level` | VARCHAR(16)| Multi-factorial risk decision tree | Stratified early intervention tier (`High Risk`, `Moderate Risk`, `Low Risk`). |

---

## 4. Data Quality Validation Rules & Gates

| Rule ID | Field | Condition | Action on Failure |
| :--- | :--- | :--- | :--- |
| `VAL-01` | `student_id` | Must be non-null and format matches `STU_\d+` | Reject record to error log |
| `VAL-02` | `Age at enrollment`| Value must lie in range `[16, 80]` | Reject record to error log |
| `VAL-03` | `Admission grade` | Value must lie in range `[0.0, 200.0]` | Reject record to error log |
| `VAL-04` | Semester Grades | `sem1_grade` and `sem2_grade` in `[0.0, 20.0]` | Reject record to error log |
| `VAL-05` | `attendance_rate`| Value must lie in range `[0.0, 100.0]` | Reject record to error log |
| `VAL-06` | `Target` | Must be one of `['Dropout', 'Graduate', 'Enrolled']` | Reject record to error log |
| `VAL-07` | Uniqueness | No duplicate `student_id` allowed | Deduplicate; keep earliest record |
