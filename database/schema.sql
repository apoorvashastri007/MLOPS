-- Student Performance and Dropout-Risk Prediction System
-- Relational Star Schema DDL (ANSI SQL - Compatible with PostgreSQL and SQLite)

-- 1. Student Dimension
CREATE TABLE IF NOT EXISTS dim_student (
    student_id VARCHAR(32) PRIMARY KEY,
    gender VARCHAR(10),
    age_at_enrollment INTEGER,
    international VARCHAR(5),
    displaced VARCHAR(5),
    special_needs VARCHAR(5),
    debtor VARCHAR(5),
    tuition_fees_up_to_date VARCHAR(5),
    scholarship_holder VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Course Dimension
CREATE TABLE IF NOT EXISTS dim_course (
    course_id INTEGER PRIMARY KEY,
    course_code VARCHAR(32),
    course_name VARCHAR(128),
    attendance_mode VARCHAR(32)
);

-- 3. Demographics & Socio-Economic Dimension
CREATE TABLE IF NOT EXISTS dim_demographics (
    student_id VARCHAR(32) PRIMARY KEY,
    marital_status VARCHAR(64),
    nationality VARCHAR(64),
    mother_qualification VARCHAR(128),
    father_qualification VARCHAR(128),
    mother_occupation VARCHAR(128),
    father_occupation VARCHAR(128),
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id)
);

-- 4. Macroeconomic Environment Dimension
CREATE TABLE IF NOT EXISTS dim_macro_economy (
    student_id VARCHAR(32) PRIMARY KEY,
    unemployment_rate REAL,
    inflation_rate REAL,
    gdp_growth REAL,
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id)
);

-- 5. Fact Table: Academic Performance, Attendance & Assessments
CREATE TABLE IF NOT EXISTS fact_student_performance (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id VARCHAR(32) NOT NULL,
    course_id INTEGER NOT NULL,
    admission_grade REAL,
    previous_qualification_grade REAL,
    sem1_enrolled INTEGER,
    sem1_evaluations INTEGER,
    sem1_approved INTEGER,
    sem1_grade REAL,
    sem2_enrolled INTEGER,
    sem2_evaluations INTEGER,
    sem2_approved INTEGER,
    sem2_grade REAL,
    attendance_rate REAL,
    internal_marks_avg REAL,
    assignment_completion_rate REAL,
    lms_activity_score REAL,
    target VARCHAR(16) NOT NULL,
    is_dropout INTEGER NOT NULL,
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id),
    FOREIGN KEY (course_id) REFERENCES dim_course(course_id)
);

-- 6. Unified Analytical Data Mart
CREATE TABLE IF NOT EXISTS student_analytical_mart (
    student_id VARCHAR(32) PRIMARY KEY,
    gender VARCHAR(10),
    age_at_enrollment INTEGER,
    debtor VARCHAR(5),
    tuition_fees_up_to_date VARCHAR(5),
    scholarship_holder VARCHAR(5),
    course_name VARCHAR(128),
    attendance_mode VARCHAR(32),
    admission_grade REAL,
    sem1_enrolled INTEGER,
    sem1_approved INTEGER,
    sem1_grade REAL,
    sem2_enrolled INTEGER,
    sem2_approved INTEGER,
    sem2_grade REAL,
    attendance_rate REAL,
    internal_marks_avg REAL,
    assignment_completion_rate REAL,
    lms_activity_score REAL,
    approval_rate_overall REAL,
    grade_progression REAL,
    financial_stress_flag INTEGER,
    target VARCHAR(16),
    is_dropout INTEGER,
    risk_level VARCHAR(16),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
