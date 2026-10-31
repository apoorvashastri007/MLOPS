# System Architecture & Pipeline Flow Specification

## 1. Architectural Overview

The **Student Performance and Dropout-Risk Prediction System** is structured as an end-to-end, multi-tier data engineering and MLOps platform. The architecture transitions raw institutional data across seven decoupled layers:

1. **Source Layer**: Heterogeneous administrative sources (UCI historical dataset, College ERP exports, Moodle LMS activity logs).
2. **Ingestion Layer**: Python/Pandas extraction connectors with cryptographic SHA-256 auditing and metadata logging.
3. **Raw / Staging Layer**: Raw preservation and student ID standardization across sources (`STU_10000+`).
4. **Validation & Quality Layer**: Business rule engine checking boundary conditions, logging rejected records, and eliminating duplicates.
5. **Transformation & Storage Layer**: Star Schema dimension and fact tables loaded into a PostgreSQL / SQLite Relational Data Warehouse.
6. **Analytics Layer**: Denormalized `student_analytical_mart` serving the interactive Streamlit analytics dashboard.
7. **MLOps Layer**: Stratified train/val/test feature pipeline, MLflow experiment tracking, model registry, FastAPI REST prediction service, and Kolmogorov-Smirnov / PSI drift monitoring.

---

## 2. End-to-End System Architecture Diagram

```mermaid
flowchart TD
    subgraph S1["1. Source Layer"]
        A1["UCI Higher Ed CSV<br/>(Demographics & Marks)"]
        A2["College ERP Database<br/>(Attendance & Internal Marks)"]
        A3["Moodle LMS Logs<br/>(Logins, Quizzes, Activity)"]
    end

    subgraph S2["2. Ingestion & Raw Layer"]
        B1["ingest_uci.py<br/>(Format cleaning & SHA256)"]
        B2["ingest_erp.py<br/>(Attendance & CIE Marks)"]
        B3["ingest_lms.py<br/>(Telemetry & Engagement)"]
        B4[("data/raw/<br/>Preserved Raw Data & Metadata")]
        A1 --> B1 --> B4
        A2 --> B2 --> B4
        A3 --> B3 --> B4
    end

    subgraph S3["3. Staging & Quality Gates"]
        C1["Staging & ID Standardization<br/>(data/staging/staged_student_data.csv)"]
        C2{"Data Quality Engine<br/>validator.py"}
        C3[("data/cleaned/rejected_records.csv<br/>Error Logs")]
        C4[("data/cleaned/cleaned_students.csv<br/>Validated Records")]
        B4 --> C1 --> C2
        C2 -- "Invalid / Out-of-Bounds" --> C3
        C2 -- "Valid Records (100%)" --> C4
    end

    subgraph S4["4. Transformation & Storage Layer"]
        D1["transformer.py<br/>(Feature Engineering & Star Schema)"]
        D2[("Relational Data Warehouse<br/>student_warehouse.db / Postgres")]
        D3["dim_student"]
        D4["dim_course"]
        D5["dim_demographics"]
        D6["dim_macro_economy"]
        D7["fact_student_performance"]
        D8["student_analytical_mart"]
        C4 --> D1
        D1 --> D2
        D2 --> D3
        D2 --> D4
        D2 --> D5
        D2 --> D6
        D2 --> D7
        D2 --> D8
    end

    subgraph S5["5. Analytics & Dashboard Layer"]
        E1["Streamlit Academic Dashboard<br/>dashboard/app.py"]
        D8 --> E1
        E2["View 1: Executive KPI Overview"]
        E3["View 2: Attendance vs Marks Scatter"]
        E4["View 3: Dept & Sem Pass/Fail Distribution"]
        E5["View 4: High-Risk Student Roster"]
        E6["View 5: Student 360 Profile & Plan"]
        E7["View 6: Interactive Risk Simulator"]
        E1 --> E2
        E1 --> E3
        E1 --> E4
        E1 --> E5
        E1 --> E6
        E1 --> E7
    end

    subgraph S6["6. MLOps Pipeline Layer"]
        F1["AcademicFeatureStore<br/>(Stratified Split: 70/15/15)"]
        F2["Model Training & Tuning<br/>(Logistic Regression, RF, XGBoost)"]
        F3[("MLflow Tracking & Registry<br/>database/mlflow.db")]
        F4[("Champion Model & Preprocessor<br/>models/best_model.joblib")]
        F5["FastAPI REST Inference Service<br/>src/api/main.py"]
        F6["Drift Monitoring & Quality Engine<br/>src/mlops/monitoring.py (KS-test & PSI)"]
        D8 --> F1 --> F2
        F2 <--> F3
        F2 --> F4 --> F5
        F4 --> E7
        F5 --> F6
    end

    subgraph S7["7. Orchestration & Containers"]
        G1["Apache Airflow DAG<br/>dags/student_data_pipeline_dag.py"]
        G2["Docker Compose Orchestration<br/>(FastAPI + Streamlit + Postgres)"]
    end
```

---

## 3. Star Schema Relational Entity Relationship Diagram

```mermaid
erDiagram
    dim_student ||--o{ fact_student_performance : "achieves"
    dim_student ||--|| dim_demographics : "has"
    dim_student ||--|| dim_macro_economy : "enrolled_under"
    dim_course ||--o{ fact_student_performance : "registered_in"

    dim_student {
        string student_id PK
        string gender
        int age_at_enrollment
        string international
        string displaced
        string special_needs
        string debtor
        string tuition_fees_up_to_date
        string scholarship_holder
    }

    dim_course {
        int course_id PK
        string course_code
        string course_name
        string attendance_mode
    }

    dim_demographics {
        string student_id PK, FK
        string marital_status
        string nationality
        string mother_qualification
        string father_qualification
        string mother_occupation
        string father_occupation
    }

    dim_macro_economy {
        string student_id PK, FK
        float unemployment_rate
        float inflation_rate
        float gdp_growth
    }

    fact_student_performance {
        int fact_id PK
        string student_id FK
        int course_id FK
        float admission_grade
        float previous_qualification_grade
        int sem1_enrolled
        int sem1_evaluations
        int sem1_approved
        float sem1_grade
        int sem2_enrolled
        int sem2_evaluations
        int sem2_approved
        float sem2_grade
        float attendance_rate
        float internal_marks_avg
        float assignment_completion_rate
        float lms_activity_score
        string target
        int is_dropout
    }
```

---

## 4. Orchestration & Scheduling

Pipeline orchestration is defined in `dags/student_data_pipeline_dag.py`:

```
start_pipeline 
   │
   ├──> ingest_uci_dataset ──┐
   ├──> ingest_erp_records ──┼──> etl_transform_and_load_warehouse ──> train_mlops_models ──> monitor_data_and_concept_drift ──> end_pipeline
   └──> ingest_lms_logs   ───┘
```

The DAG runs weekly on Apache Airflow and also includes an embedded standalone executor (`python dags/student_data_pipeline_dag.py` or `python run_pipeline.py`) ensuring zero-friction local execution across standard workstations.
