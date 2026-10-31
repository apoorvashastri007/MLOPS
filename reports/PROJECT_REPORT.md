# Comprehensive Project Report: Student Performance and Dropout-Risk Prediction System

**Course:** Data Engineering and MLOps  
**Academic Year:** 2026–2027  
**Project:** End-to-End Academic Analytics & Early Intervention Platform  
**Submission Deadline:** First Week of September 2026  

---

## Executive Summary
Higher education institutions globally face persistent student retention challenges. A significant percentage of undergraduate dropouts occur during or immediately following the first academic year, often due to undetected academic distress, financial burdens, or low course engagement. Traditional institutional reporting relies on retrospective semester transcripts generated long after a student has disengaged.

This project delivers an end-to-end, production-grade **Student Performance and Dropout-Risk Prediction System**. The platform integrates heterogeneous data across academic assessments, institutional Enterprise Resource Planning (ERP) systems (attendance, continuous internal marks), and Moodle Learning Management System (LMS) digital engagement logs. 

Built across two rigorous phases:
1. **Part 1 (Data Engineering):** Automates multi-source ingestion, enforces strict data validation rules, creates an auditable rejected-record error log, transforms raw entities into a normalized/star schema relational warehouse, and builds a multi-view interactive Streamlit analytics dashboard.
2. **Part 2 (MLOps Extension):** Engineers a stratified machine learning pipeline benchmarking Logistic Regression, Random Forest, and XGBoost, tracked end-to-end using MLflow. The champion XGBoost model (88.86% Test Accuracy, 0.8195 F1-Score, 0.9407 ROC-AUC) is packaged with scikit-learn preprocessors, served via a high-performance FastAPI microservice, containerized using Docker, and monitored continuously for feature and concept drift using two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) algorithms.

---

## 1. Problem Understanding and Objectives

### 1.1 Context and Motivation
Student attrition incurs severe institutional, economic, and societal costs. When a student discontinues their studies prematurely, the university loses tuition revenue and instructional efficiency, while the student bears accumulated debt without a credential. Academic administrators and student counselors require an automated, transparent early-warning decision-support tool capable of flagging at-risk individuals during early semester weeks—enabling timely, personalized academic and financial interventions.

### 1.2 Core Project Objectives
- **Multi-Source Ingestion Pipeline:** Unify primary historical academic datasets with simulated ERP attendance records and LMS digital activity logs.
- **Data Quality & Auditability:** Ensure zero manual alterations to analytical data, maintain raw checksums (SHA-256), log failed/rejected records, and produce reproducible ingestion audits.
- **Star Schema Data Warehouse:** Normalize student, course, demographic, and macroeconomic dimensions around a central academic performance fact table and an analytical data mart.
- **Executive & Operational Dashboards:** Provide academic advisors with intuitive interactive views covering attendance correlation, pass/fail distributions, departmental benchmarks, and individual student 360° diagnostic profiles.
- **MLOps Predictive Lifecycle:** Train, evaluate, register, serve, and monitor predictive models capable of estimating dropout probabilities and generating automated, rule-based academic intervention prescriptions.

---

## 2. Data Sources and Ingestion Architecture

### 2.1 Selected Datasets and Provenance
In adherence to assignment regulations, only free, legally accessible, public, or instructor-approved datasets are utilized:
1. **Primary Historical Dataset:** UCI Machine Learning Repository *Predict Students' Dropout and Academic Success* dataset (Real-world records from the Polytechnic Institute of Portalegre, Portugal). It features 4,424 student records across 37 demographic, socio-economic, admission, and curricular attributes.
2. **Institutional ERP Export:** Simulated institutional records capturing lecture/lab attendance percentages (45%–100%), continuous internal assessment (CIE) marks out of 25, and assignment submission rates.
3. **Moodle LMS Telemetry:** Synthetic digital logs capturing login counts, quiz attempts, forum discussion interactions, and a composite activity score (0–100).

### 2.2 Ingestion Implementation & Minimum Expectations
The ingestion engine (`src/ingestion/`) was implemented using Python and Pandas, orchestrated via Apache Airflow-compatible workflows:
- **Raw Copy Preservation:** Pristine raw files are archived directly to `data/raw/` prior to any downstream transformation.
- **Audit Logging & Checksums:** Extraction events record source names, extraction timestamps, execution status, row counts, and cryptographic SHA-256 hashes (`extraction_metadata_*.json`).
- **Repeatability:** Workflows are completely scripted, idempotent, and support scheduled recurring execution without manual dataset manipulation.

---

## 3. ETL, Data Quality Gates, and Storage Layer

### 3.1 Data Staging and Identifier Standardization
Incoming records from distinct sources are matched and joined on a standardized primary key: `student_id` formatted as `STU_10000` through `STU_14423`. Merged records form the initial staging layer (`data/staging/staged_student_data.csv`).

### 3.2 Data Quality Validation Engine
The validation engine (`src/etl/validator.py`) enforces rigorous academic data integrity rules:
- **Boundary Verification:** Checks that admission marks conform to the European qualification range [0, 200], curricular units in Semester 1 and Semester 2 fall within [0, 20], and attendance rates are bounded by [0, 100%].
- **Structural Integrity:** Checks that student identifiers are non-null and targets belong strictly to `['Dropout', 'Graduate', 'Enrolled']`.
- **Deduplication:** Enforces unique student records.
- **Error Auditing:** Records violating any rule are segregated into `data/cleaned/rejected_records.csv`, while passing records populate `data/cleaned/cleaned_students.csv`. A machine-readable `validation_report.json` documents pass rates and rule violation frequencies.

### 3.3 Star Schema Dimensional Modeling
To support analytical and transactional reporting, cleaned data is decomposed into a relational Star Schema implemented in SQL (`database/schema.sql`) and loaded via SQLAlchemy into the local data warehouse (`database/student_warehouse.db` or PostgreSQL):

1. **`dim_student` (Dimension):** Student demographics, gender, age at enrollment, scholarship status, special educational needs, and tuition fee status.
2. **`dim_course` (Dimension):** Course identifier, official degree title (e.g., *Nursing*, *Informatics Engineering*), and attendance mode (*Daytime* vs. *Evening*).
3. **`dim_demographics` (Dimension):** Parental qualifications, occupations, marital status, and nationality.
4. **`dim_macro_economy` (Dimension):** Regional macroeconomic indicators (unemployment rate, inflation rate, GDP growth).
5. **`fact_student_performance` (Fact Table):** Enrolled units, evaluations attempted, units approved, mean grade, attendance rate, CIE marks, LMS activity score, and final academic outcome.
6. **`student_analytical_mart` (Analytical Mart):** A denormalized table featuring engineered features such as overall approval rate (`(sem1_approved + sem2_approved) / total_enrolled`), grade progression (`sem2_grade - sem1_grade`), financial stress indicator, and stratified risk tier.

---

## 4. Analytical Visualization & Streamlit Dashboard

A high-performance, responsive Streamlit dashboard (`dashboard/app.py`) provides academic leadership and faculty advisors with 7 interconnected analytical views:

1. **Executive Analytics & KPI Overview:** High-level institutional metrics (4,424 total students, 32.1% baseline dropout rate, 88.5% average attendance, 12.8/20 average marks) with interactive pie charts and risk stratification distributions.
2. **Attendance vs. Academic Marks Analysis:** Plotly interactive scatter plots mapping attendance percentages against Semester 1 grades with bubble size reflecting LMS engagement. Visualizes the critical 75% attendance threshold and 10.0 passing grade boundary alongside correlation heatmaps.
3. **Department & Semester Performance Breakdown:** Evaluates departmental attrition disparities, showing higher risk in evening management programs and lower risk in professional health sciences. Tracks semester grade progression boxplots across degrees.
4. **High-Risk Student Early Intervention Roster:** Prioritized triage table highlighting students with multiple concurrent risk flags (attendance <70%, tuition debt, low approval rates) with direct CSV export capabilities for academic counselors.
5. **Student 360° Profile & Action Plan:** Individual student diagnostic view featuring a 6-axis radar chart (Attendance, Internal Marks, Sem 1 Grade, Sem 2 Grade, LMS Activity, Course Approval) and automated advisory action items.
6. **Real-Time Risk Predictor & What-If Simulator:** Interactive form where advisors can alter student attributes to simulate outcome probabilities in real time using the champion machine learning model.
7. **MLOps Model Performance & Drift Monitoring:** Real-time tracking of champion model hyperparameters, test metrics, and Kolmogorov-Smirnov / PSI feature drift reports.

---

## 5. MLOps Pipeline Extension & Model Development

### 5.1 Problem Formulation & Prediction Target
The prediction target is formulated as an early academic intervention binary classification problem: `is_dropout` (1 for students who drop out, 0 for students who graduate or maintain active enrollment).

### 5.2 Feature Engineering & Stratified Partitioning
Features are partitioned using `scikit-learn` ColumnTransformers:
- **Numerical Features (14):** Age, admission grade, sem1/sem2 enrolled, approved, and grades, ERP attendance, internal marks, assignment completion, LMS activity score, overall approval rate, and grade progression. Imputed with median and scaled using `StandardScaler`.
- **Categorical Features (6):** Gender, debtor status, tuition fee status, scholarship status, attendance mode, and degree title. Imputed with constant placeholders and encoded via `OneHotEncoder(handle_unknown='ignore')`.
- **Partitioning:** Stratified splitting ensures preserved target distributions: **70% Training (3,096 samples)**, **15% Validation (664 samples)**, and **15% Testing (664 samples)**.

### 5.3 Model Benchmarking & Experimental Results
Three diverse algorithms were trained and benchmarked:

| Model Architecture | Hyperparameters | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | $C=1.0$, L2 Penalty, Balanced Weights | 85.99% | 75.00% | **84.51%** | 0.7947 | 0.9323 |
| **Random Forest** (Ensemble) | $N=150$, Max Depth=12, Split=4 | 87.05% | 77.49% | 84.04% | 0.8063 | 0.9387 |
| **XGBoost Classifier** (Champion) | $N=150$, Depth=5, $\eta=0.08$, Subsample=0.85 | **88.86%** | **85.28%** | 78.87% | **0.8195** | **0.9407** |

### 5.4 Champion Model Selection & Interpretability
**XGBoost** was selected as the champion model based on superior overall Test Accuracy (88.86%), F1-Score (0.8195), and discriminative ROC-AUC (0.9407). Feature importance analysis indicates that the top predictive drivers of student retention are:
1. `approval_rate_overall` (Overall curricular unit approval proportion)
2. `sem2_approved` & `sem1_approved` (Raw number of passed subjects)
3. `attendance_rate` (ERP attendance percentage)
4. `tuition_fees_up_to_date` & `debtor` (Financial liquidity signals)
5. `internal_marks_avg` (Continuous evaluation performance)

---

## 6. Model Tracking, Serving, and Containerization

### 6.1 MLflow Experiment Tracking
Experiments, hyperparameters, confusion matrices, classification metrics, and serialized model binaries were logged to MLflow (`sqlite:///database/mlflow.db`). Champion model artifacts are saved to `models/best_model.joblib` and `models/preprocessor.joblib`, accompanied by a schema card in `models/model_metadata.json`.

### 6.2 FastAPI Inference Microservice
A high-throughput REST API (`src/api/main.py`) provides production endpoints:
- `GET /health`: Service health and model memory residency.
- `GET /model_info`: Model architecture, training hyperparameters, and test metrics.
- `POST /predict`: Single student real-time prediction returning dropout probability, risk tier, and automated academic intervention recommendations.
- `POST /predict_batch`: High-volume cohort batch scoring.
- `GET /drift/report`: Summary of latest data drift health checks.

### 6.3 Containerization Architecture
The application is fully containerized using Docker and Docker Compose (`docker/`):
- `postgres-dw`: PostgreSQL 15 Relational Data Warehouse.
- `prediction-api`: FastAPI REST prediction container.
- `analytics-dashboard`: Streamlit frontend container.

---

## 7. Model Monitoring, Drift Detection, and Retraining Strategy

### 7.1 Statistical Drift Detection
Production models degrade when incoming student cohorts diverge from historical training distributions. The drift monitoring module (`src/mlops/monitoring.py`) runs two statistical tests:
1. **Two-Sample Kolmogorov-Smirnov (KS) Test:** Evaluates continuous distributions (attendance, marks, LMS scores) against baseline training distributions. Rejection threshold: $p < 0.05$.
2. **Population Stability Index (PSI):** Quantifies distributional divergence across binned percentiles. Threshold: $\text{PSI} > 0.25$ indicates significant drift.
3. **Concept / Target Drift:** Tracks shifts in empirical dropout rates between historical cohorts and incoming student batches.

### 7.2 Automated Retraining Policy
Retraining is triggered when:
- Three or more key features exhibit statistically significant drift ($p < 0.05$).
- Target dropout rate deviates by greater than 8.0 percentage points from baseline.
- Production prediction latency exceeds SLA thresholds (>500ms).

---

## 8. Conclusion and Future Directions

The developed **Student Performance and Dropout-Risk Prediction System** demonstrates an end-to-end implementation of both Data Engineering and MLOps best practices. By unifying raw data ingestion, relational star-schema storage, interactive diagnostic dashboards, and reproducible machine learning with automated drift monitoring, the platform equips educational institutions with a reliable tool to proactively improve student retention and academic success.

Future extensions could explore:
- Integrating Large Language Model (LLM) agents for automated student counseling message generation.
- Connecting live Canvas/Blackboard LMS Webhook streams for real-time daily activity ingestion.
- Deploying Kubernetes orchestration with Kubeflow for automated distributed retraining.
