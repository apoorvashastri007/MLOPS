# Student Performance and Dropout-Risk Prediction System
### Data Engineering & MLOps Academic Analytics Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
The **Student Performance and Dropout-Risk Prediction System** is an end-to-end Data Engineering and MLOps platform designed to detect students at academic risk early in their educational journey and prescribe personalized academic and financial interventions.

The project fulfills **100% of Assignment Part 1 (Data Engineering)** and **Assignment Part 2 (MLOps Extension)** requirements.

### Key Capabilities:
- **Multi-Source Ingestion:** Ingests primary historical academic data (UCI dataset, 4,424 records), College ERP continuous evaluation marks and attendance %, and Moodle LMS engagement telemetry with raw archiving and SHA-256 checksums.
- **Data Quality & Validation Gates:** Validates boundary conditions on marks and attendance, generates audit error logs for rejected records, and removes duplicates.
- **Relational Star Schema Warehouse:** Decomposes entities into `dim_student`, `dim_course`, `dim_demographics`, `dim_macro_economy`, `fact_student_performance`, and a unified `student_analytical_mart` loaded into SQLite/PostgreSQL.
- **7-View Streamlit Dashboard:** Features interactive Plotly visualizations, attendance-vs-marks scatter plots, departmental failure analysis, high-risk rosters, 360° student diagnostic radar charts, and real-time what-if risk simulators.
- **MLOps Predictive Models:** Benchmarks **Logistic Regression**, **Random Forest**, and **XGBoost Classifier** with MLflow tracking. The champion XGBoost model achieves **88.86% Test Accuracy, 0.8195 F1-Score, and 0.9407 ROC-AUC**.
- **REST Prediction API:** High-performance FastAPI microservice (`/predict`, `/predict_batch`, `/model_info`, `/health`, `/drift/report`).
- **Drift Monitoring:** Tracks feature drift (Kolmogorov-Smirnov test and PSI) and target concept drift to trigger automated model retraining.
- **Containerization & Orchestration:** Fully reproducible via Docker Compose and Apache Airflow DAGs.

---

## 🏗️ Repository Architecture

```
d:\apoorva shastri cmrit\
├── config/
│   ├── config.py                 # Central configurations, database URLs, paths, thresholds
│   └── logging_config.py         # Standardized logging for pipelines and audit trails
├── data/
│   ├── raw/                      # Raw copies of ingested sources + SHA256 audit metadata
│   ├── staging/                  # Staged parsed records with standardized student IDs
│   ├── cleaned/                  # Validated clean dataset, rejected records & error logs
│   └── analytical/               # Star schema CSVs and student analytical data mart
├── database/
│   ├── schema.sql                # SQL DDL for Star Schema and Analytical Mart
│   ├── db_manager.py             # SQLite and PostgreSQL database connector & loader
│   ├── student_warehouse.db      # Populated SQLite database warehouse (4,424 rows)
│   └── mlflow.db                 # MLflow experiment tracking SQLite database
├── src/
│   ├── ingestion/
│   │   ├── ingest_uci.py         # Ingests primary UCI dataset with SHA256 checksum
│   │   ├── ingest_erp.py         # Ingests College ERP attendance and internal marks
│   │   ├── ingest_lms.py         # Ingests Moodle LMS activity logs
│   │   └── ingestion_pipeline.py # Orchestrates repeatable multi-source ingestion
│   ├── etl/
│   │   ├── validator.py          # Data validation rules, quality checks & error logger
│   │   ├── transformer.py        # Cleans, deduplicates, and engineers analytical features
│   │   └── etl_pipeline.py       # Full ETL execution pipeline raw -> staging -> clean -> DB
│   ├── mlops/
│   │   ├── feature_store.py      # Feature definitions, scaling, and stratified splits
│   │   ├── train.py              # Model training (Logistic Regression, RF, XGBoost) & MLflow
│   │   ├── evaluate.py           # Precision, recall, F1, ROC-AUC, confusion matrices
│   │   └── monitoring.py         # Kolmogorov-Smirnov & PSI drift detection engine
│   └── api/
│       ├── main.py               # FastAPI application with REST inference endpoints
│       └── schemas.py            # Pydantic schemas for request and response validation
├── dags/
│   └── student_data_pipeline_dag.py # Apache Airflow DAG definition for scheduling & orchestration
├── dashboard/
│   └── app.py                    # 7-View Streamlit Analytics Dashboard
├── models/
│   ├── best_model.joblib         # Serialized champion XGBoost classifier
│   ├── preprocessor.joblib       # Serialized scikit-learn preprocessing pipeline
│   └── model_metadata.json       # Champion model metadata card & benchmarks
├── reports/
│   ├── PROJECT_REPORT.md         # Full 8-10 page academic and technical project report
│   ├── DATA_DICTIONARY.md        # Complete data dictionary and validation specification
│   ├── ARCHITECTURE_DESIGN.md    # Multi-layer architecture diagrams and ER diagrams
│   └── drift_monitoring_report.json # Automated drift monitoring audit report
├── docker/
│   ├── Dockerfile.api            # Docker container for FastAPI prediction service
│   ├── Dockerfile.dashboard      # Docker container for Streamlit frontend
│   └── docker-compose.yml        # Multi-container orchestration (FastAPI + Streamlit + Postgres)
├── dvc.yaml                      # DVC pipeline specification for reproducible data versioning
├── run_pipeline.py               # One-click end-to-end pipeline runner
├── test_api.py                   # Automated API integration and inference test suite
├── requirements.txt              # Production dependency specifications
└── README.md                     # Setup, architecture, and execution guide
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python 3.12+** installed on your system.
- Standard virtual environment (`venv`).

### 2. Activate Virtual Environment
```bash
# Windows PowerShell / CMD
.\venv\Scripts\activate
```

### 3. Run the Entire End-to-End Pipeline (One Click)
To run the full multi-source ingestion, ETL, database warehouse population, MLOps model training, champion selection, and drift monitoring in a single command:
```bash
python run_pipeline.py
```
*Expected Execution Time: ~45–50 seconds.*

---

## 🖥️ Running the User Interfaces & Services

### Launch the Streamlit Analytics Dashboard
```bash
streamlit run dashboard/app.py
```
Access the interactive dashboard in your web browser at: **`http://localhost:8501`**

#### Available Views in Dashboard:
1. **Executive Analytics & KPI Overview:** High-level metrics, outcome distributions, and early-warning pie charts.
2. **Attendance vs. Marks Analysis:** Scatter plots showing correlation between attendance rates and semester grades, highlighting the critical 75% attendance threshold.
3. **Department & Semester Performance Breakdown:** Departmental failure comparisons and semester grade progression boxplots.
4. **High-Risk Student Early Intervention Roster:** Prioritized triage list with downloadable CSV export for student counseling teams.
5. **Student 360° Profile & Action Plan:** Individual student diagnostic view featuring a 6-axis radar chart and automated interventions.
6. **Real-Time Risk Predictor & What-If Simulator:** Interactive form where advisors can adjust academic inputs to simulate predicted dropout probability in real time.
7. **MLOps Model Performance & Drift Monitoring:** Real-time metrics on the champion model and live feature drift reports.

---

### Launch the FastAPI Prediction Service
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API Documentation: **`http://127.0.0.1:8000/docs`**
- Alternative ReDoc Documentation: **`http://127.0.0.1:8000/redoc`**

#### API Endpoints:
- `GET /health` - Health status and model readiness.
- `GET /model_info` - Champion model name, hyperparameters, and test metrics.
- `POST /predict` - Real-time risk scoring for an individual student.
- `POST /predict_batch` - Batch scoring for student cohorts.
- `GET /drift/report` - Summary of latest Kolmogorov-Smirnov and PSI drift checks.

---

### Run Automated Integration & Inference Tests
To verify all REST endpoints and model predictions:
```bash
python test_api.py
```
*Outputs: 100% test pass confirmation across single prediction, batch scoring, model info, and drift detection.*

---

## 📊 Model Evaluation Benchmark

Trained and evaluated on a stratified **70% Train (3,096)**, **15% Validation (664)**, and **15% Test (664)** split:

| Model Architecture | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 85.99% | 75.00% | 84.51% | 0.7947 | 0.9323 | Baseline |
| **Random Forest** (Ensemble) | 87.05% | 77.49% | 84.04% | 0.8063 | 0.9387 | Candidate |
| **XGBoost Classifier** | **88.86%** | **85.28%** | **78.87%** | **0.8195** | **0.9407** | **🏆 Champion** |

All runs, parameters, metrics, and artifacts are tracked in MLflow (`database/mlflow.db`).

---

## 🐳 Docker Deployment (Multi-Container Setup)

To spin up the complete containerized stack (PostgreSQL Data Warehouse + FastAPI Prediction API + Streamlit Dashboard):

```bash
docker-compose -f docker/docker-compose.yml up --build
```

- **PostgreSQL Data Warehouse:** `localhost:5432`
- **FastAPI Prediction Microservice:** `http://localhost:8000`
- **Streamlit Analytics Dashboard:** `http://localhost:8501`

---

## 📋 Assignment Submission Package Checklist

| Item | Requirement | File Location / Artifact | Status |
| :--- | :--- | :--- | :--- |
| **1** | Source code and ETL scripts | `src/ingestion/`, `src/etl/` | Completed |
| **2** | Airflow DAG / orchestration | `dags/student_data_pipeline_dag.py` | Completed |
| **3** | Database schema & populated tables | `database/schema.sql`, `database/student_warehouse.db` | Completed (4,424 rows) |
| **4** | Dataset source info & instructions | `reports/DATA_DICTIONARY.md` | Completed |
| **5** | Architecture diagram & pipeline flow | `reports/ARCHITECTURE_DESIGN.md` | Completed |
| **6** | Data dictionary & validation rules | `reports/DATA_DICTIONARY.md` | Completed |
| **7** | Streamlit application | `dashboard/app.py` | Completed (7 Views) |
| **8** | Project report (8-10 pages) | `reports/PROJECT_REPORT.md` | Completed |
| **9** | Execution evidence & test suite | `test_api.py`, `run_pipeline.py` | Completed |
| **10**| README with setup & run instructions| `README.md` | Completed |
| **Part 2**| MLOps: MLflow, FastAPI, Drift, Docker | `src/mlops/`, `src/api/`, `docker/` | Completed |

---

## ⚖️ Academic Integrity & Ethics
- All student identifiers in this system are anonymized (`STU_10000+`).
- No confidential, identifiable personal data is exposed or stored.
- Uses exclusively public and institutional-approved data sources with full attribution.
