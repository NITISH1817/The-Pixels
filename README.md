# Student Performance Prediction and MLOps Decision Support System

## Executive Summary

The Student Performance Prediction and MLOps Decision Support System is an end-to-end machine learning platform designed to predict academic outcomes, quantify student failure risks, and provide actionable educational interventions. Built upon a 2,015-student record dataset, the system integrates an automated machine learning training pipeline, a dual-role Web Application for teachers and students, an asynchronous telemetry logger, an Evidently AI Data Drift detection module, and an MLflow model registry.

---

## Key Performance Indicators

- **Active Model Architecture**: XGBoost Classifier (Gradient Boosting Trees)
- **Model Test Accuracy**: 89.1%
- **F1-Score**: 0.8268
- **Recall Rate**: 82.6%
- **Cross-Validation**: 5-Fold Stratified Cross-Validation (`f1_macro` optimization)
- **Dataset Volume**: 2,015 Student Records across 20 Demographics, Behavioral, and Academic Attributes

---

## Architectural System Overview

The system architecture consists of five decoupled layers:

1. **Data Ingestion and Feature Pipeline (`src/data_preprocessing.py`)**:
   - Parses numeric percentages, handles missing values, and applies consistent one-hot encoding (`drop_first=False`) aligned against trained feature columns.
   - Normalizes numerical variables via `StandardScaler`.

2. **Model Training and Experimentation (`src/train.py`)**:
   - Evaluates Logistic Regression, Random Forest, and XGBoost Classifiers.
   - Automatically tracks model runs, metrics, and parameters in **MLflow**.
   - Registers the top-performing model (**XGBoost**) as the Production Champion.

3. **FastAPI Web Application and Dual-Role Portal (`api/main.py`)**:
   - **Teacher Roster View**: Provides class-wide analytics (Total Students, Pass/Fail counts, Average Success Probability) and a filterable student roster.
   - **Student Portal View**: Provides an interactive manual entry risk calculator with custom intervention recommendations.

4. **Telemetry and Asynchronous Audit Logging (`monitoring/logger.py`)**:
   - Logs every prediction request asynchronously to CSV (`monitoring/logs/predictions.csv`) and JSON Lines (`monitoring/logs/predictions.jsonl`).

5. **Operational MLOps Control Center (`monitoring/dashboard.py`)**:
   - A Streamlit operational dashboard providing real-time telemetry, Plotly risk distribution charts, Evidently AI statistical data drift reports, and automated retraining triggers.

---

## System Requirements

- Python 3.12+ / 3.13
- Docker Desktop and Docker Compose
- Key Dependencies: `scikit-learn`, `xgboost`, `fastapi`, `streamlit`, `mlflow`, `evidently`, `plotly`, `pandas`, `pytest`

---

## Installation and Execution Guide

### Option 1: Docker Compose Execution (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/NITISH1817/The-Pixels.git
   cd Student-Performance-Prediction-with-MLOps
   ```

2. Build and run all services:
   ```bash
   docker compose up -d --build
   ```

3. Access endpoints:
   - Web Application: `http://localhost:8000`
   - MLflow Registry: `http://localhost:5000`

---

### Option 2: Local Python Execution

1. Initialize virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Train model and log to MLflow:
   ```bash
   python src/train.py
   ```

3. Launch FastAPI server:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

4. Launch Streamlit MLOps Monitoring Dashboard:
   ```bash
   streamlit run monitoring/dashboard.py --server.port 8501
   ```

5. Run test suite:
   ```bash
   pytest tests/ -v
   ```

---

## Continuous Integration and Deployment

The project includes a GitHub Actions CI pipeline (`.github/workflows/ci.yml`) that executes on every push:
- Validates code style and syntax.
- Runs full model training and artifact generation.
- Executes 14 Pytest unit tests covering API endpoints, data preprocessing, and decision support guardrails.
- Validates Docker container builds.
