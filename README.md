# AI-Powered Student Academic Risk Assessment & Decision Support System with MLOps

An enterprise-grade production MLOps system for predicting student academic outcomes, assessing risk scores (0-100), generating personalized intervention plans, tracking experiments via MLflow, versioning data with DVC, monitoring drift with Evidently AI, and executing automated retraining.

---

## 🌟 Architecture & MLOps Highlights

- **Machine Learning**: Random Forest Classifier with `StandardScaler` & 5-Fold Cross-Validation (`F1-Score: 0.8125`, `Accuracy: 73.40%`).
- **Educational Decision Support Engine**: Calculates **Risk Score (0-100)**, **Academic Risk Level (Low/Medium/High)**, and generates **Actionable Teacher Recommendations** with root-cause reasons.
- **REST Serving API & UI**: Built with **FastAPI** & **Jinja2/CSS3** light theme UI. Supports single predictions, batch CSV upload (`POST /predict-batch`), health metrics (`GET /health`), model info (`GET /model-info`), and system metrics (`GET /metrics`).
- **Prediction Logging**: Every prediction is automatically logged asynchronously to `monitoring/logs/predictions.csv` and `predictions.jsonl`.
- **Experiment Tracking & Model Registry**: Automatic logging of hyperparameters, metrics, confusion matrix plots, and model versioning via **MLflow**.
- **Data Version Control**: Dataset tracking and 1-command reproducible pipeline using **DVC** (`dvc.yaml`).
- **Real-Time Monitoring Dashboard**: Interactive **Streamlit** dashboard (`monitoring/dashboard.py`).
- **Data Drift Detection**: Automated statistical drift checks comparing reference data vs production prediction logs via **Evidently AI** (`monitoring/drift_detection.py`).
- **Automated Retraining Loop**: Champion vs challenger evaluation and automatic model promotion (`monitoring/retrain.py`).
- **CI/CD Automation**: **GitHub Actions** pipeline (`.github/workflows/ci.yml`) enforcing automated testing, import verification, and Docker builds on push.
- **Containerization**: 1-click execution via **Dockerfile** & **docker-compose.yml**.

---

## 🚀 Quick Start Guide

### 1. Local Execution (Without Docker)

```bash
# Activate environment
.venv\Scripts\activate

# Reproduce DVC pipeline
dvc repro

# Run FastAPI Web Application & API Server
uvicorn api.main:app --reload
```
- **Web App Dashboard**: `http://127.0.0.1:8000`
- **Swagger API Docs**: `http://127.0.0.1:8000/docs`

---

### 2. MLflow Experiment Tracking Dashboard

```bash
mlflow ui
```
- **MLflow Dashboard**: `http://127.0.0.1:5000`

---

### 3. Real-Time Streamlit Monitoring Dashboard

```bash
streamlit run monitoring/dashboard.py
```
- **Streamlit Dashboard**: `http://localhost:8501`

---

### 4. Evidently AI Data Drift Detection

```bash
python monitoring/drift_detection.py
```
- Output generated at `monitoring/reports/drift_report.html` and `drift_summary.json`.

---

### 5. Automated Retraining & Model Promotion

```bash
python monitoring/retrain.py
```

---

### 6. Running with Docker Compose

```bash
docker compose up --build
```
- **Web Application**: `http://localhost:8000`
- **MLflow Tracking**: `http://localhost:5000`

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
student-mlops/
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI/CD Pipeline
├── api/
│   ├── main.py                    # FastAPI REST API + Decision Support Engine
│   ├── static/style.css           # Light Theme Fullscreen Stylesheet
│   └── templates/index.html        # Web Dashboard UI
├── data/
│   └── raw/                       # DVC tracked dataset
├── models/
│   ├── best_model.pkl             # Saved Random Forest Pipeline
│   └── preprocessor_info.pkl      # Column schema metadata
├── monitoring/
│   ├── dashboard.py               # Streamlit Monitoring Dashboard
│   ├── drift_detection.py         # Evidently AI Drift Engine
│   ├── logger.py                  # Automatic Prediction Event Logger
│   ├── retrain.py                 # Automated Retraining & Promotion Pipeline
│   ├── logs/                      # Logged prediction CSV & JSONL files
│   └── reports/                   # HTML & JSON Drift Reports
├── src/
│   ├── data_preprocessing.py      # Preprocessing & cleaning module
│   └── train.py                   # Training pipeline & MLflow logging
├── tests/                         # Pytest unit test suite (14 passing tests)
├── Dockerfile                     # Docker container definition
├── docker-compose.yml             # Multi-container service setup
├── dvc.yaml                       # DVC pipeline file
├── dvc.lock                       # DVC pipeline lock file
├── mlflow.db                      # MLflow SQLite tracking database
└── requirements.txt               # Pinned dependencies
```
