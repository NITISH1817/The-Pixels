# Student Performance Prediction & MLOps Decision Support System 🎓🚀

![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=for-the-badge&logo=docker)
![Render](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge)

An end-to-end Machine Learning Operations (MLOps) platform and educational decision support system designed to predict academic performance, evaluate student failure risks, generate personalized educational interventions, track model metrics in real-time, and detect statistical data drift.

---

## 🔗 Quick Links & Live Application

- 🌐 **Live Web Application (Render)**: [https://student-performance-prediction-with-mlops.onrender.com/](https://student-performance-prediction-with-mlops.onrender.com/)
- 📁 **SRS & Technical Report**: [Google Drive Folder](https://drive.google.com/drive/folders/1fpKmCC49v9iGz3c8f8x4KvDeHdSiM3Qx?usp=drive_link)
- 👥 **Team Details**: [TEAM.md](file:///r:/AI-hack/student-mlops/TEAM.md)
- 🐙 **GitHub Repository**: [https://github.com/NITISH1817/The-Pixels](https://github.com/NITISH1817/The-Pixels)

---

## 👨‍💻 Project Team - The Pixels

| Role | Name | Register / Roll Number |
| --- | --- | --- |
| **Team Lead** | NITISH PRIYAN R S | `7376241CS311` |
| **Member 2** | NISHAL A T | `7376241CS301` |
| **Member 3** | HARISH P | `7376241CS204` |
| **Member 4** | VENMUGIL RAJAN S | `7376241CS456` |

---

## 🎯 Executive Summary

Educational institutions face challenges in identifying at-risk students before final assessments occur. This platform bridges machine learning research and operational decision-making by:
1. **Predicting Academic Outcomes**: Classifying student performance into Pass/Fail categories based on 20 demographic, behavioral, and academic indicators.
2. **Quantifying Risk Scores**: Providing percentage-based failure probability scores along with risk category classifications (Low, Medium, High Risk).
3. **Educational Intervention Engine**: Automatically generating custom academic recommendations based on specific student feature values (e.g., low attendance, high absences, low parental support).
4. **Dual-Role Web Portal**: Serving both **Teachers** (class roster analytics & batch CSV predictions) and **Students** (interactive risk calculator).
5. **Continuous MLOps Infrastructure**: Integrating MLflow model registry, DVC data versioning, asynchronous telemetry logging, and Evidently AI data drift detection.

---

## 📊 Key Performance Indicators (KPIs)

- **Champion Model Architecture**: XGBoost Classifier (Gradient Boosted Decision Trees)
- **Model Test Accuracy**: **89.1%**
- **F1-Score (Macro)**: **0.8268**
- **Recall Rate (At-Risk Identification)**: **82.6%**
- **Validation Protocol**: 5-Fold Stratified Cross-Validation (`f1_macro` optimization)
- **Dataset Volume**: 2,015 Student Records across 20 Demographic, Behavioral, and Academic Attributes

---

## ⭐ Key Features

### 1. Dual-Role Web Portal (FastAPI + Jinja2)
- **Teacher Dashboard**:
  - Class-wide metric cards: Total Students, Pass/Fail Counts, Average Success Probability.
  - Interactive, searchable, and filterable student roster.
  - Batch CSV Upload: Process full class CSV files instantly to receive predictions, risk scores, and actionable recommendations.
- **Student Calculator**:
  - Interactive form for real-time risk assessment.
  - Custom recommendations tailored to attendance rates, study hours, tutoring, and parental involvement.

### 2. Decision Support & Recommendation Engine
- Analyzes individual student risk vectors to recommend targeted actions:
  - **Attendance Alert**: Triggers mandatory attendance support if attendance is below 75%.
  - **Study Hours Plan**: Advises structured study schedules if weekly study hours fall below 10 hours.
  - **Tutoring Counseling**: Suggests academic peer tutoring if tutoring is absent.
  - **Parental Engagement**: Recommends parent-teacher conferences if parental support is low.

### 3. Asynchronous Telemetry & Audit Logger
- Every prediction request is logged asynchronously without blocking response threads:
  - CSV Log: `monitoring/logs/predictions.csv`
  - JSON Lines Log: `monitoring/logs/predictions.jsonl`
- Telemetry captures timestamp, input features, predicted outcome, probability score, risk level, and execution latency.

### 4. Operational MLOps & Data Drift Dashboard (Streamlit)
- Real-time telemetry visualization using Plotly charts.
- **Evidently AI Statistical Drift Detection**: Runs Wasserstein Distance & Kolmogorov-Smirnov tests comparing production telemetry against baseline training data.
- **Automated Model Retraining**: Retraining triggers accessible directly from the monitoring dashboard.

---

## 🏗️ System Architecture & Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                          MLOps Pipeline Layer                          │
│                                                                        │
│   [Raw Dataset] ──► [Preprocessing] ──► [MLflow Experiments]          │
│   (2,015 Records)   (StandardScaler &    (LogisticReg, RandomForest,  │
│                      OneHotEncoder)    XGBoost Champion)               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Saves Model & Artifacts
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Production Serving Layer                        │
│                                                                        │
│                   ┌──────────────────────────────┐                     │
│                   │ FastAPI Application (main.py)│                     │
│                   └──────────────┬───────────────┘                     │
│                                  │                                     │
│        ┌─────────────────────────┴─────────────────────────┐           │
│        ▼                                                   ▼           │
│  [Teacher Roster & Batch API]                    [Student Risk Engine] │
│  - Class Analytics Summary                       - Interactive Predict │
│  - Batch CSV Upload                              - Actionable Rules    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Asynchronous Telemetry
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Monitoring & Data Drift Layer                        │
│                                                                        │
│   [Telemetry CSV/JSONL Logs] ──► [Streamlit Operational Dashboard]    │
│                                  - Real-time Plotly Analytics          │
│                                  - Evidently AI Data Drift Reports     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Project Structure

```
Student-Performance-Prediction-with-MLOps/
├── .github/
│   └── workflows/
│       └── ci.yml              # Continuous Integration workflow (Pytest, Lint, Build)
├── api/
│   ├── main.py                 # FastAPI prediction server & HTML portal routes
│   ├── static/
│   │   └── style.css           # Premium glassmorphism UI styling
│   └── templates/
│       └── index.html          # Dual-role Web Application frontend (Teacher & Student)
├── data/
│   ├── processed/              # Processed train/test data splits
│   └── raw/                    # 2,015 Student records raw dataset
├── models/
│   ├── best_model.pkl          # Trained XGBoost Production Champion model
│   └── preprocessor_info.pkl   # Fitted StandardScaler and OneHotEncoder metadata
├── monitoring/
│   ├── dashboard.py            # Streamlit MLOps Monitoring Dashboard
│   ├── drift_detection.py      # Evidently AI statistical drift reporting module
│   ├── logger.py               # Asynchronous CSV and JSONL telemetry audit logger
│   └── logs/                   # Production prediction telemetry logs
├── src/
│   ├── data_preprocessing.py   # Data parsing, missing value handling, feature scaling
│   ├── predict.py              # Single and batch inference utility functions
│   └── train.py                # MLflow model training, 5-fold CV, and experiment tracking
├── tests/
│   ├── test_api.py             # API endpoint unit tests
│   ├── test_data.py            # Data preprocessing unit tests
│   ├── test_model.py           # Model loading & prediction unit tests
│   └── test_new_endpoints.py   # Health, logs, roster, and batch API unit tests
├── .dockerignore
├── docker-compose.yml          # Multi-container orchestration (FastAPI + MLflow + Streamlit)
├── Dockerfile                  # Container build instructions for production deployment
├── dvc.yaml                    # Data Version Control pipeline stages
├── requirements.txt            # Python project dependencies
├── README.md                   # System documentation
├── TEAM.md                     # Team members & register numbers
└── verify_model.py             # Pre-flight verification script for model integrity
```

---

## 🔌 API Endpoints Specification

| Method | Endpoint | Description | Input / Parameters | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/` | Dual-Role Web Portal Frontend | None | Rendered HTML Template (`index.html`) |
| `POST` | `/predict` | Single Student Prediction & Risk Calculation | JSON payload with 20 features | Risk level, probability score, recommendations |
| `POST` | `/predict-batch` | Batch CSV Prediction | Multipart CSV File Upload | Processed prediction rows with risk scores |
| `GET` | `/roster` | Class-wide Student Roster & Summary | None | Class statistics & student list JSON |
| `GET` | `/health` | System Health & Uptime Metrics | None | API status, uptime, total predictions count |
| `GET` | `/logs` | Recent Telemetry Audit Logs | Optional `limit` parameter (default: 50) | Last `N` logged prediction events |

---

## 📝 Dataset Features (20 Attributes)

| Attribute Category | Features | Description / Encoding |
| --- | --- | --- |
| **Demographic** | `Age`, `Gender`, `Ethnicity`, `ParentalEducation` | Student background metadata |
| **Academic** | `GPA`, `Absences`, `Tutoring`, `ParentalSupport` | Study background and support systems |
| **Behavioral** | `StudyTimeWeekly`, `Extracurricular`, `Sports`, `Music` | Student engagement and activity levels |
| **Target Variable** | `GradeClass` / `Pass/Fail` | Outcome target (0 = High Risk / Fail, 1 = Pass) |

---

## 🛠️ Installation and Execution Guide

### Prerequisites
- **Python**: 3.12 or 3.13
- **Docker Desktop** (optional, for containerized run)
- **Git**

---

### Option 1: Docker Compose Execution (Recommended)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NITISH1817/The-Pixels.git
   cd Student-Performance-Prediction-with-MLOps
   ```

2. **Launch Container Services**:
   ```bash
   docker compose up -d --build
   ```

3. **Access Services**:
   - **FastAPI Web Portal**: [http://localhost:8000](http://localhost:8000)
   - **Streamlit MLOps Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)

---

### Option 2: Local Python Execution

1. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Model Training Pipeline**:
   ```bash
   python src/train.py
   ```

4. **Launch FastAPI Web Server**:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   Open `http://localhost:8000` in your browser.

5. **Launch Streamlit Monitoring Dashboard**:
   ```bash
   streamlit run monitoring/dashboard.py --server.port 8501
   ```

---

## 🧪 Testing & Quality Assurance

The project includes an extensive test suite covering API endpoints, data preprocessing pipelines, model serving guardrails, and telemetry logging:

```bash
pytest tests/ -v
```

**Test Coverage Highlights**:
- Data validation and percentage string parsing.
- Model artifact loading & risk score computation.
- Single and batch CSV prediction endpoints.
- System health metrics and log retrieval endpoints.

---

## 🔄 Continuous Integration (CI/CD)

Automated by GitHub Actions (`.github/workflows/ci.yml`):
- Runs on every `push` and `pull_request` to `main`.
- **Code Linting & Formatting**: Checks Python code structure.
- **Automated Training & Verification**: Runs training script to ensure model artifacts generate cleanly.
- **Automated Test Suite**: Executes all Pytest unit tests.
- **Container Build Verification**: Ensures Docker image builds without errors.

---

## 📄 License & Team

Developed by **Team The Pixels** for the Student Performance Prediction and MLOps Decision Support initiative. All rights reserved.
