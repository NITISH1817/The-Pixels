# Student Performance Prediction & MLOps Decision Support System

An end-to-end production MLOps pipeline for predicting student academic performance, evaluating risk scores, and generating actionable educational intervention plans.

![Architecture Flow](https://raw.githubusercontent.com/NITISH1817/The-Pixels/main/api/static/style.css)

## 🌟 Key Features

- **Decision Support System**: Calculates **Risk Score (0-100)**, **Academic Risk Level (Low/Medium/High)**, and generates **Personalized Recommendations** with root-cause risk reasons.
- **Machine Learning Pipeline**: Trained and benchmarked Logistic Regression, Random Forest, and XGBoost using **5-Fold Cross-Validation** & `StandardScaler`.
- **Model Tracking & Registry**: Automatic logging of hyperparameters, metrics, confusion matrix plots, and model versions via **MLflow**.
- **Data Version Control**: Dataset tracking and 1-command reproducible pipeline using **DVC** (`dvc.yaml`).
- **REST API & Web UI**: Fullscreen Light Theme Dashboard built with **FastAPI**, **HTML5/CSS3/JS**, and **Pydantic** input validation.
- **Containerization**: 1-click execution via **Docker** & **Docker Compose**.

---

## 🐳 Running with Docker (Recommended)

### Option A: Using Docker Compose (Starts Web App + MLflow Server together)

```bash
# Build and run container services
docker compose up --build
```

- **Web Application UI**: `http://localhost:8000`
- **MLflow Tracking Dashboard**: `http://localhost:5000`

### Option B: Building & Running Docker Image directly

```bash
# 1. Build Docker image
docker build -t student-mlops-app .

# 2. Run container
docker run -p 8000:8000 student-mlops-app
```

Then open `http://localhost:8000` in your browser.

---

## 🚀 Local Quick Start (Without Docker)

```bash
# 1. Clone repo & navigate
git clone https://github.com/NITISH1817/The-Pixels.git
cd student-mlops

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows (CMD)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Reproduce DVC pipeline & train model
dvc repro

# 5. Launch FastAPI Web Application
uvicorn api.main:app --reload
```

---

## 🧪 Testing

```bash
# Run unit test suite
pytest tests/ -v
```

---

## 🛠️ Project Structure

```
student-mlops/
├── api/
│   ├── main.py             # FastAPI REST Server + Decision Support Engine
│   ├── static/style.css    # Fullscreen Light Theme Stylesheet
│   └── templates/index.html # Web Dashboard UI
├── data/
│   └── raw/                # DVC tracked dataset
├── models/
│   ├── best_model.pkl      # Saved Random Forest Pipeline
│   └── preprocessor_info.pkl # Metadata & feature schema
├── src/
│   ├── data_preprocessing.py # Preprocessing & cleaning module
│   └── train.py            # Training pipeline & MLflow logging
├── tests/                  # Pytest unit tests
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
├── dvc.yaml                # DVC reproducible pipeline
├── mlflow.db               # MLflow SQLite tracking database
└── requirements.txt        # Pinned dependencies
```
