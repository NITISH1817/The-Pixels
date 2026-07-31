# Student Performance Prediction with MLOps

A college portfolio project demonstrating an end-to-end MLOps pipeline for predicting student academic performance (Pass/Fail).

## Project Structure

```
student-mlops/
├── data/
│   ├── raw/          # Original dataset (versioned with DVC)
│   └── processed/    # Cleaned & feature-engineered data
├── src/              # Reusable Python modules
│   ├── preprocess.py # Data cleaning & feature engineering
│   ├── train.py      # Model training with MLflow logging
│   └── evaluate.py   # Evaluation metrics
├── notebooks/        # Exploratory Data Analysis
├── models/           # Saved model artifacts
├── api/              # FastAPI prediction service
│   └── main.py
├── tests/            # pytest test suite
├── monitoring/       # Evidently drift reports + Streamlit dashboard
├── .github/
│   └── workflows/    # GitHub Actions CI/CD
├── dvc.yaml          # DVC pipeline definition
├── requirements.txt  # All dependencies
└── README.md
```

## Phases

| Phase | What's Built |
|-------|-------------|
| 0 | Project setup, virtual env, git init |
| 1 | EDA → Train models → MLflow experiment tracking |
| 2 | DVC data versioning + reproducible pipeline |
| 3 | FastAPI `/predict` endpoint |
| 4 | pytest tests + GitHub Actions CI |
| 5 | Evidently drift detection + Streamlit dashboard |

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd student-mlops

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full DVC pipeline
dvc repro

# 5. Launch the API
uvicorn api.main:app --reload

# 6. Launch MLflow UI
mlflow ui
```

## Dataset

**Students Performance in Exams** — 1,000 students, 8 features.  
Target: Binary Pass/Fail based on average exam score ≥ 50.

## Tech Stack

- **ML:** pandas, scikit-learn, XGBoost
- **Tracking:** MLflow
- **Versioning:** DVC
- **Serving:** FastAPI + Uvicorn
- **Testing:** pytest + GitHub Actions
- **Monitoring:** Evidently AI + Streamlit
