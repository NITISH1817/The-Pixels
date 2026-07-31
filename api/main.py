"""
api/main.py
------------
FastAPI prediction server + HTML Frontend serving for Student Performance Prediction.

Run with:
    uvicorn api.main:app --reload
"""

import os
import sys
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make sure src module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = "models/best_model.pkl"
PREP_PATH  = "models/preprocessor_info.pkl"

app = FastAPI(
    title="Student Performance Prediction API",
    description="MLOps Prediction Service using Random Forest Model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files & templates directory
STATIC_DIR = os.path.join("api", "static")
TEMPLATES_DIR = os.path.join("api", "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global variables for model and metadata
model = None
prep_info = None

def init_artifacts():
    global model, prep_info
    if model is None or prep_info is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH):
            model = joblib.load(MODEL_PATH)
            prep_info = joblib.load(PREP_PATH)

@app.on_event("startup")
def load_model_artifacts():
    init_artifacts()


# ------------------------------------------------------------------
# Pydantic Input Schema for Validation
# ------------------------------------------------------------------
class StudentInput(BaseModel):
    Gender: str
    Age: int = Field(..., ge=15, le=30)
    Study_Hours_per_Week: float = Field(..., ge=0, le=50)
    Attendance_Rate: float = Field(..., ge=0, le=100)
    Past_Exam_Scores: float = Field(..., ge=0, le=100)
    Assignment_Submission_Rate: float = Field(..., ge=0, le=100)
    Quiz_Average: float = Field(..., ge=0, le=100)
    Previous_Failures: int = Field(..., ge=0, le=10)
    Internet_Access: str
    Parents_Support: str
    Parental_Education: str
    Family_Income: str
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Screen_Time: float = Field(..., ge=0, le=24)
    Stress_Level: str
    Motivation_Level: str
    Class_Participation: str
    Extracurricular_Activities: int = Field(..., ge=0, le=10)
    School_Support: str
    Travel_Time: float = Field(..., ge=0, le=180)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def serve_ui(request: Request):
    """Serve the Web Dashboard Frontend UI."""
    init_artifacts()
    metrics = prep_info.get("test_metrics", {}) if prep_info else {}
    return templates.TemplateResponse(request=request, name="index.html", context={"metrics": metrics})


@app.get("/health")
def health_check():
    """Health check endpoint for CI/CD or monitoring."""
    init_artifacts()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": prep_info.get("best_model_name") if prep_info else None
    }


@app.post("/predict")
def predict_performance(data: StudentInput):
    """Predict Pass / Fail for a student."""
    init_artifacts()
    if model is None or prep_info is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        # Convert input Pydantic model to dict -> DataFrame
        input_dict = data.dict()
        df_raw = pd.DataFrame([input_dict])

        # One-hot encode using categorical columns
        cat_cols = prep_info["categorical_cols"]
        df_encoded = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True, dtype=int)

        # Align columns with trained feature columns
        trained_cols = prep_info["feature_columns"]
        for col in trained_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Ensure correct column order
        df_encoded = df_encoded[trained_cols]

        # Make prediction
        pred_class = int(model.predict(df_encoded)[0])
        probabilities = model.predict_proba(df_encoded)[0]
        confidence = float(probabilities[pred_class])

        result_label = "PASS" if pred_class == 1 else "FAIL"

        return {
            "prediction": result_label,
            "prediction_code": pred_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "FAIL": round(float(probabilities[0]) * 100, 2),
                "PASS": round(float(probabilities[1]) * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
