"""
api/main.py
------------
FastAPI prediction server + HTML Frontend serving for Student Performance Prediction.
Includes Risk Score Engine (Option 1) and Personalised Recommendations (Option 2).

Run with:
    uvicorn api.main:app --reload
"""

import os
import sys
import joblib
import pandas as pd
from typing import Optional, List, Dict
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
    title="Student Performance Prediction & Decision Support System",
    description="MLOps Prediction Service + Educational Decision Support Engine",
    version="2.0.0"
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
# HELPER: Generate Reasons & Personalised Actionable Recommendations
# ------------------------------------------------------------------
def generate_reasons_and_recommendations(data: StudentInput) -> Dict[str, List[str]]:
    reasons = []
    recommendations = []

    if data.Attendance_Rate < 85.0:
        reasons.append(f"Low Attendance Rate ({data.Attendance_Rate}%)")
        recommendations.append("Increase classroom attendance to at least 85%")

    if data.Assignment_Submission_Rate < 80.0:
        reasons.append(f"Low Assignment Completion Rate ({data.Assignment_Submission_Rate}%)")
        recommendations.append("Complete and submit all pending assignments on time")

    if data.Past_Exam_Scores < 65.0:
        reasons.append(f"Low Past Exam Score ({data.Past_Exam_Scores}/100)")
        recommendations.append("Enroll in subject-specific remedial tutoring sessions")

    if data.Study_Hours_per_Week < 12.0:
        reasons.append(f"Insufficient Weekly Study Time ({data.Study_Hours_per_Week} hrs/week)")
        recommendations.append("Increase self-study time to at least 15 hours per week")

    if data.Screen_Time > 4.0:
        reasons.append(f"Excessive Daily Screen Time ({data.Screen_Time} hrs/day)")
        recommendations.append("Limit non-academic screen time to under 4 hours per day")

    if data.Previous_Failures > 0:
        reasons.append(f"History of Class Failures ({data.Previous_Failures} previous failure(s))")
        recommendations.append("Schedule 1-on-1 academic counseling with faculty mentor")

    if data.Sleep_Hours < 6.0:
        reasons.append(f"Inadequate Sleep ({data.Sleep_Hours} hrs/night)")
        recommendations.append("Maintain a healthy sleep schedule of 7 to 8 hours daily")

    if data.Stress_Level == "High":
        reasons.append("High Self-Reported Stress Level")
        recommendations.append("Utilize campus wellness & stress management resources")

    if data.Motivation_Level == "Low":
        reasons.append("Low Academic Motivation Level")
        recommendations.append("Join peer study circles to build motivation & peer accountability")

    if data.Quiz_Average < 65.0:
        reasons.append(f"Low Quiz Average ({data.Quiz_Average}%)")
        recommendations.append("Review weekly quiz feedback and practice mock quizzes")

    # Default positive reinforcement if student has good habits
    if not reasons:
        reasons.append("Consistent academic habits & strong class engagement")
        recommendations.append("Maintain current study routine and continue active participation")

    return {
        "reasons": reasons,
        "recommendations": recommendations
    }


# ------------------------------------------------------------------
# HELPER: Calculate Risk Metrics
# ------------------------------------------------------------------
def calculate_risk_metrics(pass_probability: float):
    # Risk Score ranges from 0 (safest) to 100 (highest risk)
    risk_score = round((1.0 - pass_probability) * 100, 1)

    if risk_score < 30.0:
        risk_level = "Low"
        risk_color = "#10b981"  # Green
    elif risk_score <= 60.0:
        risk_level = "Medium"
        risk_color = "#f59e0b"  # Amber/Yellow
    else:
        risk_level = "High"
        risk_color = "#ef4444"  # Red

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color
    }


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
    """Predict Pass / Fail + Risk Metrics + Actionable Recommendations."""
    init_artifacts()
    if model is None or prep_info is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        input_dict = data.dict()
        df_raw = pd.DataFrame([input_dict])

        # One-hot encode categorical fields
        cat_cols = prep_info["categorical_cols"]
        df_encoded = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True, dtype=int)

        # Align with feature columns
        trained_cols = prep_info["feature_columns"]
        for col in trained_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[trained_cols]

        # Make prediction
        pred_class = int(model.predict(df_encoded)[0])
        probabilities = model.predict_proba(df_encoded)[0]
        pass_probability = float(probabilities[1])

        # Compute Option 1: Risk Metrics
        risk = calculate_risk_metrics(pass_probability)

        # Compute Option 2: Reasons & Personalized Recommendations
        advisory = generate_reasons_and_recommendations(data)

        result_label = "PASS" if pred_class == 1 else "FAIL"

        return {
            "prediction": result_label,
            "prediction_code": pred_class,
            "success_probability": round(pass_probability * 100, 1),
            "academic_risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "risk_color": risk["risk_color"],
            "probabilities": {
                "FAIL": round(float(probabilities[0]) * 100, 1),
                "PASS": round(float(probabilities[1]) * 100, 1)
            },
            "reasons": advisory["reasons"],
            "recommendations": advisory["recommendations"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
