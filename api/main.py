"""
api/main.py
------------
FastAPI prediction server + HTML Frontend serving for Student Performance Prediction.
Includes Risk Score Engine, Actionable Recommendations, Batch Prediction, Logging & Health Metrics.

Run with:
    uvicorn api.main:app --reload
"""

import os
import sys
import io
import joblib
import pandas as pd
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make sure src and monitoring modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.logger import log_prediction

MODEL_PATH = "models/best_model.pkl"
PREP_PATH  = "models/preprocessor_info.pkl"
CSV_PATH   = "data/raw/Student Performance Prediction with MLOps - Sheet1.csv"

START_TIME = datetime.now()
PREDICTION_COUNT = 0

app = FastAPI(
    title="Student Performance Prediction & Decision Support System",
    description="MLOps Prediction Service + Educational Decision Support Engine",
    version="2.2.0"
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
def generate_reasons_and_recommendations(data_dict: dict) -> Dict[str, List[str]]:
    reasons = []
    recommendations = []

    att = float(str(data_dict.get("Attendance_Rate", 100)).rstrip("%"))
    sub = float(str(data_dict.get("Assignment_Submission_Rate", 100)).rstrip("%"))
    past = float(data_dict.get("Past_Exam_Scores", 100))
    study = float(data_dict.get("Study_Hours_per_Week", 20))
    screen = float(data_dict.get("Screen_Time", 0))
    fails = int(data_dict.get("Previous_Failures", 0))
    sleep = float(data_dict.get("Sleep_Hours", 8))
    stress = str(data_dict.get("Stress_Level", "Low"))
    motiv = str(data_dict.get("Motivation_Level", "High"))
    quiz = float(data_dict.get("Quiz_Average", 100))

    if att < 85.0:
        reasons.append(f"Low Attendance Rate ({att}%)")
        recommendations.append("Increase classroom attendance to at least 85%")

    if sub < 80.0:
        reasons.append(f"Low Assignment Completion Rate ({sub}%)")
        recommendations.append("Complete and submit all pending assignments on time")

    if past < 65.0:
        reasons.append(f"Low Past Exam Score ({past}/100)")
        recommendations.append("Enroll in subject-specific remedial tutoring sessions")

    if study < 12.0:
        reasons.append(f"Insufficient Weekly Study Time ({study} hrs/week)")
        recommendations.append("Increase self-study time to at least 15 hours per week")

    if screen > 4.0:
        reasons.append(f"Excessive Daily Screen Time ({screen} hrs/day)")
        recommendations.append("Limit non-academic screen time to under 4 hours per day")

    if fails > 0:
        reasons.append(f"History of Class Failures ({fails} previous failure(s))")
        recommendations.append("Schedule 1-on-1 academic counseling with faculty mentor")

    if sleep < 6.0:
        reasons.append(f"Inadequate Sleep ({sleep} hrs/night)")
        recommendations.append("Maintain a healthy sleep schedule of 7 to 8 hours daily")

    if stress == "High":
        reasons.append("High Self-Reported Stress Level")
        recommendations.append("Utilize campus wellness & stress management resources")

    if motiv == "Low":
        reasons.append("Low Academic Motivation Level")
        recommendations.append("Join peer study circles to build motivation & peer accountability")

    if quiz < 65.0:
        reasons.append(f"Low Quiz Average ({quiz}%)")
        recommendations.append("Review weekly quiz feedback and practice mock quizzes")

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
    risk_score = round((1.0 - pass_probability) * 100, 1)

    if risk_score < 30.0:
        risk_level = "Low"
        risk_color = "#10b981"
    elif risk_score <= 60.0:
        risk_level = "Medium"
        risk_color = "#f59e0b"
    else:
        risk_level = "High"
        risk_color = "#ef4444"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color
    }


# ------------------------------------------------------------------
# HELPER: Load Teacher Roster Students
# ------------------------------------------------------------------
def get_teacher_student_roster():
    init_artifacts()
    if not os.path.exists(CSV_PATH) or model is None or prep_info is None:
        return []

    try:
        df = pd.read_csv(CSV_PATH)
        # Parse percentage strings
        for col in ["Attendance_Rate", "Assignment_Submission_Rate"]:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.rstrip("%").astype(float)

        cat_cols = prep_info["categorical_cols"]
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

        trained_cols = prep_info["feature_columns"]
        for col in trained_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[trained_cols]

        predictions = model.predict(df_encoded)
        probabilities = model.predict_proba(df_encoded)

        roster = []
        for idx, row in df.iterrows():
            pred_class = int(predictions[idx])
            pass_prob = float(probabilities[idx][1])
            risk = calculate_risk_metrics(pass_prob)
            student_name = str(row.get("Name", f"Student {idx + 1}"))

            roster.append({
                "id": idx + 1,
                "name": student_name,
                "gender": str(row.get("Gender", "N/A")),
                "age": int(row.get("Age", 0)),
                "attendance_rate": float(row.get("Attendance_Rate", 0)),
                "past_exam_scores": float(row.get("Past_Exam_Scores", 0)),
                "quiz_average": float(row.get("Quiz_Average", 0)),
                "study_hours": float(row.get("Study_Hours_per_Week", 0)),
                "previous_failures": int(row.get("Previous_Failures", 0)),
                "prediction": "PASS" if pred_class == 1 else "FAIL",
                "success_probability": round(pass_prob * 100, 1),
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "risk_color": risk["risk_color"]
            })
        return roster
    except Exception as err:
        print(f"Error loading student roster: {err}")
        return []


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def serve_ui(request: Request):
    """Serve the Dual-Role Web Dashboard Frontend UI."""
    init_artifacts()
    metrics = prep_info.get("test_metrics", {}) if prep_info else {}
    model_name = prep_info.get("best_model_name", "XGBoost") if prep_info else "XGBoost"
    roster = get_teacher_student_roster()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"metrics": metrics, "model_name": model_name, "roster": roster, "total_students": len(roster)}
    )


@app.get("/api/students")
def get_students_api():
    """Return all student records for Teacher Roster view."""
    return get_teacher_student_roster()


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    init_artifacts()
    uptime_seconds = round((datetime.now() - START_TIME).total_seconds(), 2)
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": "v2.0.0",
        "uptime_seconds": uptime_seconds,
        "prediction_count": PREDICTION_COUNT
    }


@app.get("/model-info")
def get_model_info():
    """Returns trained model details, features, and metrics."""
    init_artifacts()
    if not prep_info:
        raise HTTPException(status_code=500, detail="Model info not found.")
    return {
        "model_name": prep_info.get("best_model_name"),
        "model_version": "v2.0.0",
        "feature_count": len(prep_info.get("feature_columns", [])),
        "test_metrics": prep_info.get("test_metrics", {})
    }


@app.get("/metrics")
def get_system_metrics():
    """System prediction metrics for Prometheus or monitoring services."""
    uptime_seconds = round((datetime.now() - START_TIME).total_seconds(), 2)
    return {
        "total_predictions": PREDICTION_COUNT,
        "uptime_seconds": uptime_seconds,
        "status": "active"
    }


@app.post("/predict")
def predict_performance(data: StudentInput):
    """Predict Pass / Fail + Risk Metrics + Actionable Recommendations."""
    global PREDICTION_COUNT
    init_artifacts()
    if model is None or prep_info is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        df_raw = pd.DataFrame([input_dict])

        cat_cols = prep_info["categorical_cols"]
        df_encoded = pd.get_dummies(df_raw, columns=cat_cols, drop_first=False, dtype=int)

        trained_cols = prep_info["feature_columns"]
        for col in trained_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[trained_cols]

        pred_class = int(model.predict(df_encoded)[0])
        probabilities = model.predict_proba(df_encoded)[0]
        pass_probability = float(probabilities[1])

        risk = calculate_risk_metrics(pass_probability)
        advisory = generate_reasons_and_recommendations(input_dict)

        result_label = "PASS" if pred_class == 1 else "FAIL"

        PREDICTION_COUNT += 1

        # Automatically log prediction event
        try:
            log_prediction(
                input_features=input_dict,
                prediction=result_label,
                success_probability=round(pass_probability * 100, 1),
                risk_score=risk["risk_score"],
                risk_level=risk["risk_level"],
                model_version="v2.0.0"
            )
        except Exception as log_err:
            print(f"Logging error: {log_err}")

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


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts CSV file upload, processes batch predictions for multiple students,
    and returns predictions, risk scores, probabilities, and recommendations for every student.
    """
    global PREDICTION_COUNT
    init_artifacts()
    if model is None or prep_info is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df_batch = pd.read_csv(io.BytesIO(contents))

        # Clean percentage columns if present
        for col in ["Attendance_Rate", "Assignment_Submission_Rate"]:
            if col in df_batch.columns and df_batch[col].dtype == object:
                df_batch[col] = df_batch[col].astype(str).str.rstrip("%").astype(float)

        cat_cols = prep_info["categorical_cols"]
        df_encoded = pd.get_dummies(df_batch, columns=cat_cols, drop_first=False, dtype=int)

        trained_cols = prep_info["feature_columns"]
        for col in trained_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[trained_cols]

        predictions = model.predict(df_encoded)
        probabilities = model.predict_proba(df_encoded)

        batch_results = []
        for idx, row_raw in df_batch.iterrows():
            pred_class = int(predictions[idx])
            pass_prob = float(probabilities[idx][1])
            result_label = "PASS" if pred_class == 1 else "FAIL"

            risk = calculate_risk_metrics(pass_prob)
            row_dict = row_raw.to_dict()
            advisory = generate_reasons_and_recommendations(row_dict)

            batch_results.append({
                "student_index": idx + 1,
                "prediction": result_label,
                "success_probability": round(pass_prob * 100, 1),
                "academic_risk_level": risk["risk_level"],
                "risk_score": risk["risk_score"],
                "reasons": advisory["reasons"],
                "recommendations": advisory["recommendations"]
            })

            # Log prediction event
            try:
                log_prediction(
                    input_features=row_dict,
                    prediction=result_label,
                    success_probability=round(pass_prob * 100, 1),
                    risk_score=risk["risk_score"],
                    risk_level=risk["risk_level"],
                    model_version="v2.0.0"
                )
            except Exception:
                pass

        PREDICTION_COUNT += len(batch_results)

        return {
            "total_students_processed": len(batch_results),
            "results": batch_results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")
