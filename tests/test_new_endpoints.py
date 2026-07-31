"""
tests/test_new_endpoints.py
----------------------------
pytest unit tests for /model-info, /metrics, /predict-batch, and monitoring logger.
"""

import io
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_count" in data
    assert data["feature_count"] == 28


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "uptime_seconds" in data


def test_predict_batch_csv_upload():
    csv_content = (
        "Gender,Age,Study_Hours_per_Week,Attendance_Rate,Past_Exam_Scores,Assignment_Submission_Rate,Quiz_Average,Previous_Failures,Internet_Access,Parents_Support,Parental_Education,Family_Income,Sleep_Hours,Screen_Time,Stress_Level,Motivation_Level,Class_Participation,Extracurricular_Activities,School_Support,Travel_Time\n"
        "Female,20,15,95.0,85,98.0,88,0,Yes,High,Bachelor's,Medium,7,3,Medium,High,High,2,High,30\n"
        "Male,22,5,60.0,55,50.0,60,2,Yes,Low,High School,Low,5,6,High,Low,Low,0,Medium,60\n"
    )
    files = {"file": ("test_batch.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict-batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_students_processed"] == 2
    assert len(data["results"]) == 2
    assert "risk_score" in data["results"][0]
    assert "recommendations" in data["results"][0]
