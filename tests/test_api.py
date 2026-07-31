"""
tests/test_api.py
------------------
pytest unit tests for FastAPI prediction endpoints.

Run with:
    pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["model_loaded"] is True


def test_predict_pass_student():
    payload = {
        "Gender": "Female",
        "Age": 20,
        "Study_Hours_per_Week": 25.0,
        "Attendance_Rate": 98.0,
        "Past_Exam_Scores": 90.0,
        "Assignment_Submission_Rate": 95.0,
        "Quiz_Average": 92.0,
        "Previous_Failures": 0,
        "Internet_Access": "Yes",
        "Parents_Support": "High",
        "Parental_Education": "Master's",
        "Family_Income": "High",
        "Sleep_Hours": 8.0,
        "Screen_Time": 2.0,
        "Stress_Level": "Low",
        "Motivation_Level": "High",
        "Class_Participation": "High",
        "Extracurricular_Activities": 3,
        "School_Support": "High",
        "Travel_Time": 15.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["PASS", "FAIL"]
    assert "confidence" in data
    assert "probabilities" in data
