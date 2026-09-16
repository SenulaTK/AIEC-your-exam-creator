"""
Client Adapter for AIEC Streamlit Frontend.
Provides a unified interface allowing Streamlit to seamlessly route
exam generation and grading to the Cloud Run backend API.
"""
import os
import requests
from typing import Dict, Any, Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080").rstrip("/")

def is_backend_available() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def generate_exam_api(
    subject: str,
    grade_level: str,
    topic: str,
    num_questions: int,
    difficulty_mix: str,
    question_types: list,
    syllabus_context: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash"
) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/api/v1/exams/generate"
    headers = {}
    if api_key:
        headers["X-Gemini-API-Key"] = api_key

    payload = {
        "subject": subject,
        "grade_level": grade_level,
        "topic": topic,
        "num_questions": num_questions,
        "difficulty_mix": difficulty_mix,
        "question_types": question_types,
        "syllabus_context": syllabus_context,
        "model_name": model_name
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()

def regenerate_question_api(
    original_question: dict,
    edit_instructions: str,
    exam_title: str = "Exam Paper",
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash"
) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/api/v1/exams/regenerate-question"
    headers = {}
    if api_key:
        headers["X-Gemini-API-Key"] = api_key

    payload = {
        "original_question": original_question,
        "edit_instructions": edit_instructions,
        "exam_title": exam_title,
        "model_name": model_name
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()

def grade_exam_api(
    exam: dict,
    student_answers: dict,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash"
) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/api/v1/exams/grade"
    headers = {}
    if api_key:
        headers["X-Gemini-API-Key"] = api_key

    payload = {
        "exam": exam,
        "student_answers": student_answers,
        "model_name": model_name
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()
