import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from .models.exam import ExamPaper, Question, GradingResponse
from .models.requests import GenerateExamRequest, RegenerateQuestionRequest, GradeExamRequest
from .services.gemini_service import GeminiService
from .services.hybrid_grader import HybridGrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aiec-backend")

app = FastAPI(
    title="AIEC Cloud Run Model Service",
    version="1.0.0",
    description="Production-grade Google Cloud Run service for AI Exam Generation and Hybrid Grading."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gemini_service(
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key")
) -> GeminiService:
    try:
        return GeminiService(api_key=x_gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize GeminiService: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_hybrid_grader(gemini_service: GeminiService = Depends(get_gemini_service)) -> HybridGrader:
    return HybridGrader(gemini_service)

@app.get("/health")
def health():
    return {"status": "ok", "service": "aiec-backend", "version": "1.0.0"}

@app.post("/api/v1/exams/generate", response_model=ExamPaper)
def generate_exam(
    req: GenerateExamRequest,
    gemini_svc: GeminiService = Depends(get_gemini_service)
):
    """Generate a structured exam paper using Gemini."""
    try:
        logger.info(f"Generating exam for subject={req.subject}, topic={req.topic}, questions={req.num_questions}")
        return gemini_svc.generate_exam(req)
    except Exception as e:
        logger.error(f"Exam generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/v1/exams/regenerate-question", response_model=Question)
def regenerate_question(
    req: RegenerateQuestionRequest,
    gemini_svc: GeminiService = Depends(get_gemini_service)
):
    """Regenerate or refine a single question with targeted teacher instructions."""
    try:
        logger.info(f"Regenerating question with instructions: {req.edit_instructions[:60]}...")
        return gemini_svc.regenerate_question(req)
    except Exception as e:
        logger.error(f"Question regeneration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")

@app.post("/api/v1/exams/grade", response_model=GradingResponse)
def grade_exam(
    req: GradeExamRequest,
    grader: HybridGrader = Depends(get_hybrid_grader)
):
    """
    Perform hybrid grading:
    - Objective questions (MCQ, True/False, Matching, Ordering, Calculation) are graded instantly in-memory.
    - Subjective questions (Short Answer, Essay) are batched to Gemini for rubric scoring.
    """
    try:
        logger.info(f"Grading exam: '{req.exam.title}' with {len(req.student_answers)} answers submitted")
        return grader.grade(req.exam, req.student_answers, model_name=req.model_name)
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")
