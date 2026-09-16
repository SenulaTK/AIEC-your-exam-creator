from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .exam import ExamPaper, Question

class GenerateExamRequest(BaseModel):
    subject: str = Field(..., description="Subject or discipline, e.g. 'Chemistry'")
    grade_level: str = Field(default="Grade 10", description="Grade or target difficulty level")
    topic: str = Field(..., description="Specific syllabus topic or chapter")
    num_questions: int = Field(default=10, ge=1, le=50)
    difficulty_mix: str = Field(default="Balanced (30% Easy, 50% Medium, 20% Hard)")
    question_types: List[str] = Field(
        default=["mcq", "short_answer", "essay", "true_false", "matching", "ordering", "calculation"]
    )
    syllabus_context: Optional[str] = Field(default=None, description="Optional pasted or extracted syllabus text")
    model_name: Optional[str] = Field(default="gemini-3.6-flash")

class RegenerateQuestionRequest(BaseModel):
    original_question: Question
    edit_instructions: str = Field(..., description="Specific instructions for how to rewrite or adjust the question")
    exam_title: Optional[str] = Field(default="Exam Paper")
    model_name: Optional[str] = Field(default="gemini-3.6-flash")

class GradeExamRequest(BaseModel):
    exam: ExamPaper
    student_answers: Dict[str, Any] = Field(..., description="Map of question index string/int to student answer")
    model_name: Optional[str] = Field(default="gemini-3.6-flash")
