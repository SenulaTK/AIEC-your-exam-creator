from pydantic import BaseModel, Field
from typing import List, Optional

class Question(BaseModel):
    question_text: str = Field(description="The question text or problem prompt.")
    question_type: str = Field(
        description="One of: 'mcq', 'short_answer', 'essay', 'matching', 'fill_blank', 'true_false', 'ordering', 'categorization', 'labeling', 'calculation'."
    )
    topic: Optional[str] = Field(default=None, description="Topic or subtopic tested, e.g. 'Photosynthesis'.")
    difficulty: Optional[str] = Field(default=None, description="'easy', 'medium', or 'hard'.")
    marks: int = Field(default=2, description="Marks allocated for this question.")
    
    options: Optional[List[str]] = Field(default=None, description="List of options for MCQ.")
    left_items: Optional[List[str]] = Field(default=None, description="Left column items for matching.")
    right_items: Optional[List[str]] = Field(default=None, description="Right column items for matching (shuffled).")
    items: Optional[List[str]] = Field(default=None, description="List of items for ordering/sequencing or categorizing.")
    categories: Optional[List[str]] = Field(default=None, description="List of category names for categorization questions.")
    label_prompts: Optional[List[str]] = Field(default=None, description="List of label prompts/keys for labeling questions.")
    expected_units: Optional[str] = Field(default=None, description="Expected unit of measurement for calculation questions.")
    
    correct_answer: str = Field(
        description="Correct answer string, key mapping, or marking criteria breakdown."
    )

class ExamPaper(BaseModel):
    title: str = Field(description="The title of the exam paper.")
    instructions: str = Field(description="Any general instructions for the student.")
    questions: List[Question]

class GradedQuestion(BaseModel):
    question_index: int = Field(description="The index of the question in the original list.")
    score: int = Field(description="The marks awarded to the student.")
    feedback: str = Field(description="Constructive feedback explaining the score based on the marking criteria.")
    graded_by: Optional[str] = Field(default="deterministic", description="'deterministic' or 'gemini'")

class GradingResponse(BaseModel):
    graded_questions: List[GradedQuestion]
    total_score: Optional[int] = Field(default=None, description="Calculated sum of all marks awarded.")
    total_possible: Optional[int] = Field(default=None, description="Total possible marks for the exam.")
