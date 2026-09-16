from typing import Dict, Any, List
from ..models.exam import ExamPaper, GradedQuestion, GradingResponse
from .deterministic_grader import is_objective_question, grade_objective_question
from .gemini_service import GeminiService

class HybridGrader:
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service

    def grade(self, exam: ExamPaper, student_answers: Dict[str, Any], model_name: str = None) -> GradingResponse:
        graded_results: List[GradedQuestion] = []
        subjective_queue: List[Dict[str, Any]] = []

        total_possible = sum(q.marks for q in exam.questions)

        for idx, question in enumerate(exam.questions):
            ans = student_answers.get(str(idx), student_answers.get(idx, ""))

            # 1. Deterministic grading for objective questions
            if is_objective_question(question.question_type):
                graded_item = grade_objective_question(question, ans, idx)
                if graded_item is not None:
                    graded_results.append(graded_item)
                    continue

            # 2. Collect subjective questions for batched Gemini grading
            subjective_queue.append({
                "question_index": idx,
                "question_text": question.question_text,
                "question_type": question.question_type,
                "marks_available": question.marks,
                "marking_criteria_or_correct_answer": question.correct_answer,
                "student_answer": ans
            })

        # 3. Call Gemini in a single batch only if subjective questions exist
        if subjective_queue:
            gemini_graded = self.gemini_service.grade_subjective_batch(
                exam_title=exam.title,
                subjective_questions=subjective_queue,
                model_name=model_name
            )
            graded_results.extend(gemini_graded)

        # Sort back into question index order
        graded_results.sort(key=lambda g: g.question_index)
        total_awarded = sum(g.score for g in graded_results)

        return GradingResponse(
            graded_questions=graded_results,
            total_score=total_awarded,
            total_possible=total_possible
        )
