import json
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
from ..config import settings
from ..models.exam import ExamPaper, Question, GradedQuestion, GradingResponse
from ..models.requests import GenerateExamRequest, RegenerateQuestionRequest

class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.get_api_key()
        self._client = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def client(self) -> genai.Client:
        if not self._client:
            raise ValueError(
                "Gemini API key is not configured. Set GEMINI_API_KEY environment variable "
                "or configure Google Secret Manager."
            )
        return self._client

    def _call_with_fallback(
        self,
        requested_model: str,
        contents: Any,
        config: types.GenerateContentConfig
    ):
        model_chain = [requested_model]
        for fallback in ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
            if fallback not in model_chain:
                model_chain.append(fallback)
        
        last_exception = None
        for model in model_chain:
            try:
                return self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "capacity" in err_str.lower() or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    last_exception = e
                    continue
                raise e
        raise last_exception if last_exception else RuntimeError("Failed to generate content with available Gemini models.")

    def generate_exam(self, req: GenerateExamRequest) -> ExamPaper:
        prompt = (
            f"You are an expert academic curriculum designer and senior examiner.\n"
            f"Create a rigorous, highly professional exam paper tailored to the following specifications:\n"
            f"- Subject: {req.subject}\n"
            f"- Target Grade / Level: {req.grade_level}\n"
            f"- Topic / Area: {req.topic}\n"
            f"- Number of Questions: {req.num_questions}\n"
            f"- Difficulty Breakdown: {req.difficulty_mix}\n"
            f"- Allowed Question Types: {', '.join(req.question_types)}\n\n"
        )
        if req.syllabus_context:
            prompt += (
                "--- CURRICULUM SYLLABUS / SOURCE MATERIAL ---\n"
                f"{req.syllabus_context}\n"
                "----------------------------------------------\n"
                "Align questions directly to learning objectives and concepts in the source material.\n\n"
            )

        prompt += (
            "Formatting & Quality Rules:\n"
            "1. Allocate marks appropriately (e.g. 1-2 for MCQ/TF, 2-4 for short answer/match/calc, 5-12 for essay).\n"
            "2. For 'mcq', provide 4 clear options and set 'correct_answer' to the exact matching option string.\n"
            "3. For 'true_false', set 'correct_answer' to 'True' or 'False' with a 1-sentence justification if false.\n"
            "4. For 'matching', provide 'left_items' and 'right_items' (shuffled). Set 'correct_answer' to pairs formatted as: Item A -> Match 1; Item B -> Match 2\n"
            "5. For 'ordering', provide 'items' (shuffled). Set 'correct_answer' to the numbered sequential order.\n"
            "6. For 'calculation', provide problem context, state expected units in 'expected_units', and clear step-by-step mark breakdown.\n"
            "7. For 'essay', provide analytical essay prompt with detailed marking rubric criteria in 'correct_answer'.\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExamPaper,
            temperature=0.7,
        )
        target_model = req.model_name or settings.DEFAULT_MODEL
        response = self._call_with_fallback(target_model, prompt, config)
        return ExamPaper.model_validate_json(response.text)

    def regenerate_question(self, req: RegenerateQuestionRequest) -> Question:
        prompt = (
            f"You are an expert examiner editing an existing exam paper titled '{req.exam_title}'.\n\n"
            f"Original Question:\n{req.original_question.model_dump_json(indent=2)}\n\n"
            f"Teacher's Refinement Instructions:\n{req.edit_instructions}\n\n"
            "Rewrite and regenerate this single question following all original quality guidelines."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Question,
            temperature=0.6,
        )
        target_model = req.model_name or settings.DEFAULT_MODEL
        response = self._call_with_fallback(target_model, prompt, config)
        return Question.model_validate_json(response.text)

    def grade_subjective_batch(
        self,
        exam_title: str,
        subjective_questions: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> List[GradedQuestion]:
        """Send only subjective questions (short answers, essays) to Gemini for rubric-based grading."""
        if not subjective_questions:
            return []

        prompt = (
            "You are an impartial academic examiner. Grade the following subjective exam questions "
            "strictly against the provided marking criteria and correct answers.\n\n"
            f"Exam Title: {exam_title}\n\n"
            f"Questions, Criteria, and Student Responses:\n"
            f"{json.dumps(subjective_questions, indent=2)}\n\n"
            "Return a GradedQuestion object for every question index provided, with fair mark allocation and constructive feedback."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GradingResponse,
            temperature=0.2,
        )
        target_model = model_name or settings.DEFAULT_MODEL
        response = self._call_with_fallback(target_model, prompt, config)
        parsed = GradingResponse.model_validate_json(response.text)
        for g in parsed.graded_questions:
            g.graded_by = "gemini"
        return parsed.graded_questions

