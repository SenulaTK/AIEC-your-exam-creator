import json
import re
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

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        """Strip markdown code block fences and whitespace from Gemini response text."""
        if not raw_text:
            return "{}"
        clean_text = raw_text.strip()
        if "```" in clean_text:
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
            clean_text = re.sub(r"\s*```$", "", clean_text)
            clean_text = clean_text.strip()
        return clean_text

    def _call_with_fallback(
        self,
        requested_model: str,
        contents: Any,
        config: types.GenerateContentConfig,
    ):
        # Always prioritize requested model, followed by valid production fallbacks.
        model_chain = [requested_model]
        valid_fallbacks = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

        for fallback in valid_fallbacks:
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
                # Retry on rate limits or transient server errors.
                if any(
                    key in err_str.lower()
                    for key in ["503", "unavailable", "capacity", "429", "resource_exhausted"]
                ):
                    last_exception = e
                    continue
                # If a 404 model error occurs, continue to the next valid fallback.
                if "404" in err_str or "NOT_FOUND" in err_str:
                    last_exception = e
                    continue
                raise

        raise last_exception if last_exception else RuntimeError(
            "Failed to generate content with available Gemini models."
        )

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
        cleaned_json = self._clean_json_text(response.text)
        return ExamPaper.model_validate_json(cleaned_json)

    def regenerate_question(self, req: RegenerateQuestionRequest) -> Question:
        prompt = (
            f"You are an expert examiner editing an existing exam paper titled '{req.exam_title}'.\n\n"
            f"Original Question:\n{req.original_question.model_dump_json(indent=2)}\n\n"
            f"Teacher's Refinement Instructions:\n{req.edit_instructions}\n\n"
            "Rewrite and regenerate this single question following all quality guidelines:\n"
            "1. MATHEMATICAL FORMATTING: Wrap math formulas and variables in LaTeX dollar signs ($...$) across all JSON fields.\n"
            "2. WORD SPACING: Keep plain English text outside LaTeX math dollars.\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Question,
            temperature=0.6,
        )
        target_model = req.model_name or settings.DEFAULT_MODEL
        response = self._call_with_fallback(target_model, prompt, config)
        cleaned_json = self._clean_json_text(response.text)
        return Question.model_validate_json(cleaned_json)

    def grade_subjective_batch(
        self,
        exam_title: str,
        subjective_questions: List[Dict[str, Any]],
        model_name: Optional[str] = None,
    ) -> List[GradedQuestion]:
        """Send only subjective questions to Gemini for rubric-based grading."""
        if not subjective_questions:
            return []

        prompt = (
            "You are an impartial academic examiner. Grade the following subjective exam questions "
            "strictly against the provided marking criteria and correct answers.\n\n"
            "Formatting Rules:\n"
            "- Use LaTeX math delimiters ($...$) strictly for mathematical feedback or formulas.\n"
            "- Ensure clean, natural word spacing in all feedback text. Never wrap standard sentences in `$`.\n\n"
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
        cleaned_json = self._clean_json_text(response.text)
        parsed = GradingResponse.model_validate_json(cleaned_json)
        for graded_question in parsed.graded_questions:
            graded_question.graded_by = "gemini"
        return parsed.graded_questions
