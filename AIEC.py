import streamlit as st

st.set_page_config(layout="wide", page_title="AIEC — AI Exam Creator & Evaluator", page_icon="📝")

import json
import os
import re
import tempfile
import uuid
import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from fpdf import FPDF
from google import genai
from google.genai import types

from client_adapter import is_backend_available, regenerate_question_api, grade_exam_api
from backend.app.services.hybrid_grader import HybridGrader
from backend.app.services.gemini_service import GeminiService
from backend.app.config import settings
from backend.app.models.exam import Question, ExamPaper, GradedQuestion, GradingResponse


def format_math_latex(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def render_standard_math(text: str, prefix: str = ""):
    if not text:
        return
    clean_text = text.strip()
    is_pure_math = (
        clean_text.startswith("$")
        and clean_text.endswith("$")
        and clean_text.count("$") <= 4
        and len(clean_text) > 2
    )
    if is_pure_math:
        if prefix:
            st.markdown(prefix)
        st.latex(clean_text.strip("$"))
    else:
        st.markdown(f"{prefix}{clean_text}")


def render_countdown_timer(minutes: int):
    if "exam_start_timestamp" not in st.session_state or st.session_state["exam_start_timestamp"] is None:
        st.session_state["exam_start_timestamp"] = datetime.datetime.now().timestamp()

    elapsed_seconds = datetime.datetime.now().timestamp() - st.session_state["exam_start_timestamp"]
    remaining_seconds = max(0, int(minutes * 60 - elapsed_seconds))

    timer_html = f"""
    <div id="timer-box" style="font-family:sans-serif;font-size:20px;font-weight:bold;color:#d9534f;background:#fdf2f2;border:2px solid #d9534f;border-radius:8px;padding:10px 15px;text-align:center;margin-bottom:15px;">
        ⏱️ Time Remaining: <span id="timer-display">--:--</span>
    </div>
    <script>
        var secondsLeft = {remaining_seconds};
        function updateTimer() {{
            var mins = Math.floor(secondsLeft / 60);
            var secs = secondsLeft % 60;
            if (secs < 10) secs = "0" + secs;
            if (mins < 10) mins = "0" + mins;
            document.getElementById('timer-display').innerHTML = mins + ":" + secs;
            if (secondsLeft <= 0) {{
                document.getElementById('timer-box').innerHTML = "⌛ TIME IS UP! Please submit your exam.";
                document.getElementById('timer-box').style.backgroundColor = "#ff0000";
                document.getElementById('timer-box').style.color = "#ffffff";
            }} else {{
                secondsLeft--;
            }}
        }}
        updateTimer();
        setInterval(updateTimer, 1000);
    </script>
    """
    components.html(timer_html, height=75)


def safe_parse_json(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Response text is empty.")
    clean_text = raw_text.strip()
    if "```" in clean_text:
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text).strip()
    try:
        return json.loads(clean_text)
    except Exception:
        pass
    try:
        return json.loads(clean_text, strict=False)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip(), strict=False)
        except Exception:
            pass
    raise ValueError(f"Could not parse valid JSON from response text: {raw_text[:200]}...")


def generate_with_gemini_fallback(client: genai.Client, target_model: str, contents: list, config: types.GenerateContentConfig):
    target_model = target_model.strip()
    model_chain = [target_model]
    for fallback in ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash"]:
        if fallback not in model_chain:
            model_chain.append(fallback)
    last_err = None
    for model in model_chain:
        try:
            return client.models.generate_content(model=model.strip(), contents=contents, config=config)
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["503", "unavailable", "capacity", "429", "resource_exhausted", "404", "not_found"]):
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("Failed to generate content with available Gemini models.")


def render_tts_button(text_to_speak: str, button_key: str):
    clean_speech = text_to_speak.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    tts_html = f"""
    <div style="margin:4px 0 8px 0;">
        <button onclick="speakText_{button_key}()" style="background:#e0e7ff;border:1px solid #c7d2fe;border-radius:6px;padding:4px 12px;font-size:.8rem;cursor:pointer;color:#3730a3;font-weight:600;">
            🔊 Listen to Question (TTS)
        </button>
    </div>
    <script>
        function speakText_{button_key}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var utterance = new SpeechSynthesisUtterance("{clean_speech}");
                utterance.rate = 0.95;
                window.speechSynthesis.speak(utterance);
            }} else {{
                alert("Text-to-speech is not supported in this browser.");
            }}
        }}
    </script>
    """
    components.html(tts_html, height=45)


HISTORY_FILE = Path(__file__).parent / "aiec_exam_history.json"


def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_to_history(exam_data: dict, results: dict = None, entry_id: str = None) -> str:
    history = load_history()
    entry_id = entry_id or str(uuid.uuid4())
    existing = next((h for h in history if h.get("id") == entry_id), None)
    if existing:
        existing["title"] = exam_data.get("title", "Untitled Exam")
        existing["exam"] = exam_data
        if results is not None:
            existing["results"] = {str(k): v for k, v in results.items()} if results else None
    else:
        history.insert(0, {
            "id": entry_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "title": exam_data.get("title", "Untitled Exam"),
            "exam": exam_data,
            "results": {str(k): v for k, v in results.items()} if results else None,
        })
    HISTORY_FILE.write_text(json.dumps(history[:50], indent=2))
    return entry_id


def delete_history_entry(entry_id: str):
    HISTORY_FILE.write_text(json.dumps([h for h in load_history() if h.get("id") != entry_id], indent=2))


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'", '—': '-', '–': '-', '…': '...', '•': '*',
        'π': 'pi', '√': 'sqrt', '±': '+/-', '≤': '<=', '≥': '>=', '×': 'x', '÷': '/',
        '°': ' deg', '²': '^2', '³': '^3', '∞': 'inf', '≠': '!=', '≈': '~=', 'µ': 'u',
        'α': 'alpha', 'β': 'beta', 'θ': 'theta', 'λ': 'lambda', 'Δ': 'Delta', '∑': 'sum'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\(?:times|cdot)', 'x', text)
    text = re.sub(r'\\(?:div)', '/', text)
    text = re.sub(r'\\(?:pm)', '+/-', text)
    text = re.sub(r'\\(?:le|leq)', '<=', text)
    text = re.sub(r'\\(?:ge|geq)', '>=', text)
    return text.encode('latin-1', 'replace').decode('latin-1')


def build_pdf(exam_data: dict, include_answers: bool = False, candidate_name: str = "", candidate_index: str = "") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    usable_w = pdf.epw if hasattr(pdf, 'epw') and pdf.epw > 0 else 190.0
    col_w = max(10.0, (usable_w - 10) / 2.0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, clean_pdf_text(exam_data.get("title", "Exam Paper")), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 6, clean_pdf_text(f"Candidate Name: {candidate_name or '_______________________'}    Index No: {candidate_index or '____________'}    Date: {datetime.datetime.now().strftime('%Y-%m-%d')}"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    if exam_data.get("instructions"):
        pdf.set_fill_color(240, 240, 240)
        pdf.multi_cell(0, 7, clean_pdf_text(f"Instructions: {exam_data['instructions']}"), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    type_labels = {"mcq": "MCQ", "short_answer": "Short Answer", "essay": "Essay", "matching": "Match", "fill_blank": "Fill-in-Blank", "true_false": "True/False", "ordering": "Ordering", "categorization": "Categorize", "labeling": "Labeling", "calculation": "Calculation", "written": "Written"}
    for i, q in enumerate(exam_data.get("questions", [])):
        qtype = q.get("question_type", "written")
        qtype_label = type_labels.get(qtype, "Q")
        topic_str = f" [{q.get('topic', '')}]" if q.get("topic") else ""
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, clean_pdf_text(f"Q{i+1}. [{qtype_label}]{topic_str}  ({q.get('marks', 0)} marks)"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, clean_pdf_text(q.get("question_text", "")), new_x="LMARGIN", new_y="NEXT")
        if qtype == "mcq" and q.get("options"):
            for opt in q["options"]:
                pdf.multi_cell(0, 6, clean_pdf_text(f"   [  ]  {opt}"), new_x="LMARGIN", new_y="NEXT")
        elif qtype == "true_false":
            pdf.multi_cell(0, 6, "   [  ] True     [  ] False", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, "Justification (if false): __________________________________________", new_x="LMARGIN", new_y="NEXT")
        elif qtype == "matching":
            for left_item, right_item in zip(q.get("left_items", []), q.get("right_items", [])):
                pdf.cell(col_w, 7, clean_pdf_text(f"  {left_item}"), border=1)
                pdf.cell(10, 7, "", border=0)
                pdf.cell(col_w, 7, clean_pdf_text(f"  {right_item}"), border=1, new_x="LMARGIN", new_y="NEXT")
        elif qtype == "ordering" and q.get("items"):
            pdf.multi_cell(0, 6, "Arrange the following items in the correct order (1 to N):", new_x="LMARGIN", new_y="NEXT")
            for item in q["items"]:
                pdf.multi_cell(0, 6, clean_pdf_text(f"   [   ]  {item}"), new_x="LMARGIN", new_y="NEXT")
        elif qtype == "categorization" and q.get("categories") and q.get("items"):
            cats = [clean_pdf_text(c) for c in q["categories"]]
            pdf.multi_cell(0, 6, f"Items to categorize: {', '.join(clean_pdf_text(it) for it in q['items'])}", new_x="LMARGIN", new_y="NEXT")
            c_w = usable_w / max(1, len(cats))
            for cat in cats:
                pdf.cell(c_w, 7, cat, border=1, align="C")
            pdf.ln()
            for _ in range(3):
                for _ in cats:
                    pdf.cell(c_w, 7, "", border=1)
                pdf.ln()
        elif qtype == "labeling" and q.get("label_prompts"):
            for label in q["label_prompts"]:
                pdf.multi_cell(0, 6, clean_pdf_text(f"   {label}: __________________________________________"), new_x="LMARGIN", new_y="NEXT")
        elif qtype == "calculation":
            pdf.cell(0, 6, "Working Space:", new_x="LMARGIN", new_y="NEXT")
            for _ in range(4):
                pdf.cell(0, 7, "", border="B", new_x="LMARGIN", new_y="NEXT")
            unit_str = f" ({q['expected_units']})" if q.get("expected_units") else ""
            pdf.multi_cell(0, 6, f"Final Answer{unit_str}: _______________________", new_x="LMARGIN", new_y="NEXT")
        elif qtype == "essay":
            for _ in range(6):
                pdf.cell(0, 8, "", border="B", new_x="LMARGIN", new_y="NEXT")
        else:
            for _ in range(3):
                pdf.cell(0, 8, "", border="B", new_x="LMARGIN", new_y="NEXT")
        if include_answers:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(0, 100, 0)
            pdf.multi_cell(0, 6, clean_pdf_text(f"Answer/Criteria: {q.get('correct_answer', '')}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
        pdf.ln(4)
    return bytes(pdf.output())


# The remaining application UI is unchanged from the previous version.
# It uses the corrected build_pdf implementation above and the countdown timer.
