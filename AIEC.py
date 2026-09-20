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
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from google import genai
from google.genai import types

from client_adapter import is_backend_available, regenerate_question_api, grade_exam_api
from backend.app.services.hybrid_grader import HybridGrader
from backend.app.services.gemini_service import GeminiService
from backend.app.config import settings
from backend.app.models.exam import Question, ExamPaper, GradedQuestion, GradingResponse

st.set_page_config(layout="wide", page_title="AIEC — AI Exam Creator & Evaluator", page_icon="📝")

# ════════════════════════════════════════════════════════════════
# FORMATTING & MATH RENDERING HELPERS
# ════════════════════════════════════════════════════════════════

def format_math_latex(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def render_standard_math(text: str, prefix: str = ""):
    """
    Renders pure standalone math expressions using st.latex(), 
    and mixed text/inline math using st.markdown() which natively supports KaTeX.
    """
    if not text:
        return
        
    clean_text = text.strip()
    
    # Check if string is strictly a pure math expression wrapped in single or double dollars
    is_pure_math = (
        (clean_text.startswith("$") and clean_text.endswith("$")) 
        and clean_text.count("$") <= 4 
        and len(clean_text) > 2
    )
    
    if is_pure_math:
        if prefix:
            st.markdown(f"{prefix}")
        raw_math = clean_text.strip("$")
        st.latex(raw_math)
    else:
        st.markdown(f"{prefix}{clean_text}")

# ════════════════════════════════════════════════════════════════
# MODEL FALLBACK, SAFE JSON PARSER & ACCESSIBILITY TTS HELPERS
# ════════════════════════════════════════════════════════════════

def safe_parse_json(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Response text is empty.")
    
    clean_text = raw_text.strip()
    
    if "```" in clean_text:
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        clean_text = clean_text.strip()
        
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
    valid_fallbacks = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash"]
    
    for fallback in valid_fallbacks:
        if fallback not in model_chain:
            model_chain.append(fallback)
    
    last_err = None
    for model in model_chain:
        try:
            return client.models.generate_content(
                model=model.strip(),
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e)
            if any(k in err_str.lower() for k in ["503", "unavailable", "capacity", "429", "resource_exhausted", "404", "not_found"]):
                last_err = e
                continue
            raise e
    raise last_err if last_err else RuntimeError("Failed to generate content with available Gemini models.")

def render_tts_button(text_to_speak: str, button_key: str):
    clean_speech = text_to_speak.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    tts_html = f"""
    <div style="margin: 4px 0 8px 0;">
        <button onclick="speakText_{button_key}()" style="
            background: #e0e7ff;
            border: 1px solid #c7d2fe;
            border-radius: 6px;
            padding: 4px 12px;
            font-size: 0.8rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #3730a3;
            font-weight: 600;
        ">
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

# ════════════════════════════════════════════════════════════════
# HISTORY HELPERS
# ════════════════════════════════════════════════════════════════

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
    if not entry_id:
        entry_id = str(uuid.uuid4())
    existing = next((h for h in history if h.get("id") == entry_id), None)
    if existing:
        existing["title"] = exam_data.get("title", "Untitled Exam")
        existing["exam"] = exam_data
        if results is not None:
            existing["results"] = {str(k): v for k, v in results.items()} if results else None
    else:
        entry = {
            "id": entry_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "title": exam_data.get("title", "Untitled Exam"),
            "exam": exam_data,
            "results": {str(k): v for k, v in results.items()} if results else None,
        }
        history.insert(0, entry)
    history = history[:50]
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    return entry_id

def delete_history_entry(entry_id: str):
    history = [h for h in load_history() if h.get("id") != entry_id]
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

# ════════════════════════════════════════════════════════════════
# PDF HELPERS
# ════════════════════════════════════════════════════════════════

def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '—': '-', '–': '-', '…': '...', '•': '*',
        'π': 'pi', '√': 'sqrt', '±': '+/-', '≤': '<=', '≥': '>=',
        '×': 'x', '÷': '/', '°': ' deg', '²': '^2', '³': '^3',
        '∞': 'inf', '≠': '!=', '≈': '~=', 'µ': 'u', 'α': 'alpha',
        'β': 'beta', 'θ': 'theta', 'λ': 'lambda', 'Δ': 'Delta', '∑': 'sum'
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

def build_pdf(exam_data: dict, include_answers: bool = False) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    usable_w = pdf.epw if hasattr(pdf, 'epw') and pdf.epw > 0 else 190.0
    col_w = max(10.0, (usable_w - 10) / 2.0)
    
    pdf.set_font("Helvetica", "B", 16)
    title = clean_pdf_text(exam_data.get("title", "Exam Paper"))
    pdf.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 11)
    if exam_data.get("instructions"):
        pdf.set_fill_color(240, 240, 240)
        instr = clean_pdf_text(f"Instructions: {exam_data['instructions']}")
        pdf.multi_cell(0, 7, instr, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    type_labels = {
        "mcq": "MCQ", "short_answer": "Short Answer", "essay": "Essay",
        "matching": "Match", "fill_blank": "Fill-in-Blank", "true_false": "True/False",
        "ordering": "Ordering", "categorization": "Categorize", "labeling": "Labeling",
        "calculation": "Calculation", "written": "Written"
    }

    for i, q in enumerate(exam_data.get("questions", [])):
        qtype = q.get("question_type", "written")
        qtype_label = type_labels.get(qtype, "Q")
        topic_str = f" [{q.get('topic', '')}]" if q.get("topic") else ""
        header_str = clean_pdf_text(f"Q{i+1}. [{qtype_label}]{topic_str}  ({q.get('marks', 0)} marks)")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, header_str, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 11)
        qtext = clean_pdf_text(q.get("question_text", ""))
        pdf.multi_cell(0, 7, qtext, new_x="LMARGIN", new_y="NEXT")
        
        if qtype == "mcq" and q.get("options"):
            for opt in q["options"]:
                opt_str = clean_pdf_text(f"   [  ]  {opt}")
                pdf.multi_cell(0, 6, opt_str, new_x="LMARGIN", new_y="NEXT")

        elif qtype == "true_false":
            pdf.multi_cell(0, 6, "   [  ] True     [  ] False", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, "Justification (if false): __________________________________________", new_x="LMARGIN", new_y="NEXT")

        elif qtype == "matching":
            lefts  = q.get("left_items", [])
            rights = q.get("right_items", [])
            for l, r in zip(lefts, rights):
                l_str = clean_pdf_text(f"  {l}")
                r_str = clean_pdf_text(f"  {r}")
                pdf.cell(col_w, 7, l_str, border=1)
                pdf.cell(10, 7, "", border=0)
                pdf.cell(col_w, 7, r_str, border=1, new_x="LMARGIN", new_y="NEXT")

        elif qtype == "ordering" and q.get("items"):
            pdf.multi_cell(0, 6, "Arrange the following items in the correct order (1 to N):", new_x="LMARGIN", new_y="NEXT")
            for idx, item in enumerate(q["items"]):
                item_str = clean_pdf_text(f"   [   ]  {item}")
                pdf.multi_cell(0, 6, item_str, new_x="LMARGIN", new_y="NEXT")

        elif qtype == "categorization" and q.get("categories") and q.get("items"):
            cats = [clean_pdf_text(c) for c in q["categories"]]
            items = [clean_pdf_text(it) for it in q["items"]]
            pdf.multi_cell(0, 6, f"Items to categorize: {', '.join(items)}", new_x="LMARGIN", new_y="NEXT")
            c_w = usable_w / max(1, len(cats))
            for c in cats:
                pdf.cell(c_w, 7, c, border=1, align="C")
            pdf.ln()
            for _ in range(3):
                for _ in cats:
                    pdf.cell(c_w, 7, "", border=1)
                pdf.ln()

        elif qtype == "labeling" and q.get("label_prompts"):
            for label in q["label_prompts"]:
                lbl_str = clean_pdf_text(f"   {label}: __________________________________________")
                pdf.multi_cell(0, 6, lbl_str, new_x="LMARGIN", new_y="NEXT")

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
            ans_str = clean_pdf_text(f"Answer/Criteria: {q.get('correct_answer', '')}")
            pdf.multi_cell(0, 6, ans_str, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
        pdf.ln(4)

    return bytes(pdf.output())

# ════════════════════════════════════════════════════════════════
# MARKED SCRIPT PDF BUILDER
# ════════════════════════════════════════════════════════════════

def build_marked_script_pdf(exam_data: dict, grading_result: dict, student_answers: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    graded_qs = grading_result.get("graded_questions", [])
    graded_map = {g.get("question_index"): g for g in graded_qs}

    total_awarded = sum(g.get("score", 0) for g in graded_qs)
    total_possible = sum(q.get("marks", 0) for q in exam_data.get("questions", []))
    pct = round((total_awarded / total_possible * 100), 1) if total_possible > 0 else 0

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, clean_pdf_text(exam_data.get("title", "Exam Paper") + " — MARKED SCRIPT"),
                   new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 6,
                   new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 100, 0)
    pdf.multi_cell(0, 8, clean_pdf_text(f"TOTAL SCORE: {total_awarded} / {total_possible}  ({pct}%)"),
                   new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for i, q in enumerate(exam_data.get("questions", [])):
        g_info = graded_map.get(i, {})
        q_score = g_info.get("score", 0)
        q_max = q.get("marks", 0)
        student_ans = str(student_answers.get(i, "No answer provided"))
        feedback = g_info.get("feedback", "")
        is_correct = q_score >= q_max
        is_partial = 0 < q_score < q_max

        pdf.set_font("Helvetica", "B", 11)
        if is_correct:
            pdf.set_text_color(0, 128, 0)
            marker = "[CORRECT]"
        elif is_partial:
            pdf.set_text_color(180, 100, 0)
            marker = "[PARTIAL]"
        else:
            pdf.set_text_color(180, 0, 0)
            marker = "[INCORRECT]"
        pdf.multi_cell(0, 7,
                       clean_pdf_text(f"Q{i+1}. [{q.get('question_type','').upper()}] {marker}  {q_score}/{q_max} marks"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, clean_pdf_text(q.get("question_text", "")), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(0, 0, 180)
        pdf.multi_cell(0, 6, clean_pdf_text(f"Your Answer: {student_ans}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(0, 120, 0)
        pdf.multi_cell(0, 6, clean_pdf_text(f"Model Answer: {q.get('correct_answer', '')}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(100, 0, 0)
        if feedback:
            pdf.multi_cell(0, 6, clean_pdf_text(f"Feedback: {feedback}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(5)

    return bytes(pdf.output())

# ════════════════════════════════════════════════════════════════
# AI PERSONALISED STUDY PLAN GENERATOR
# ════════════════════════════════════════════════════════════════

def generate_ai_study_plan(api_key: str, model_name: str, weak_topics: list, exam_title: str) -> str:
    if not api_key or not weak_topics:
        return ""
    try:
        client = genai.Client(api_key=api_key)
        topics_str = ", ".join([f"{t['topic']} ({t['pct']}%)" for t in weak_topics])
        prompt = (
            f"You are an expert academic tutor. A student just completed an exam titled '{exam_title}'. "
            f"Their weakest topics are: {topics_str}. "
            "Write a concise, motivating, personalised 2-week revision plan targeting ONLY these weak topics. "
            "Format as a numbered markdown list. Each item should have: topic name, what to review, a specific practice activity. "
            "Keep the tone encouraging and constructive. Maximum 300 words."
        )
        resp = generate_with_gemini_fallback(
            client, model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=512)
        )
        return resp.text.strip() if resp and resp.text else ""
    except Exception:
        return ""

# ════════════════════════════════════════════════════════════════
# CUSTOM STYLING & HERO HEADER
# ════════════════════════════════════════════════════════════════

st.write("")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');
    
    /* Import OpenDyslexic Font from CDN */
    @font-face {
        font-family: 'OpenDyslexic';
        src: url('https://cdn.jsdelivr.net/npm/open-dyslexic@1.0.3/fonts/OpenDyslexic-Regular.otf') format('opentype');
        font-weight: 400;
        font-style: normal;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #e2e8f0;
    }

    /* Texture & Theme Styles */
    .paper-sheet-cream {
        background-color: #fdfbf7 !important;
        background-image: radial-gradient(#e2d9cd 0.8px, transparent 0.8px), radial-gradient(#e2d9cd 0.8px, #fdfbf7 0.8px) !important;
        background-size: 24px 24px !important;
        border: 2px solid #e5dec9 !important;
        color: #1e293b !important;
    }
    .paper-sheet-cream * { color: #1e293b !important; }

    .paper-sheet-white {
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        color: #0f172a !important;
    }
    .paper-sheet-white * { color: #0f172a !important; }

    .paper-sheet-dark {
        background: #1e293b !important;
        border: 2px solid #334155 !important;
        color: #f8fafc !important;
    }
    .paper-sheet-dark * { color: #f8fafc !important; }

    .paper-sheet-contrast {
        background: #000000 !important;
        border: 4px solid #facc15 !important;
        color: #ffffff !important;
    }
    .paper-sheet-contrast * { color: #ffffff !important; }

    /* Font Style Applications */
    .font-serif, .font-serif * { font-family: 'Cinzel', 'Times New Roman', serif !important; }
    .font-sans, .font-sans * { font-family: 'Inter', sans-serif !important; }
    .font-dyslexic, .font-dyslexic * { font-family: 'OpenDyslexic', sans-serif !important; line-height: 1.6 !important; letter-spacing: 0.05em !important; }

    .exam-custom-heading {
        margin: 1.5rem 0 1.25rem 0;
        padding: 1.5rem 2rem;
        border: 1px solid rgba(129, 140, 248, 0.45);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(49, 46, 129, 0.9), rgba(88, 28, 135, 0.85));
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    .exam-custom-heading-kicker {
        color: #c4b5fd;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        margin-bottom: 0.45rem;
    }

    .exam-custom-heading h1 {
        margin: 0;
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .exam-custom-heading-meta {
        margin-top: 0.65rem;
        color: #ddd6fe;
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

backend_online = is_backend_available()

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 24px 30px; margin-bottom: 18px; box-shadow: 0 20px 40px rgba(15, 23, 42, 0.25);">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 18px;">
            <div style="width: 56px; height: 56px; background: linear-gradient(135deg, #6366f1, #a855f7); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);">
                📝
            </div>
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    AIEC — AI Exam Creator & Evaluator
                </h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.92rem; font-weight: 500;">
                    An AI powered Exam generator where it creates and marks your exam and all you need to do is to answer it within the website.
                </p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("### ☁️ Set-Up Options")

default_api_key = settings.get_api_key() or ""
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=default_api_key,
    type="password",
    help="Get your Google Gemini API key here: [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)"
)
selected_model = "gemini-3.8-flash"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Mode")
teacher_mode = st.sidebar.toggle("Teacher Mode", value=False, help="Shows full mark schemes, inline correct answers, and criteria.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Paper Settings")

diff_auto = st.sidebar.checkbox("Auto Difficulty (AI Decides)", value=True)
if diff_auto:
    selected_difficulty = "Auto"
else:
    selected_difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Question Counts & Types")
q_count_auto = st.sidebar.checkbox("Auto Question Mix (AI Decides)", value=True)

if q_count_auto:
    n_mcq = 0
    n_short = 0
    n_essay = 0
    n_matching = 0
    n_true_false = 0
    n_other = 0
else:
    n_mcq = st.sidebar.slider("Multiple Choice (MCQ)", 0, 15, 3)
    n_short = st.sidebar.slider("Short Answer", 0, 10, 3)
    n_essay = st.sidebar.slider("Extended Essay", 0, 5, 1)
    n_matching = st.sidebar.slider("Matching", 0, 5, 1)
    n_true_false = st.sidebar.slider("True / False", 0, 10, 2)
    n_other = st.sidebar.slider("Other (Ordering/Calc/Labeling)", 0, 5, 1)


# ════════════════════════════════════════════════════════════════
# SIDEBAR HISTORY
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Saved Exam History")

history_list = load_history()
if not history_list:
    st.sidebar.caption("No past exams saved yet.")
else:
    for h in history_list:
        h_id = h.get("id")
        h_title = h.get("title", "Untitled Exam")
        h_time = h.get("timestamp", "")
        has_results = " ✅" if h.get("results") else ""

        with st.sidebar.expander(f"📄 {h_title[:22]}...{has_results}"):
            st.caption(f"Created: {h_time}")
            c_load, c_del = st.columns(2)
            with c_load:
                if st.button("Load", key=f"sb_load_{h_id}", use_container_width=True):
                    st.session_state["exam_paper"] = h.get("exam")
                    st.session_state["grading_result"] = h.get("results")
                    st.session_state["active_exam_id"] = h_id
                    st.session_state["exam_start_timestamp"] = datetime.datetime.now().timestamp()
                    st.session_state["flagged_questions"] = set()
                    st.rerun()
            with c_del:
                if st.button("Delete", key=f"sb_del_{h_id}", use_container_width=True):
                    delete_history_entry(h_id)
                    if st.session_state.get("active_exam_id") == h_id:
                        st.session_state["exam_paper"] = None
                        st.session_state["grading_result"] = None
                    st.rerun()

# ════════════════════════════════════════════════════════════════
# MAIN INPUT FORM & CREATION UI
# ════════════════════════════════════════════════════════════════

has_active_exam = "exam_paper" in st.session_state and st.session_state["exam_paper"] is not None

if not has_active_exam:
    with st.container(border=True):
        st.write("Type your prompt / instructions here:")
        input1 = st.text_area("Prompt Instructions", key="extra_info", height=100, label_visibility="hidden")

    Col2, Col3, Col4 = st.columns(3)

    with Col2:
        with st.container(border=True):
            st.write("Upload your course work (SoW, Notes, Images, etc)")
            file1 = st.file_uploader("Course work", label_visibility="hidden", key="file1", accept_multiple_files=True)
    with Col3:
        with st.container(border=True):
            st.write("Upload your mark scheme for each past paper.")
            file3 = st.file_uploader("Mark scheme", label_visibility="hidden", key="file3", accept_multiple_files=True)
    with Col4:
        with st.container(border=True):
            st.write("Upload your past papers here.")
            file2 = st.file_uploader("Past papers", label_visibility="hidden", key="file2", accept_multiple_files=True)

    if st.button("Generate Exam Paper", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            client = genai.Client(api_key=api_key)
            all_files = []

            def save_file(uploaded):
                ext = os.path.splitext(uploaded.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(uploaded.getbuffer())
                tmp.close()
                return tmp.name

            with st.spinner("Processing uploaded files..."):
                for f in (file1 or []):
                    p = save_file(f)
                    all_files.append(client.files.upload(file=p))
                for f in (file2 or []):
                    p = save_file(f)
                    all_files.append(client.files.upload(file=p))
                for f in (file3 or []):
                    p = save_file(f)
                    all_files.append(client.files.upload(file=p))

            with st.spinner("Generating exam paper with Gemini..."):
                prompt = (
                    "You are an expert exam paper creator. Build a highly balanced, comprehensive exam paper based on the uploaded materials and user instructions.\n"
                    "You MUST support a rich variety of question types:\n"
                    "- 'mcq': Multiple choice (provide 4 choices in 'options').\n"
                    "- 'short_answer': Concise 1-4 mark response.\n"
                    "- 'essay': Comprehensive structured essay prompt (5-20 marks).\n"
                    "- 'matching': Dual column list ('left_items' and 'right_items', shuffle right_items).\n"
                    "- 'fill_blank': Text passage with [blank] placeholders.\n"
                    "- 'true_false': Statement verification.\n"
                    "- 'ordering': Sequence list of items in scrambled order ('items').\n"
                    "- 'categorization': Sorting items into category buckets ('categories' and 'items').\n"
                    "- 'labeling': Diagram / prompt part labeling ('label_prompts').\n"
                    "- 'calculation': Math/Physics problem with units ('expected_units').\n\n"
                )

                if selected_difficulty == "Auto":
                    prompt += "Determine the optimal difficulty level based on the material.\n"
                else:
                    prompt += f"Target difficulty level: {selected_difficulty}.\n"

                if q_count_auto:
                    prompt += "Determine the best total question count and mix of question types dynamically to test the material thoroughly.\n"
                else:
                    prompt += (
                        f"Requested Question Mix:\n"
                        f"- MCQ: {n_mcq}\n"
                        f"- Short Answer: {n_short}\n"
                        f"- Essay: {n_essay}\n"
                        f"- Matching: {n_matching}\n"
                        f"- True/False: {n_true_false}\n"
                        f"- Other (Ordering/Calc/Labeling): {n_other}\n"
                    )

                if input1:
                    prompt += f"\nUser instructions: {input1}\n"

                contents = all_files + [prompt]

                try:
                    exam = None
                    last_gen_err = None
                    for attempt in range(3):
                        try:
                            response = generate_with_gemini_fallback(
                                client=client,
                                target_model=selected_model,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=ExamPaper,
                                ),
                            )
                            exam = safe_parse_json(response.text)
                            if exam and "questions" in exam:
                                break
                        except Exception as attempt_err:
                            last_gen_err = attempt_err
                            continue

                    if not exam:
                        raise last_gen_err or RuntimeError("Failed to parse valid exam structure after retries.")

                    st.session_state["exam_paper"] = exam
                    st.session_state["exam_answers"] = {}
                    st.session_state["grading_result"] = None
                    st.session_state["flagged_questions"] = set()
                    st.session_state["exam_start_timestamp"] = datetime.datetime.now().timestamp()
                    e_id = save_to_history(exam)
                    st.session_state["active_exam_id"] = e_id
                    st.success("Exam paper generated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating exam paper: {e}")
else:
   if st.button("➕ Create New Exam", use_container_width=True):
            st.session_state["exam_paper"] = None
            st.session_state["grading_result"] = None
            st.session_state["active_exam_id"] = None
            st.session_state["exam_start_timestamp"] = None
            st.session_state["flagged_questions"] = set()
            st.rerun()

# ════════════════════════════════════════════════════════════════
# EXAM PAPER DISPLAY & INTERACTIVE PRACTICE MODE
# ════════════════════════════════════════════════════════════════

if "exam_paper" in st.session_state and st.session_state["exam_paper"]:
    exam = st.session_state["exam_paper"]
    questions = exam.get("questions", [])

    st.markdown("---")
    st.subheader("📝 Examination Paper Workspace")

    exam_title = str(exam.get("title", "Examination Paper")).strip() or "Examination Paper"
    
    custom_name = str(exam.get("custom_name", "")).strip()
    if custom_name:
        exam_title = f"{exam_title} ({custom_name})"
        
    exam_subject = str(exam.get("subject", "")).strip()
    exam_grade = str(exam.get("grade_level", "")).strip()
    exam_meta = " · ".join(value for value in [exam_subject, exam_grade] if value)

    st.markdown(
        f"""
        <div class="exam-custom-heading">
            <div class="exam-custom-heading-kicker">📝 EXAMINATION PAPER</div>
            <h1>{exam_title}</h1>
            {f'<div class="exam-custom-heading-meta">{exam_meta}</div>' if exam_meta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "exam_answers" not in st.session_state:
        st.session_state["exam_answers"] = {}
        
    if "flagged_questions" not in st.session_state:
        st.session_state["flagged_questions"] = set()

    for i, q in enumerate(questions):
        raw_qtype = q.get("question_type", "written").lower()
        qtype = raw_qtype
        display_qtype = raw_qtype.upper()
        marks = q.get("marks", 2)
        topic = q.get("topic", "")

        with st.container(border=True):
            col_q_hdr, col_q_flag = st.columns([5, 1])
            with col_q_hdr:
                q_text_fmt = format_math_latex(q.get('question_text', ''))
                badges_html = f"""<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap;">
                    <span style="background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; text-transform: uppercase;">{display_qtype}</span>
                    {f'<span style="background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 6px;">{topic}</span>' if topic else ''}
                    <span style="background: #fef3c7; color: #92400e; border: 1px solid #fde68a; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;">{marks} Marks</span>
                </div>"""
                st.markdown(badges_html, unsafe_allow_html=True)
                
                render_standard_math(q_text_fmt, prefix=f"**Q{i+1}.** ")
                
            with col_q_flag:
                is_flagged = st.checkbox("🚩 Flag", key=f"flag_chk_{i}", value=(i in st.session_state["flagged_questions"]))
                if is_flagged:
                    st.session_state["flagged_questions"].add(i)
                else:
                    st.session_state["flagged_questions"].discard(i)

            render_tts_button(q.get("question_text", ""), f"q_{i}")

            if teacher_mode:
                st.markdown("---")
                render_standard_math(q.get('correct_answer', 'N/A'), prefix="💡 **Teacher Key / Criteria:** ")

            if raw_qtype == "mcq" and q.get("options"):
                opts = [format_math_latex(o) for o in q["options"]]
                st.session_state["exam_answers"][i] = st.radio(
                    f"Select Answer for Q{i+1}:",
                    opts,
                    key=f"ans_{i}",
                    index=None
                )

            elif qtype == "matching" and q.get("left_items") and q.get("right_items"):
                st.write("🔗 **Match each item from Left to Right:**")
                lefts = q["left_items"]
                rights = ["-- Select Match --"] + q["right_items"]
                user_matches = {}
                for l_idx, left_item in enumerate(lefts):
                    m_col1, m_col2 = st.columns([1, 1])
                    with m_col1:
                        render_standard_math(left_item)
                    with m_col2:
                        sel = st.selectbox(
                            f"Match for '{left_item}'",
                            rights,
                            key=f"match_{i}_{l_idx}",
                            label_visibility="collapsed"
                        )
                        if sel != "-- Select Match --":
                            user_matches[left_item] = sel
                st.session_state["exam_answers"][i] = json.dumps(user_matches)

            elif qtype == "true_false":
                tf_choice = st.radio(
                    "Statement is True or False?",
                    ["True", "False"],
                    key=f"ans_tf_{i}",
                    index=None,
                    horizontal=True
                )
                tf_reason = st.text_input("Justification (optional/if false):", key=f"ans_tf_reason_{i}")
                st.session_state["exam_answers"][i] = f"Choice: {tf_choice} | Reasoning: {tf_reason}"

            elif qtype == "ordering" and q.get("items"):
                st.write("🔢 **Assign the correct step position (1 to N) for each item:**")
                items = q["items"]
                positions = list(range(1, len(items) + 1))
                user_order = {}
                for it_idx, item in enumerate(items):
                    o_col1, o_col2 = st.columns([3, 1])
                    with o_col1:
                        render_standard_math(item)
                    with o_col2:
                        pos = st.selectbox(
                            f"Position for item {it_idx}",
                            positions,
                            key=f"order_{i}_{it_idx}",
                            label_visibility="collapsed"
                        )
                        user_order[item] = pos
                st.session_state["exam_answers"][i] = json.dumps(user_order)

            elif qtype == "categorization" and q.get("categories") and q.get("items"):
                st.write("🏷️ **Assign each item to its correct Category:**")
                cats = q["categories"]
                items = q["items"]
                user_cats = {}
                for it_idx, item in enumerate(items):
                    c_col1, c_col2 = st.columns([2, 1])
                    with c_col1:
                        render_standard_math(item)
                    with c_col2:
                        cat_sel = st.selectbox(
                            f"Category for item {it_idx}",
                            cats,
                            key=f"cat_{i}_{it_idx}",
                            label_visibility="collapsed"
                        )
                        user_cats[item] = cat_sel
                st.session_state["exam_answers"][i] = json.dumps(user_cats)

            elif qtype == "labeling" and q.get("label_prompts"):
                st.write("🏷️ **Provide labels for each key/part:**")
                prompts = q["label_prompts"]
                user_labels = {}
                for l_idx, lbl in enumerate(prompts):
                    val = st.text_input(f"Label for '{lbl}':", key=f"lbl_{i}_{l_idx}")
                    user_labels[lbl] = val
                st.session_state["exam_answers"][i] = json.dumps(user_labels)

            elif qtype == "calculation":
                unit_str = f" ({q['expected_units']})" if q.get("expected_units") else ""
                ans_val = st.text_input(f"Final Answer{unit_str}:", key=f"calc_ans_{i}")
                working = st.text_area("Working / Steps:", key=f"calc_work_{i}", height=80)
                st.session_state["exam_answers"][i] = f"Answer: {ans_val} {unit_str} | Working: {working}"

            elif qtype == "essay":
                ans_text = st.text_area("Write your essay response:", key=f"ans_essay_{i}", height=180)
                st.session_state["exam_answers"][i] = ans_text

            else:
                ans_text = st.text_area("Type your answer:", key=f"ans_written_{i}", height=90)
                st.session_state["exam_answers"][i] = ans_text

            act_col1, act_col2 = st.columns(2)
            with act_col1:
                with st.popover("💡 Reveal AI Hint"):
                    st.info(f"**Topic Focus:** `{q.get('topic', 'General Core Concept')}`\n\n💡 **Hint Guidance:** Read carefully and focus on key terminology. Break down your answer into clear, structured points using relevant examples.")
            with act_col2:
                with st.popover("⚙️ Question Options"):
                    regen_inst = st.text_input("Instructions for regeneration:", key=f"regen_inst_{i}", placeholder="e.g. Make it harder")
                    if st.button("🔄 Regenerate This Question", key=f"btn_regen_{i}"):
                        if not api_key:
                            st.error("Gemini API key is required.")
                        else:
                            with st.spinner("Regenerating question with Cloud Engine..."):
                                try:
                                    if backend_online:
                                        new_q = regenerate_question_api(
                                            original_question=q,
                                            edit_instructions=regen_inst if regen_inst else "Provide a fresh alternative question on the same topic",
                                            exam_title=exam.get("title", "Exam Paper"),
                                            api_key=api_key,
                                            model_name=selected_model
                                        )
                                    else:
                                        client = genai.Client(api_key=api_key)
                                        regen_prompt = (
                                            f"Regenerate question Q{i+1} from this exam. "
                                            f"Existing question: {json.dumps(q)}. "
                                            f"User instructions: {regen_inst if regen_inst else 'Provide a fresh alternative question on the same topic'}. "
                                            f"Return JSON matching Question schema."
                                        )
                                        resp = generate_with_gemini_fallback(
                                            client=client,
                                            target_model=selected_model,
                                            contents=regen_prompt,
                                            config=types.GenerateContentConfig(
                                                response_mime_type="application/json",
                                                response_schema=Question,
                                            ),
                                        )
                                        new_q = safe_parse_json(resp.text)
                                    exam["questions"][i] = new_q
                                    st.session_state["exam_paper"] = exam
                                    st.rerun()
                                except Exception as ex:
                                    st.error(f"Failed to regenerate: {ex}")

    if st.button("📊 Submit & Grade Exam Paper", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar or configure Secret Manager.")
        else:
            answers = st.session_state.get("exam_answers", {})
            with st.spinner("Grading submission..."):
                try:
                    if backend_online:
                        graded_data = grade_exam_api(
                            exam=exam,
                            student_answers=answers,
                            api_key=api_key,
                            model_name=selected_model
                        )
                    else:
                        g_service = GeminiService(api_key=api_key)
                        hybrid = HybridGrader(gemini_service=g_service)
                        exam_obj = ExamPaper.model_validate(exam)
                        grading_resp = hybrid.grade(exam_obj, answers, model_name=selected_model)
                        graded_data = json.loads(grading_resp.model_dump_json())

                    st.session_state["grading_result"] = graded_data
                    active_id = st.session_state.get("active_exam_id")
                    save_to_history(exam, graded_data, entry_id=active_id)
                    st.success("Exam successfully graded via Hybrid Multi-Tier Engine!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error grading exam paper: {ex}")


if "grading_result" in st.session_state and st.session_state["grading_result"]:
    g_res = st.session_state["grading_result"]
    graded_qs = g_res.get("graded_questions", [])
    exam = st.session_state.get("exam_paper", {})
    questions = exam.get("questions", [])
    graded_map = {g.get("question_index"): g for g in graded_qs}

    total_awarded = sum(g.get("score", 0) for g in graded_qs)
    total_possible = sum(q.get("marks", 0) for q in questions)
    pct = round((total_awarded / total_possible * 100), 1) if total_possible > 0 else 0
    det_count = sum(1 for g in graded_qs if g.get("graded_by") == "deterministic")
    ai_count = len(graded_qs) - det_count
    flagged_qs = st.session_state.get("flagged_questions", set())

    if pct >= 90:
        grade_str, grade_emoji, banner_grad, passed = "A* / Distinction", "🏆", "linear-gradient(135deg,#065f46,#10b981)", True
    elif pct >= 80:
        grade_str, grade_emoji, banner_grad, passed = "A / Excellent", "🌟", "linear-gradient(135deg,#166534,#22c55e)", True
    elif pct >= 70:
        grade_str, grade_emoji, banner_grad, passed = "A / Great", "🎯", "linear-gradient(135deg,#1e3a5f,#3b82f6)", True
    elif pct >= 60:
        grade_str, grade_emoji, banner_grad, passed = "B / Good", "👍", "linear-gradient(135deg,#78350f,#f59e0b)", True
    elif pct >= 50:
        grade_str, grade_emoji, banner_grad, passed = "C / Pass", "✅", "linear-gradient(135deg,#713f12,#eab308)", True
    else:
        grade_str, grade_emoji, banner_grad, passed = "Needs Revision", "📚", "linear-gradient(135deg,#374151,#6b7280)", False

    pass_label = "PASSED" if passed else "NOT PASSED"
    pass_color = "#bbf7d0" if passed else "#fecaca"
    pass_text_color = "#166534" if passed else "#991b1b"

    history = load_history()
    exam_title = exam.get("title", "")
    prev_pct = None
    prev_entry = None
    for h in history[1:]: 
        if h.get("title", "") == exam_title and h.get("results"):
            prev_gqs = h["results"].get("graded_questions", [])
            prev_possible = sum(q.get("marks", 0) for q in h.get("exam", {}).get("questions", []))
            prev_awarded = sum(g.get("score", 0) for g in prev_gqs)
            if prev_possible > 0:
                prev_pct = round(prev_awarded / prev_possible * 100, 1)
                prev_entry = {"score": f"{prev_awarded}/{prev_possible}", "pct": prev_pct,
                              "timestamp": h.get("timestamp", ""), "awarded": prev_awarded}
                break

    st.markdown("---")

    pass_badge_icon = "✅" if passed else "❌"

    hero_html = f"""<div style="background:{banner_grad};border-radius:16px;padding:36px 40px;margin-bottom:28px;position:relative;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,0.25);">
<div style="position:absolute;top:50%;right:40px;transform:translateY(-50%) rotate(-12deg);font-size:3.2rem;font-weight:900;letter-spacing:4px;color:{pass_color};opacity:0.18;pointer-events:none;">{grade_emoji}</div>
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px;">
<div>
<div style="color:rgba(255,255,255,0.75);font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Exam Results Report</div>
<h2 style="color:#fff;margin:6px 0 2px 0;font-size:1.55rem;font-weight:800;">{exam_title}</h2>
<span style="background:rgba(255,255,255,0.2);border:2px solid rgba(255,255,255,0.5);border-radius:50px;padding:8px 22px;color:#fff;font-size:1.05rem;font-weight:700;letter-spacing:0.5px;">{grade_str}</span>
<span style="margin-left:12px;background:{pass_color};color:{pass_text_color};border-radius:50px;padding:8px 22px;font-size:1rem;font-weight:800;letter-spacing:1px;">{pass_badge_icon} {pass_label}</span>
</div>
<div style="text-align:center;">
<div style="width:130px;height:130px;border-radius:50%;background:conic-gradient(rgba(255,255,255,0.95) {pct}%, rgba(255,255,255,0.15) 0%);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 0 4px rgba(255,255,255,0.15);">
<div style="width:100px;height:100px;border-radius:50%;background:rgba(0,0,0,0.25);display:flex;flex-direction:column;align-items:center;justify-content:center;">
<div style="color:#fff;font-size:1.9rem;font-weight:900;line-height:1;">{pct}%</div>
<div style="color:rgba(255,255,255,0.75);font-size:0.7rem;margin-top:2px;">SCORE</div>
</div>
</div>
<div style="color:rgba(255,255,255,0.85);font-size:0.95rem;margin-top:10px;font-weight:600;">{total_awarded} / {total_possible} marks</div>
</div>
</div>
</div>"""
    st.markdown(hero_html, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Total Score", f"{total_awarded} / {total_possible} marks")
    pct_delta = f"+{round(pct - prev_pct, 1)}%" if prev_pct is not None else None
    m2.metric("📈 Percentage", f"{pct}%", delta=pct_delta)
    m3.metric("⚡ Grading Engine", f"{det_count} Instant · {ai_count} AI")
    m4.metric("🚩 Flagged Questions", f"{len(flagged_qs)} flagged")

    st.markdown("---")

    st.subheader("📊 Topic Mastery Analytics")

    topic_scores = {}
    topic_maxes = {}
    for i, q in enumerate(questions):
        t = q.get("topic", "General") or "General"
        g_info = graded_map.get(i, {})
        topic_scores[t] = topic_scores.get(t, 0) + g_info.get("score", 0)
        topic_maxes[t] = topic_maxes.get(t, 0) + q.get("marks", 0)

    topic_pcts = {t: round(topic_scores[t] / topic_maxes[t] * 100, 1) for t in topic_maxes if topic_maxes[t] > 0}
    topic_names = list(topic_pcts.keys())
    topic_vals = list(topic_pcts.values())

    col_radar, col_table = st.columns([1, 1])

    with col_radar:
        if len(topic_names) >= 3:
            fig_radar = go.Figure(go.Scatterpolar(
                r=topic_vals + [topic_vals[0]],
                theta=topic_names + [topic_names[0]],
                fill='toself',
                fillcolor='rgba(99,102,241,0.18)',
                line=dict(color='#6366f1', width=2.5),
                marker=dict(size=7, color='#6366f1')
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%",
                                    gridcolor='rgba(148,163,184,0.3)', linecolor='rgba(148,163,184,0.3)'),
                    angularaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=30, t=20, b=20),
                height=320
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            for t, tp in topic_pcts.items():
                st.progress(tp / 100.0, text=f"{t}: {tp}%")

    with col_table:
        table_rows = []
        weak_topics_for_plan = []
        for t in topic_names:
            tp = topic_pcts[t]
            sc = topic_scores[t]
            mx = topic_maxes[t]
            if tp >= 80:
                status = "✅ Mastered"
            elif tp >= 50:
                status = "🔄 Developing"
            else:
                status = "⚠️ Needs Focus"
                weak_topics_for_plan.append({"topic": t, "pct": tp})
            table_rows.append({"Topic": t, "Awarded": sc, "Max": mx, "%": f"{tp}%", "Status": status})
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    if prev_entry:
        st.subheader("📊 Comparison with Previous Attempt")
        delta_marks = total_awarded - prev_entry["awarded"]
        delta_pct = round(pct - prev_pct, 1)
        delta_mark_str = f"▲ +{delta_marks}" if delta_marks > 0 else (f"▼ {delta_marks}" if delta_marks < 0 else "—")
        delta_pct_str = f"▲ +{delta_pct}%" if delta_pct > 0 else (f"▼ {delta_pct}%" if delta_pct < 0 else "—")
        improved_msg = "🎉 Great improvement!" if delta_pct > 0 else ("📉 Keep working at it — you'll get there!" if delta_pct < 0 else "Same score — consistent performance!")

        comp_df = pd.DataFrame([
            {"Metric": "Score", "Previous Attempt": prev_entry["score"], "This Attempt": f"{total_awarded}/{total_possible}", "Change": delta_mark_str},
            {"Metric": "Percentage", "Previous Attempt": f"{prev_pct}%", "This Attempt": f"{pct}%", "Change": delta_pct_str},
            {"Metric": "Grade", "Previous Attempt": "—", "This Attempt": grade_str, "Change": "—"},
            {"Metric": "Date", "Previous Attempt": prev_entry["timestamp"], "This Attempt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Change": "—"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        if delta_pct > 0:
            st.success(f"{improved_msg} You went up by **{delta_marks} marks** and **{delta_pct}%** from your last attempt.")
        elif delta_pct < 0:
            st.warning(improved_msg)
        else:
            st.info(improved_msg)
        st.markdown("---")

    st.subheader("📄 Detailed Question Review — Marked Script")
    st.caption("Your answers reviewed question-by-question, with model answers and feedback.")

    for i, q in enumerate(questions):
        g_info = graded_map.get(i, {})
        q_score = g_info.get("score", 0)
        q_max = q.get("marks", 0)
        q_pct = round((q_score / q_max * 100), 1) if q_max > 0 else 0
        topic = q.get("topic", "General")
        qtype = q.get("question_type", "written").upper()
        is_det = g_info.get("graded_by") == "deterministic"
        student_ans = st.session_state.get("exam_answers", {}).get(i, None)

        is_correct = q_pct >= 100
        is_partial = 0 < q_pct < 100

        if is_correct:
            pen_color = "#16a34a"
            pen_bg = "rgba(220,252,231,0.25)"
            pen_mark = "✅"
            result_label = "CORRECT"
        elif is_partial:
            pen_color = "#d97706"
            pen_bg = "rgba(254,243,199,0.25)"
            pen_mark = "⚠️"
            result_label = "PARTIAL"
        else:
            pen_color = "#dc2626"
            pen_bg = "rgba(254,226,226,0.25)"
            pen_mark = "❌"
            result_label = "INCORRECT"

        engine_label = "⚡ Deterministic" if is_det else f"🤖 {selected_model}"
        expander_label = f"{pen_mark} Q{i+1} [{qtype}] `{topic}` — {q_score}/{q_max} marks ({q_pct}%) · {engine_label}"

        with st.expander(expander_label):
            st.markdown(f"""
            <div style="
                border-left: 4px solid {pen_color};
                background: {pen_bg};
                border-radius: 0 8px 8px 0;
                padding: 14px 18px;
                margin-bottom: 12px;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:700; font-size:0.9rem;">Question {i+1} · {qtype} · {topic}</span>
                    <span style="
                        background:{pen_color}; color:#fff;
                        border-radius:50px; padding:3px 14px;
                        font-size:0.8rem; font-weight:700;
                    ">{result_label} &nbsp;{q_score}/{q_max}</span>
                </div>
                <div style="font-size:0.95rem; margin-bottom:10px;">
                    <strong>Question:</strong> {q.get('question_text','')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            a_col, b_col = st.columns(2)
            with a_col:
                if student_ans is None or student_ans == "":
                    st.markdown("""<div style="background:rgba(254,226,226,0.4);border:1px solid rgba(220,38,38,0.3);border-radius:8px;padding:10px 14px;">
                        <span style="font-weight:700;color:#dc2626;">🔴 No Answer Provided</span></div>""",
                        unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:rgba(219,234,254,0.3);border:1px solid rgba(59,130,246,0.3);border-radius:8px;padding:10px 14px;">
                        <div style="font-weight:700;color:#1d4ed8;font-size:0.8rem;margin-bottom:4px;">YOUR ANSWER</div>
                        <div>{student_ans}</div></div>""", unsafe_allow_html=True)
            with b_col:
                st.markdown(f"""<div style="background:rgba(220,252,231,0.3);border:1px solid rgba(22,163,74,0.3);border-radius:8px;padding:10px 14px;">
                    <div style="font-weight:700;color:#15803d;font-size:0.8rem;margin-bottom:4px;">✏️ MODEL ANSWER / CRITERIA</div>
                    <div>{q.get('correct_answer','')}</div></div>""", unsafe_allow_html=True)

            st.markdown(f"**💬 Feedback:** {g_info.get('feedback', 'No detailed feedback available.')}")

    st.markdown("---")

   

    st.subheader("📥 Download Your Results")
    dl1, dl2 = st.columns(2)

    with dl1:
        marked_pdf = build_marked_script_pdf(
            exam_data=exam,
            grading_result=g_res,
            student_answers=st.session_state.get("exam_answers", {})
        )
        st.download_button(
            "📄 Download Marked Script (PDF)",
            data=marked_pdf,
            file_name=f"marked_script.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
        st.caption("Your answers with ✅ ❌ ⚠️ markers, model answers & feedback per question")

    with dl2:
        report_pdf = build_pdf(
            exam_data=exam,
            include_answers=True
        )
        st.download_button(
            "📊 Download Performance Report (PDF)",
            data=report_pdf,
            file_name=f"performance_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.caption("Full exam paper with mark scheme")

    st.markdown("---")

    st.subheader("🔄 What would you like to do next?")
    act1, act2, act3 = st.columns(3)

    with act1:
        if st.button("🔄 Retake This Exam", use_container_width=True, type="primary"):
            st.session_state["exam_answers"] = {}
            st.session_state["grading_result"] = None
            st.session_state["exam_start_timestamp"] = None
            st.session_state["flagged_questions"] = set()
            st.rerun()

    with act2:
        if st.button("🆕 Create a New Exam", use_container_width=True):
            for k in ["exam_paper", "grading_result", "exam_answers", "exam_start_timestamp",
                      "flagged_questions", "active_exam_id"]:
                st.session_state.pop(k, None)
            st.rerun()

    with act3:
        if st.button("📚 View Exam History", use_container_width=True):
            st.session_state["show_history_panel"] = True
            st.rerun()
