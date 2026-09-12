import streamlit as st
import tempfile
import os
import uuid
import json
import datetime
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from fpdf import FPDF

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & SECRETS
# ══════════════════════════════════════════════════════════════════════════════
APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "logo.png"
HAS_LOGO = LOGO_PATH.exists()

st.set_page_config(
    layout="wide", 
    page_title="AIEC — AI Exam Creator", 
    page_icon=str(LOGO_PATH) if HAS_LOGO else "📝"
)

# Safely retrieve the API key string from .streamlit/secrets.toml
API_KEY = st.secrets["GEMINI_API_KEY"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

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
    label_prompts: Optional[List[str]] = Field(default=None, description="List of label prompts/keys (e.g. ['Part A', 'Part B']) for labeling questions.")
    expected_units: Optional[str] = Field(default=None, description="Expected unit of measurement for calculation questions, e.g. 'm/s^2' or 'Joules'.")
    
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

class GradingResponse(BaseModel):
    graded_questions: List[GradedQuestion]


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

HISTORY_FILE = APP_DIR / "aiec_exam_history.json"

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


# ══════════════════════════════════════════════════════════════════════════════
# PDF HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '—': '-', '–': '-', '…': '...', '•': '*'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
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


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### 🎭 Mode")
teacher_mode = st.sidebar.toggle("Teacher Mode", value=False, key="sidebar_teacher_mode", help="Shows full mark schemes, inline correct answers, and criteria.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Paper Settings")

diff_auto = st.sidebar.checkbox("Auto Difficulty (AI Decides)", value=True, key="sidebar_diff_auto")
if diff_auto:
    selected_difficulty = "Auto"
else:
    selected_difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], key="sidebar_diff_select")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Question Counts & Types")
q_count_auto = st.sidebar.checkbox("Auto Question Mix (AI Decides)", value=True, key="sidebar_q_count_auto")

if q_count_auto:
    n_mcq = 0
    n_short = 0
    n_essay = 0
    n_matching = 0
    n_true_false = 0
    n_other = 0
else:
    n_mcq = st.sidebar.slider("Multiple Choice (MCQ)", 0, 15, 3, key="sidebar_n_mcq")
    n_short = st.sidebar.slider("Short Answer", 0, 10, 3, key="sidebar_n_short")
    n_essay = st.sidebar.slider("Extended Essay", 0, 5, 1, key="sidebar_n_essay")
    n_matching = st.sidebar.slider("Matching (Draw Line)", 0, 5, 1, key="sidebar_n_matching")
    n_true_false = st.sidebar.slider("True / False", 0, 10, 2, key="sidebar_n_tf")
    n_other = st.sidebar.slider("Other (Ordering/Calc/Labeling)", 0, 5, 1, key="sidebar_n_other")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR HISTORY
# ══════════════════════════════════════════════════════════════════════════════

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
                    st.rerun()
            with c_del:
                if st.button("Delete", key=f"sb_del_{h_id}", use_container_width=True):
                    delete_history_entry(h_id)
                    if st.session_state.get("active_exam_id") == h_id:
                        st.session_state["exam_paper"] = None
                        st.session_state["grading_result"] = None
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INPUT FORM & CREATION UI
# ══════════════════════════════════════════════════════════════════════════════

has_active_exam = "exam_paper" in st.session_state and st.session_state["exam_paper"] is not None

if not has_active_exam: 
    if HAS_LOGO:
        st.image(str(LOGO_PATH), width=700)
    st.title("AIEC — AI Exam Creator")
    st.caption("Generate a full exam paper from your course material, then practice, grade, and export it.")
    
    with st.container(border=True):
        st.write("Type your prompt / instructions here:")
        input1 = st.text_area("Prompt Instructions", key="extra_info_prompt", height=100, label_visibility="hidden")

    Col2, Col3, Col4 = st.columns(3)

    with Col2:
        with st.container(border=True):
            st.write("Upload your course work (SoW, Notes, Images, etc)")
            file1 = st.file_uploader("Course work", label_visibility="hidden", key="file1_coursework", accept_multiple_files=True, max_upload_size=100000)
    with Col3:
        with st.container(border=True):
            st.write("Upload your mark scheme for each past paper.")
            file3 = st.file_uploader("Mark scheme", label_visibility="hidden", key="file3_markscheme", accept_multiple_files=True, max_upload_size=100000)
    with Col4:
        with st.container(border=True):
            st.write("Upload your past papers here for structure and layout.")
            file2 = st.file_uploader("Past papers", label_visibility="hidden", key="file2_pastpapers", accept_multiple_files=True, max_upload_size=100000)

    if st.button("Generate Exam Paper", key="btn_generate_exam", use_container_width=True, type="primary"):
            client = genai.Client(api_key=API_KEY)
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
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ExamPaper,
                        ),
                    )
                    exam = json.loads(response.text)
                    st.session_state["exam_paper"] = exam
                    st.session_state["exam_answers"] = {}
                    st.session_state["grading_result"] = None
                    e_id = save_to_history(exam)
                    st.session_state["active_exam_id"] = e_id
                    st.success("Exam paper generated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating exam paper: {e}")
else:
    c_hdr1, c_hdr2 = st.columns([4, 1])
    with c_hdr1:
        st.caption("Active Exam Mode")
    with c_hdr2:
        if st.button("➕ Create New Exam", key="btn_create_new", use_container_width=True):
            st.session_state["exam_paper"] = None
            st.session_state["grading_result"] = None
            st.session_state["active_exam_id"] = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EXAM PAPER DISPLAY & INTERACTIVE PRACTICE MODE
# ══════════════════════════════════════════════════════════════════════════════

if "exam_paper" in st.session_state and st.session_state["exam_paper"]:
    exam = st.session_state["exam_paper"]
    questions = exam.get("questions", [])
    
    st.markdown("---")
    st.header(exam.get("title", "Exam Paper"))
    if exam.get("instructions"):
        st.info(f"**Instructions:** {exam['instructions']}")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download Printable Exam (PDF)",
            data=build_pdf(exam, include_answers=False),
            file_name="exam_paper.pdf",
            mime="application/pdf",
            key="dl_pdf_exam",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            "📥 Download Exam + Mark Scheme (PDF)",
            data=build_pdf(exam, include_answers=True),
            file_name="exam_paper_with_answers.pdf",
            mime="application/pdf",
            key="dl_pdf_answers",
            use_container_width=True
        )

    if teacher_mode:
        with st.expander("📋 Full Mark Scheme & Answer Key (Teacher View)", expanded=True):
            import pandas as pd
            ms_rows = [
                {
                    "Q": i+1,
                    "Topic": q.get("topic", "—"),
                    "Type": q.get("question_type", "").upper(),
                    "Marks": q.get("marks", 0),
                    "Answer / Criteria": q.get("correct_answer", "")
                }
                for i, q in enumerate(questions)
            ]
            st.dataframe(pd.DataFrame(ms_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📝 Practice & Answer Workspace")

    if "exam_answers" not in st.session_state:
        st.session_state["exam_answers"] = {}

    for i, q in enumerate(questions):
        qtype = q.get("question_type", "written")
        marks = q.get("marks", 2)
        topic = q.get("topic", "")
        topic_badge = f"`{topic}`" if topic else ""

        with st.container(border=True):
            st.markdown(f"**Q{i+1}.** {q.get('question_text', '')}  {topic_badge} `({marks} marks)`")

            if teacher_mode:
                st.success(f"💡 **Teacher Key / Criteria:** {q.get('correct_answer', 'N/A')}")

            # 1. MCQ
            if qtype == "mcq" and q.get("options"):
                opts = q["options"]
                st.session_state["exam_answers"][i] = st.radio(
                    f"Select Answer for Q{i+1}:",
                    opts,
                    key=f"ans_{i}",
                    index=None
                )

            # 2. Matching
            elif qtype == "matching" and q.get("left_items") and q.get("right_items"):
                st.write("🔗 **Match each item from Left to Right:**")
                lefts = q["left_items"]
                rights = ["-- Select Match --"] + q["right_items"]
                user_matches = {}
                for l_idx, left_item in enumerate(lefts):
                    m_col1, m_col2 = st.columns([1, 1])
                    with m_col1:
                        st.markdown(f"**{left_item}**")
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

            # 3. True / False
            elif qtype == "true_false":
                tf_choice = st.radio(
                    f"Statement is True or False?",
                    ["True", "False"],
                    key=f"ans_tf_{i}",
                    index=None,
                    horizontal=True
                )
                tf_reason = st.text_input("Justification (optional/if false):", key=f"ans_tf_reason_{i}")
                st.session_state["exam_answers"][i] = f"Choice: {tf_choice} | Reasoning: {tf_reason}"

            # 4. Ordering / Sequencing
            elif qtype == "ordering" and q.get("items"):
                st.write("🔢 **Assign the correct step position (1 to N) for each item:**")
                items = q["items"]
                positions = list(range(1, len(items) + 1))
                user_order = {}
                for it_idx, item in enumerate(items):
                    o_col1, o_col2 = st.columns([3, 1])
                    with o_col1:
                        st.write(item)
                    with o_col2:
                        pos = st.selectbox(
                            f"Position for item {it_idx}",
                            positions,
                            key=f"order_{i}_{it_idx}",
                            label_visibility="collapsed"
                        )
                        user_order[item] = pos
                st.session_state["exam_answers"][i] = json.dumps(user_order)

            # 5. Categorization / Sorting
            elif qtype == "categorization" and q.get("categories") and q.get("items"):
                st.write("🏷️ **Assign each item to its correct Category:**")
                cats = q["categories"]
                items = q["items"]
                user_cats = {}
                for it_idx, item in enumerate(items):
                    c_col1, c_col2 = st.columns([2, 1])
                    with c_col1:
                        st.write(item)
                    with c_col2:
                        cat_sel = st.selectbox(
                            f"Category for item {it_idx}",
                            cats,
                            key=f"cat_{i}_{it_idx}",
                            label_visibility="collapsed"
                        )
                        user_cats[item] = cat_sel
                st.session_state["exam_answers"][i] = json.dumps(user_cats)

            # 6. Diagram Labeling
            elif qtype == "labeling" and q.get("label_prompts"):
                st.write("🏷️ **Provide labels for each key/part:**")
                prompts = q["label_prompts"]
                user_labels = {}
                for l_idx, lbl in enumerate(prompts):
                    val = st.text_input(f"Label for '{lbl}':", key=f"lbl_{i}_{l_idx}")
                    user_labels[lbl] = val
                st.session_state["exam_answers"][i] = json.dumps(user_labels)

            # 7. Calculation
            elif qtype == "calculation":
                unit_str = f" ({q['expected_units']})" if q.get("expected_units") else ""
                ans_val = st.text_input(f"Final Answer{unit_str}:", key=f"calc_ans_{i}")
                working = st.text_area("Working / Steps:", key=f"calc_work_{i}", height=80)
                st.session_state["exam_answers"][i] = f"Answer: {ans_val} {unit_str} | Working: {working}"

            # 8. Extended Essay
            elif qtype == "essay":
                ans_text = st.text_area("Write your essay response:", key=f"ans_essay_{i}", height=180)
                st.session_state["exam_answers"][i] = ans_text

            # 9. Short Answer / Default Written
            else:
                ans_text = st.text_area("Type your answer:", key=f"ans_written_{i}", height=90)
                st.session_state["exam_answers"][i] = ans_text

            with st.popover("⚙️ Question Actions"):
                regen_inst = st.text_input("Instructions for regeneration:", key=f"regen_inst_{i}", placeholder="e.g. Make it harder")
                if st.button("🔄 Regenerate This Question", key=f"btn_regen_{i}"):
                    if not API_KEY:
                        st.error("API Key not found in .streamlit/secrets.toml.")
                    else:
                        client = genai.Client(api_key=API_KEY)
                        with st.spinner("Regenerating question..."):
                            regen_prompt = (
                                f"Regenerate question Q{i+1} from this exam. "
                                f"Existing question: {json.dumps(q)}. "
                                f"User instructions: {regen_inst if regen_inst else 'Provide a fresh alternative question on the same topic'}. "
                                f"Return JSON matching Question schema."
                            )
                            try:
                                resp = client.models.generate_content(
                                    model='gemini-2.5-flash',
                                    contents=regen_prompt,
                                    config=types.GenerateContentConfig(
                                        response_mime_type="application/json",
                                        response_schema=Question,
                                    ),
                                )
                                new_q = json.loads(resp.text)
                                exam["questions"][i] = new_q
                                st.session_state["exam_paper"] = exam
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed to regenerate: {ex}")

    # ══════════════════════════════════════════════════════════════════════════════
    # SUBMIT & GRADE EXAM
    # ══════════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    if st.button("📊 Submit & Grade Exam Paper", key="btn_submit_grade", use_container_width=True, type="primary"):
        if not API_KEY:
            st.error("API Key not found. Please add GEMINI_API_KEY to your .streamlit/secrets.toml file.")
        else:
            client = genai.Client(api_key=API_KEY)
            answers = st.session_state.get("exam_answers", {})
            
            grade_prompt = (
                "You are an impartial academic examiner. Grade the student's exam submission against the original mark scheme and correct answers.\n\n"
                f"Exam Title: {exam.get('title')}\n"
                f"Questions and Criteria:\n{json.dumps(exam.get('questions'), indent=2)}\n\n"
                f"Student Answers:\n{json.dumps(answers, indent=2)}\n\n"
                "Return a GradedQuestion breakdown for every question with exact scores and constructive feedback."
            )

            with st.spinner("Grading your submission with Gemini..."):
                try:
                    g_resp = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=grade_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GradingResponse,
                        ),
                    )
                    graded_data = json.loads(g_resp.text)
                    st.session_state["grading_result"] = graded_data
                    active_id = st.session_state.get("active_exam_id")
                    save_to_history(exam, graded_data, entry_id=active_id)
                    st.success("Exam successfully graded!")
                except Exception as ex:
                    st.error(f"Error grading exam paper: {ex}")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS & PERFORMANCE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

if "grading_result" in st.session_state and st.session_state["grading_result"]:
    g_res = st.session_state["grading_result"]
    graded_qs = g_res.get("graded_questions", [])
    questions = st.session_state["exam_paper"].get("questions", [])

    st.markdown("---")
    st.header("🏆 Performance Results & Marking Report")

    total_awarded = sum(g.get("score", 0) for g in graded_qs)
    total_possible = sum(q.get("marks", 0) for q in questions)
    pct = round((total_awarded / total_possible * 100), 1) if total_possible > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Score", f"{total_awarded} / {total_possible}")
    m2.metric("Percentage", f"{pct}%")
    if pct >= 80:
        grade_str = "A* / Excellent"
    elif pct >= 70:
        grade_str = "A / Great"
    elif pct >= 60:
        grade_str = "B / Good"
    elif pct >= 50:
        grade_str = "C / Pass"
    else:
        grade_str = "Needs Revision"
    m3.metric("Overall Grade", grade_str)

    st.subheader("Detailed Feedback per Question")

    graded_map = {g.get("question_index"): g for g in graded_qs}

    for i, q in enumerate(questions):
        g_info = graded_map.get(i, {})
        q_score = g_info.get("score", 0)
        q_max = q.get("marks", 0)
        q_pct = round((q_score / q_max * 100), 1) if q_max > 0 else 0
        topic = q.get("topic", "General")
        qtype = q.get("question_type", "written").upper()

        if q_pct >= 80:
            badge_color = "🟢"
        elif q_pct >= 50:
            badge_color = "🟡"
        else:
            badge_color = "🔴"

        with st.expander(f"Q{i+1} [{qtype}] `{topic}` — {q_score}/{q_max} marks ({q_pct}%) {badge_color}"):
            st.markdown(f"**Question:** {q.get('question_text')}")
            st.markdown(f"**Correct Answer / Criteria:** {q.get('correct_answer')}")
            st.markdown(f"**Your Answer:** {st.session_state.get('exam_answers', {}).get(i, '*No Answer Provided*')}")
            st.info(f"**Feedback:** {g_info.get('feedback', 'No detailed feedback.')}")
