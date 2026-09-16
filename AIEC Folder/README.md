# AIEC — AI Exam Creator & Practice Platform

AIEC is a Streamlit web application powered by **Google Gemini API (`gemini-3.6-flash`)** that automatically generates balanced exam papers from coursework notes, past papers, and mark schemes.

## 🚀 Features

- **10 Question Types Supported**:
  - Multiple Choice (`mcq`)
  - Short Answer (`short_answer`)
  - Extended Essay (`essay`)
  - Matching / Draw-a-Line (`matching`)
  - Fill-in-the-Blanks (`fill_blank`)
  - True / False (`true_false`)
  - Ordering / Sequencing (`ordering`)
  - Categorization (`categorization`)
  - Diagram Labeling (`labeling`)
  - Numerical Calculation (`calculation`)
- **Teacher Mode**: Includes full mark scheme tables, model answers, and inline scoring keys.
- **Interactive Practice & AI Auto-Grading**: Students can complete exams directly online and receive instant AI feedback and score breakdowns.
- **PDF Export Engine**: Generate printable student exam PDFs or full mark scheme PDFs.
- **Persistent Sidebar Exam History**: Save, load, and manage past exams across browser sessions.

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-github-repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run AIEC.py
   ```

## 📦 Requirements

- Python 3.9+
- Packages listed in `requirements.txt`:
  - `streamlit`
  - `google-genai`
  - `pydantic`
  - `fpdf2`
  - `pandas`
