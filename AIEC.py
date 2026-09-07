
import streamlit as st
import tempfile
import os
import uuid
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class Question(BaseModel):
    question_text: str = Field(description="The actual question text, including any necessary context.")
    question_type: str = Field(description="'mcq' for multiple choice or 'written' for open-ended questions.")
    options: Optional[List[str]] = Field(default=None, description="A list of options if this is an MCQ, otherwise null.")
    correct_answer: str = Field(description="The exact correct option if MCQ, or detailed marking criteria/model answer if written.")
    marks: int = Field(description="The number of marks allocated to this question.")

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

st.set_page_config(layout="wide")
st.header("AIEC, Your Exam Creator", text_alignment="center") 
st.write() 
st.write()
st.write()
st.write()


# API Key input
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

Col1, Col2, Col3 , Col4= st.columns(4)
with Col1:
    with st.container(border=True):
        st.write("Upload your cource work (SoW, Notes, Images, etc)")
        file1 = st.file_uploader("Course work", label_visibility="hidden", key="file1", max_upload_size=10240, accept_multiple_files=True)
with Col2:
    with st.container(border=True):
        st.write("Upload your past papers here for structure and layout.")
        file2 = st.file_uploader("Past papers", label_visibility="hidden", key="file2", max_upload_size=10240, accept_multiple_files=True)
with Col3:
    with st.container(border=True):
        st.write("Upload your mark scheme for each past paper.")
        file3 = st.file_uploader("Mark scheme", label_visibility="hidden", key="file3", max_upload_size=10240, accept_multiple_files=True) 
with Col4:
    with st.container(border=True):
        st.write("Type extra information you want here....") 
        input1=st.text_area("Extra Information", key="extra_info", height=150)

if st.button("Generate Exam Paper"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not file1 or not file2 or not file3:
        st.error("Please upload files in all three categories.")
    else:
        client = genai.Client(api_key=api_key)
        
        with st.spinner("Processing files and generating exam... This may take a moment."):
            try:
                # Helper function to save Streamlit UploadedFile to temp dir and upload to Gemini
                def upload_to_gemini(st_files):
                    refs = []
                    for f in st_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                            tmp.write(f.getvalue())
                            tmp_path = tmp.name
                        
                        # Upload to Gemini File API
                        gem_file = client.files.upload(file=tmp_path, config={'display_name': f.name})
                        refs.append(gem_file)
                        os.remove(tmp_path)
                    return refs

                st.info("Uploading files to Gemini...")
                gemini_files1 = upload_to_gemini(file1)
                gemini_files2 = upload_to_gemini(file2)
                gemini_files3 = upload_to_gemini(file3)
                all_files = gemini_files1 + gemini_files2 + gemini_files3 

                prompt = """You are an expert examiner. Based on the uploaded coursework, past papers, and mark schemes, generate a new past paper in JSON format.
- Use the coursework to know what content to test.
- Analyze the structure, command words, and allocated marks from the past papers and mark schemes, and mimic that structure strictly.
- Ensure the paper includes a mix of MCQs and written work exactly as formatted in the past papers.
- Provide the correct answer or detailed marking criteria for every question.
- Your output MUST strictly match the provided JSON schema."""
                if input1.strip():
                    prompt += f"\n\nAdditional instructions from the user:\n{input1}"

                st.info("Generating interactive paper...")
                response = client.models.generate_content(
                    model="gemini-3.7-flash",
                    contents=all_files + [prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.7, # Encourage variation for new papers
                        response_mime_type="application/json",
                        response_json_schema=ExamPaper.model_json_schema()
                    )
                )
                
                exam_data = json.loads(response.text)
                st.session_state['exam_data'] = exam_data
                st.session_state['paper_id'] = str(uuid.uuid4())
                
                # Clear previous results if generating a new paper
                if 'exam_results' in st.session_state:
                    del st.session_state['exam_results']
                st.success("Exam paper generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

if 'exam_data' in st.session_state:
    st.divider()
    exam = st.session_state['exam_data']
    paper_id = st.session_state.get('paper_id', 'default')
    st.title(exam.get('title', 'Generated Exam Paper'))
    
    if exam.get('instructions'):
        st.info(exam.get('instructions'))
        
    st.write("---")
    
    with st.form("exam_form"):
        for i, q in enumerate(exam.get('questions', [])):
            with st.container():
                st.markdown(f"**Q{i+1}.** {q.get('question_text')}  *(Total marks: {q.get('marks')})*")
                
                if q.get('question_type') == 'mcq' and q.get('options'):
                    st.radio("Select an option:", options=q.get('options'), key=f"q_{i}_mcq_{paper_id}", index=None)
                else:
                    st.text_area("Your Answer:", key=f"q_{i}_written_{paper_id}", height=150)
                
                st.write("") # some spacing
                
        submitted = st.form_submit_button("Submit Exam")
        
        if submitted:
            if not api_key:
                st.error("API Key is required for grading. Please enter it in the sidebar.")
            else:
                mcq_results = {}
                written_questions_to_grade = []
                
                for i, q in enumerate(exam.get('questions', [])):
                    if q.get('question_type') == 'mcq':
                        user_answer = st.session_state.get(f"q_{i}_mcq_{paper_id}")
                        correct_answer = q.get('correct_answer')
                        is_correct = (user_answer == correct_answer)
                        score = q.get('marks') if is_correct else 0
                        mcq_results[i] = {
                            'score': score,
                            'feedback': f"Correct answer is: {correct_answer}",
                            'max_marks': q.get('marks')
                        }
                    else:
                        user_answer = st.session_state.get(f"q_{i}_written_{paper_id}")
                        written_questions_to_grade.append({
                            'index': i,
                            'question': q.get('question_text'),
                            'student_answer': user_answer,
                            'marking_criteria': q.get('correct_answer'),
                            'max_marks': q.get('marks')
                        })
                
                all_results = mcq_results.copy()
                
                if written_questions_to_grade:
                    with st.spinner("AI is grading your written answers..."):
                        try:
                            grading_client = genai.Client(api_key=api_key)
                            grading_prompt = f"You are an expert examiner. Grade the following student answers based on the provided marking criteria and maximum marks. Provide a score and feedback for each.\n\n{json.dumps(written_questions_to_grade, indent=2)}"
                            
                            grade_response = grading_client.models.generate_content(
                                model="gemini-3.7-flash",
                                contents=grading_prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_json_schema=GradingResponse.model_json_schema()
                                )
                            )
                            grading_data = json.loads(grade_response.text)
                            
                            for gq in grading_data.get('graded_questions', []):
                                all_results[gq['question_index']] = {
                                    'score': gq['score'],
                                    'feedback': gq['feedback'],
                                    # Find max_marks from original request
                                    'max_marks': next((item['max_marks'] for item in written_questions_to_grade if item['index'] == gq['question_index']), 0)
                                }
                                
                            st.session_state['exam_results'] = all_results
                            st.success("Your exam has been submitted and graded!")
                        except Exception as e:
                            st.error(f"Grading failed: {e}")
                else:
                    st.session_state['exam_results'] = all_results
                    st.success("Your exam has been submitted and graded!")

if 'exam_results' in st.session_state:
    st.divider()
    st.header("Exam Results")
    
    results = st.session_state['exam_results']
    exam = st.session_state['exam_data']
    
    total_score = sum(res.get('score', 0) for res in results.values())
    total_max = sum(res.get('max_marks', 0) for res in results.values())
    
    st.subheader(f"Total Score: {total_score} / {total_max}")
    
    for i, q in enumerate(exam.get('questions', [])):
        res = results.get(i)
        if res:
            with st.expander(f"Q{i+1}: {q.get('question_text')} - Score: {res.get('score')}/{res.get('max_marks')}", expanded=True):
                st.markdown(f"**Feedback:**\n{res.get('feedback')}")