import os
import sys
import json
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from document_processor import DocumentProcessor
from answer_generator import AnswerGenerator
from student_processor import StudentProcessor
from evaluator import Evaluator
from feedback_generator import FeedbackGenerator

#–– Configuration ––#
st.set_page_config(page_title="Assessment Evaluation System", page_icon="📝", layout="wide")

TEXTBOOK_DIR = "textbooks"
STUDENT_DIR = "student_answers"
DB_PATH = "evaluation_results.db"

for d in [TEXTBOOK_DIR, STUDENT_DIR]:
    os.makedirs(d, exist_ok=True)

#–– Helpers ––#
def get_evaluation_stats():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    overall = pd.read_sql(
        "SELECT AVG(similarity) as avg_similarity, AVG(coverage) as avg_coverage, "
        "SUM(plagiarism_flag) as plagiarism_flags FROM evaluations",
        conn
    )
    questions = pd.read_sql(
        "SELECT question_id, AVG(similarity) as avg_similarity, AVG(coverage) as avg_coverage, "
        "COUNT(*) as num_responses FROM evaluations GROUP BY question_id",
        conn
    )
    students = pd.read_sql(
        "SELECT student_id, AVG(similarity) as avg_similarity, AVG(coverage) as avg_coverage, "
        "SUM(plagiarism_flag) as plagiarism_flags FROM evaluations GROUP BY student_id",
        conn
    )
    conn.close()
    return {"overall": overall.iloc[0].to_dict(), "questions": questions, "students": students}

def display_sidebar():
    st.sidebar.title("Assessment System")
    page = st.sidebar.radio("Navigation", [
        "Dashboard",
        "Upload Syllabus",
        "Upload Questions",
        "Process Documents",
        "Upload Student Answers",
        "Generate & View Results"
    ])
    st.sidebar.markdown("---")
    # Status
    st.sidebar.write(f"{'✅' if os.path.exists('syllabus.json') else '❌'} syllabus.json")
    st.sidebar.write(f"{'✅' if os.path.exists('questions.json') else '❌'} questions.json")
    st.sidebar.write(f"{'✅' if any(Path(TEXTBOOK_DIR).glob('*.pdf')) else '❌'} textbooks")
    st.sidebar.write(f"{'✅' if any(Path(STUDENT_DIR).glob('*.pdf')) else '❌'} student answers")
    st.sidebar.write(f"{'✅' if os.path.exists('model_answers.json') else '❌'} model_answers.json")
    st.sidebar.write(f"{'✅' if os.path.exists(DB_PATH) else '❌'} evaluation DB")
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 Assessment System")
    return page

#–– Pages ––#
def show_dashboard():
    st.title("Dashboard")
    stats = get_evaluation_stats()
    if stats and stats["overall"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Similarity", f"{stats['overall']['avg_similarity']:.2%}")
        c2.metric("Avg Coverage", f"{stats['overall']['avg_coverage']:.2%}")
        c3.metric("Plagiarism Flags", stats['overall']['plagiarism_flags'])
        st.subheader("By Question")
        if not stats["questions"].empty:
            fig = px.bar(
                stats["questions"],
                x="question_id",
                y=["avg_similarity", "avg_coverage"],
                barmode="group",
                title="Avg Scores per Question"
            )
            st.plotly_chart(fig)
        st.subheader("Top 5 Students")
        if not stats["students"].empty:
            top5 = stats["students"].sort_values("avg_similarity", ascending=False).head(5)
            fig2 = px.bar(top5, x="student_id", y="avg_similarity", title="Top 5 by Similarity")
            st.plotly_chart(fig2)
    else:
        st.info("No evaluation data available.")

def upload_syllabus():
    st.title("Upload Syllabus JSON")
    st.write("Upload your complete `syllabus.json` (with Course Information, Course Outcomes, Unit Entries).")
    uploaded = st.file_uploader("Select syllabus.json", type="json")
    if uploaded:
        try:
            data = json.load(uploaded)
            with open("syllabus.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            st.success("syllabus.json uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

def upload_questions():
    st.title("Upload Questions JSON")
    st.write("Upload your complete `questions.json` (with a top-level `questions` array).")
    uploaded = st.file_uploader("Select questions.json", type="json")
    if uploaded:
        try:
            data = json.load(uploaded)
            if "questions" not in data or not isinstance(data["questions"], list):
                st.error("Invalid structure: top-level `questions` array missing.")
                return
            with open("questions.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            st.success("questions.json uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

def process_documents():
    st.title("Process Textbooks")
    st.subheader("Uploaded PDFs")
    books = list(Path(TEXTBOOK_DIR).glob("*.pdf"))
    for b in books:
        st.write(f"- {b.name}")
    uploaded = st.file_uploader("Upload PDF textbooks", accept_multiple_files=True, type="pdf")
    if uploaded:
        for file in uploaded:
            path = os.path.join(TEXTBOOK_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Uploaded {file.name}")
    if books and st.button("Build FAISS Index"):
        with st.spinner("Indexing textbooks…"):
            dp = DocumentProcessor(textbook_dir=TEXTBOOK_DIR)
            dp.build_index()
        st.success("FAISS index built.")

def upload_student_answers():
    st.title("Upload Student Answers")
    st.subheader("Uploaded PDFs")
    subs = list(Path(STUDENT_DIR).glob("*.pdf"))
    for s in subs:
        st.write(f"- {s.name}")
    uploaded = st.file_uploader("Upload student PDFs", accept_multiple_files=True, type="pdf")
    if uploaded:
        for file in uploaded:
            path = os.path.join(STUDENT_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Uploaded {file.name}")
    if subs and st.button("Process Answers"):
        with st.spinner("Extracting text from student PDFs…"):
            sp = StudentProcessor()
            sp.process_directory(STUDENT_DIR)
        st.success("student_answers.json generated.")

def generate_view_results():
    st.title("Generate & View Results")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate Model Answers"):
            with st.spinner("Generating answers…"):
                ag = AnswerGenerator()
                ag.generate_all("questions.json")
            st.success("model_answers.json saved.")
    with c2:
        if st.button("Evaluate Student Answers"):
            with st.spinner("Evaluating…"):
                ev = Evaluator()
                ev.evaluate("student_answers.json", "model_answers.json")
            st.success("evaluation_results.db updated.")

    if st.button("Generate Feedback"):
        with st.spinner("Compiling feedback…"):
            fg = FeedbackGenerator()
            fg.generate_feedback()
        st.success("Feedback JSON files created.")

    # Show a quick summary
    stats = get_evaluation_stats()
    if stats and stats["overall"]:
        st.subheader("Summary Metrics")
        st.write(stats["overall"])

#–– Main ––#
page = display_sidebar()

if page == "Dashboard":
    show_dashboard()
elif page == "Upload Syllabus":
    upload_syllabus()
elif page == "Upload Questions":
    upload_questions()
elif page == "Process Documents":
    process_documents()
elif page == "Upload Student Answers":
    upload_student_answers()
elif page == "Generate & View Results":
    generate_view_results()
