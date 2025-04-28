import streamlit as st
import json
import os
import glob
from document_processor import DocumentProcessor
from student_processor import StudentAnswerProcessor
from answer_generator import AnswerGenerator
from evaluator import AnswerEvaluator
from feedback_generator import FeedbackGenerator
from db_utils import DBUtils
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.title("Descriptive Paper Evaluator")

# Initialize database
DB_PATH = "./data/main.db"
DBUtils.initialize_database(DB_PATH)

# Sidebar for file uploads
st.sidebar.header("Upload Files")
questions_file = st.sidebar.file_uploader("Upload Questions (JSON)", type="json")
pdf_files = st.sidebar.file_uploader("Upload Textbook PDFs", type="pdf", accept_multiple_files=True)
answer_pdfs = st.sidebar.file_uploader("Upload Answer Sheet PDFs", type="pdf", accept_multiple_files=True)

# Process uploaded files
if questions_file:
    try:
        questions_data = json.load(questions_file)
        DBUtils.load_questions(DB_PATH, questions_data)
        st.sidebar.success("Questions loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading questions: {e}")

# Initialize components
doc_processor = DocumentProcessor()
student_processor = StudentAnswerProcessor()
answer_generator = AnswerGenerator()
evaluator = AnswerEvaluator()
feedback_generator = FeedbackGenerator()

# Run pipeline
if st.button("Run Evaluation Pipeline"):
    with st.spinner("Processing..."):
        try:
            # Process textbooks
            if pdf_files:
                pdf_paths = []
                for pdf_file in pdf_files:
                    with open(f"./data/textbooks/{pdf_file.name}", "wb") as f:
                        f.write(pdf_file.getbuffer())
                    pdf_paths.append(f"./data/textbooks/{pdf_file.name}")
                doc_processor.extract_text_from_pdfs(pdf_paths)
                doc_processor.chunk_documents()
                doc_processor.build_faiss_index()
                # Store chunks in database
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    for i, (chunk, source) in enumerate(zip(doc_processor.text_chunks, doc_processor.chunk_sources)):
                        cursor.execute('''
                        INSERT OR REPLACE INTO textbook_chunks 
                        (id, book_title, chunk_text, page_number)
                        VALUES (?, ?, ?, ?)
                        ''', (i, source['source'], chunk, source['page']))
                    conn.commit()
                logger.info("Processed textbook PDFs")

            # Generate model answers
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, question_text, marks, co_id, rubric FROM questions')
                questions = [
                    {
                        "id": row[0],
                        "question_text": row[1],
                        "marks": row[2],
                        "co_id": row[3],
                        "rubric": json.loads(row[4])
                    } for row in cursor.fetchall()
                ]
            for q in questions:
                answer_generator.generate_model_answer(
                    str(q["id"]), q["question_text"], q["marks"], q["co_id"], q["rubric"]
                )
            logger.info("Generated model answers")

            # Process student answers
            if answer_pdfs:
                for pdf_file in answer_pdfs:
                    pdf_path = f"./data/answer_sheets/{pdf_file.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    student_id = pdf_file.name.replace(".pdf", "")
                    student_processor.process_student_paper(pdf_path, student_id, questions=questions)
                logger.info("Processed student answers")

            # Evaluate answers
            students = student_processor.get_all_students()
            for student_id, _ in students:
                for q in questions:
                    evaluator.evaluate_answer(student_id, q["id"], q)
            logger.info("Evaluated answers")

            # Generate feedback
            report = feedback_generator.generate_instructor_report()
            st.json(report)
            st.success("Evaluation complete! Report saved to ./data/reports/instructor_report.json")
        except Exception as e:
            st.error(f"Error running pipeline: {e}")
            logger.error(f"Pipeline error: {e}")

# Clean up
def cleanup():
    doc_processor.close()
    student_processor.close()
    answer_generator.close()
    evaluator.close()
    feedback_generator.close()

st.sidebar.button("Cleanup", on_click=cleanup)