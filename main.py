import glob
import json
import sqlite3
from document_processor import DocumentProcessor
from student_processor import StudentAnswerProcessor
from answer_generator import AnswerGenerator
from evaluator import AnswerEvaluator
from feedback_generator import FeedbackGenerator
from db_utils import DBUtils
import logging
from typing import List, Dict


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_student_pdf(pdf_path: str, student_processor: StudentAnswerProcessor, questions: List[Dict]):
    """Process a single student PDF."""
    try:
        student_id = pdf_path.split("/")[-1].replace(".pdf", "")
        student_processor.process_student_paper(pdf_path, student_id, questions=questions)
        logger.info(f"Processed student PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")

def main():
    """Run the full exam correction pipeline."""
    DB_PATH = "./data/main.db"
    try:
        # Initialize database
        DBUtils.initialize_database(DB_PATH)

        # Load questions
        questions_path = "./data/questions.json"
        with open(questions_path, "r") as f:
            questions = json.load(f)
        DBUtils.load_questions(DB_PATH, questions)
        logger.info("Loaded questions")

        # Initialize components
        doc_processor = DocumentProcessor()
        student_processor = StudentAnswerProcessor()
        answer_generator = AnswerGenerator()
        evaluator = AnswerEvaluator()
        feedback_generator = FeedbackGenerator()

        # Process textbooks
        pdf_files = glob.glob("./data/textbooks/*.pdf")
        if pdf_files:
            doc_processor.extract_text_from_pdfs(pdf_files)
            doc_processor.chunk_documents()
            doc_processor.build_faiss_index()
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
        for q in questions:
            answer_generator.generate_model_answer(
                str(q["id"]), q["question_text"], q["marks"], q["co_id"], q["rubric"]
            )
        logger.info("Generated model answers")

        # Process student answers sequentially
        answer_pdfs = glob.glob("./data/answer_sheets/*.pdf")
        if answer_pdfs:
            for pdf in answer_pdfs:
                process_student_pdf(pdf, student_processor, questions)
            logger.info("Processed all student answers")

        # Evaluate answers
        students = student_processor.get_all_students()
        for student_id, _ in students:
            for q in questions:
                evaluator.evaluate_answer(student_id, q["id"], q)
        logger.info("Evaluated all answers")

        # Generate feedback
        report = feedback_generator.generate_instructor_report()
        logger.info("Generated instructor report")

        # Clean up
        doc_processor.close()
        student_processor.close()
        answer_generator.close()
        evaluator.close()
        feedback_generator.close()
        logger.info("Pipeline completed successfully")



    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

if __name__ == "__main__":
    main()