import sqlite3
import logging
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)     

class DBUtils:
    @staticmethod
    def initialize_database(db_path: str):
        """Initialize the main database with all required tables."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Textbook chunks table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS textbook_chunks (
                id INTEGER PRIMARY KEY,
                book_title TEXT,
                chunk_text TEXT,
                page_number INTEGER
            )
            ''')
            
            # Questions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                question_number INTEGER,
                question_text TEXT,
                marks INTEGER,
                rubric TEXT,
                co_id TEXT
            )
            ''')
            
            # Students table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                student_id TEXT UNIQUE,
                name TEXT,
                created_at TEXT
            )
            ''')
            
            # Student answers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_answers (
                id INTEGER PRIMARY KEY,
                student_id TEXT,
                question_id INTEGER,
                answer_text TEXT,
                page_numbers TEXT,
                processed_at TEXT,
                UNIQUE(student_id, question_id)
            )
            ''')
            
            # Model answers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_answers (
                question_id TEXT PRIMARY KEY,
                model_answer TEXT,
                word_count INTEGER,
                created_at TEXT
            )
            ''')
            
            # Evaluations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY,
                student_id TEXT,
                question_id INTEGER,
                marks REAL,
                max_marks INTEGER,
                similarity_score REAL,
                rubric_coverage TEXT,
                plagiarism_score REAL,
                plagiarism_source TEXT,
                feedback TEXT,
                evaluated_at TEXT,
                UNIQUE(student_id, question_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Initialized database at {db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise

    @staticmethod
    def load_questions(db_path: str, questions: List[Dict]):
        """Load questions into the database."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for q in questions:
                cursor.execute('''
                INSERT OR REPLACE INTO questions 
                (id, question_number, question_text, marks, rubric, co_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    q['id'],
                    q['question_number'],
                    q['question_text'],
                    q['marks'],
                    json.dumps(q['rubric']),
                    q['co_id']
                ))
            conn.commit()
            conn.close()
            logger.info(f"Loaded {len(questions)} questions into database")
        except sqlite3.Error as e:
            logger.error(f"Error loading questions: {e}")
            raise