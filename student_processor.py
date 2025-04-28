import os
import fitz  # PyMuPDF
import re
import sqlite3
import json
from datetime import datetime
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StudentAnswerProcessor:
    def __init__(self, db_path: str = "./data/main.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self._create_tables()
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _create_tables(self):
        """Create necessary tables in the database."""
        try:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                student_id TEXT UNIQUE,
                name TEXT,
                created_at TEXT
            )
            ''')
            self.cursor.execute('''
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
            self.conn.commit()
            logger.info("Created student tables")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def _register_student(self, student_id: str, student_name: Optional[str] = None):
        """Register a student in the database if not already registered."""
        try:
            self.cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
            if not self.cursor.fetchone():
                self.cursor.execute('''
                INSERT INTO students (student_id, name, created_at)
                VALUES (?, ?, ?)
                ''', (student_id, student_name or student_id, datetime.now().isoformat()))
                self.conn.commit()
                logger.info(f"Registered student {student_id}")
        except sqlite3.Error as e:
            logger.error(f"Error registering student {student_id}: {e}")
            raise

    def process_student_paper(
        self,
        pdf_path: str,
        student_id: str,
        student_name: Optional[str] = None,
        questions: Optional[List[Dict]] = None
    ) -> Dict:
        """Process a student's answer paper PDF."""
        try:
            self._register_student(student_id, student_name)
            doc = fitz.open(pdf_path)
            answers_by_qnum = {}
            current_question = None
            current_text = []
            question_page_map = {}

            q_patterns = [
                r'^Q\.?\s*(\d+)[\.:]',  # Q.1:
                r'^Question\s*(\d+)[\.:]',  # Question 1:
                r'^(\d+)\.\s',  # 1. 
                r'^(\d+)\)',  # 1)
                r'^(\d+)[a-d][\.\)]'  # 1a. or 1a)
            ]

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    new_question = None
                    for pattern in q_patterns:
                        match = re.match(pattern, line)
                        if match:
                            new_question = match.group(1)
                            break
                    if new_question:
                        if current_question and current_text:
                            answers_by_qnum[current_question] = '\n'.join(current_text)
                        current_question = new_question
                        current_text = [line]
                        if current_question not in question_page_map:
                            question_page_map[current_question] = []
                        question_page_map[current_question].append(page_num)
                    elif current_question:
                        current_text.append(line)
                        if page_num not in question_page_map.get(current_question, []):
                            question_page_map[current_question].append(page_num)

            if current_question and current_text:
                answers_by_qnum[current_question] = '\n'.join(current_text)

            doc.close()
            answers_by_id = {}
            if questions:
                q_num_to_id = {str(q['question_number']): q['id'] for q in questions}
                for q_num, answer in answers_by_qnum.items():
                    if q_num in q_num_to_id:
                        q_id = q_num_to_id[q_num]
                        answers_by_id[q_id] = answer
                        try:
                            self.cursor.execute('''
                            INSERT OR REPLACE INTO student_answers 
                            (student_id, question_id, answer_text, page_numbers, processed_at)
                            VALUES (?, ?, ?, ?, ?)
                            ''', (
                                student_id,
                                q_id,
                                answer,
                                json.dumps(question_page_map.get(q_num, [])),
                                datetime.now().isoformat()
                            ))
                            self.conn.commit()
                        except sqlite3.Error as e:
                            logger.error(f"Error storing answer for {student_id}, question {q_id}: {e}")
            else:
                for q_num, answer in answers_by_qnum.items():
                    answers_by_id[q_num] = answer
                    try:
                        self.cursor.execute('''
                        INSERT OR REPLACE INTO student_answers 
                        (student_id, question_id, answer_text, page_numbers, processed_at)
                        VALUES (?, ?, ?, ?, ?)
                        ''', (
                            student_id,
                            q_num,
                            answer,
                            json.dumps(question_page_map.get(q_num, [])),
                            datetime.now().isoformat()
                        ))
                        self.conn.commit()
                    except sqlite3.Error as e:
                        logger.error(f"Error storing answer for {student_id}, question {q_num}: {e}")

            logger.info(f"Processed student paper for {student_id}")
            return answers_by_id
        except Exception as e:
            logger.error(f"Error processing student paper {pdf_path}: {e}")
            return {}

    def get_student_answer(self, student_id: str, question_id: int) -> Optional[str]:
        """Retrieve a student's answer for a specific question."""
        try:
            self.cursor.execute('''
            SELECT answer_text FROM student_answers 
            WHERE student_id = ? AND question_id = ?
            ''', (student_id, question_id))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving answer for {student_id}, question {question_id}: {e}")
            return None

    def get_student_answers(self, student_id: str) -> Dict[str, str]:
        """Retrieve all answers for a student."""
        try:
            self.cursor.execute('''
            SELECT question_id, answer_text FROM student_answers 
            WHERE student_id = ?
            ''', (student_id,))
            return {str(row[0]): row[1] for row in self.cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error retrieving answers for {student_id}: {e}")
            return {}

    def get_all_students(self) -> List[tuple]:
        """Get list of all registered students."""
        try:
            self.cursor.execute('SELECT student_id, name FROM students')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving students: {e}")
            return []

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            logger.info("Closed student processor database connection")
        except Exception as e:
            logger.error(f"Error closing database: {e}")