# components/student_processor.py (continued)
import os
import fitz  # PyMuPDF
import re
import sqlite3
import json
import numpy as np
from datetime import datetime

class StudentAnswerProcessor:
    def __init__(self, student_db_path="./data/student_answers.db"):
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(student_db_path), exist_ok=True)
        
        # Connect to the student answers database
        self.conn = sqlite3.connect(student_db_path)
        self.cursor = self.conn.cursor()
        
        # Create necessary tables
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables in the student answers database"""
        # Student table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            student_id TEXT UNIQUE,
            name TEXT,
            created_at TEXT
        )
        ''')
        
        # Student answers table
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
    
    def _register_student(self, student_id, student_name=None):
        """Register a student in the database if not already registered"""
        self.cursor.execute(
            'SELECT id FROM students WHERE student_id = ?', 
            (student_id,)
        )
        existing = self.cursor.fetchone()
        
        if not existing:
            self.cursor.execute('''
            INSERT INTO students (student_id, name, created_at)
            VALUES (?, ?, ?)
            ''', (
                student_id,
                student_name or student_id,
                datetime.now().isoformat()
            ))
            self.conn.commit()
    
    def process_student_paper(self, pdf_path, student_id, student_name=None, questions=None):
        """
        Process a student's answer paper PDF
        
        Args:
            pdf_path: Path to the PDF file
            student_id: Student ID
            student_name: Student name (optional)
            questions: List of question dictionaries with question_number and question_id
        
        Returns:
            Dictionary mapping question IDs to extracted answers
        """
        # Register the student if not already registered
        self._register_student(student_id, student_name)
        
        # Extract text from PDF
        doc = fitz.open(pdf_path)
        
        # Dictionary to store answers by question number
        answers_by_qnum = {}
        current_question = None
        current_text = []
        question_page_map = {}
        
        # Regular expressions for question number detection
        # This is a simplified version - would need customization for real papers
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
            
            # Look for question numbers at the start of paragraphs
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new question
                new_question = None
                for pattern in q_patterns:
                    match = re.match(pattern, line)
                    if match:
                        new_question = match.group(1)
                        break
                
                if new_question:
                    # Save current question text if any
                    if current_question and current_text:
                        answers_by_qnum[current_question] = '\n'.join(current_text)
                    
                    # Start new question
                    current_question = new_question
                    current_text = [line]
                    
                    # Record page number
                    if current_question not in question_page_map:
                        question_page_map[current_question] = []
                    question_page_map[current_question].append(page_num)
                    
                elif current_question:
                    # Continue with current question
                    current_text.append(line)
                    
                    # Update page numbers
                    if page_num not in question_page_map.get(current_question, []):
                        question_page_map[current_question].append(page_num)
        
        # Save the last question's text
        if current_question and current_text:
            answers_by_qnum[current_question] = '\n'.join(current_text)
        
        doc.close()
        
        # Map question numbers to question IDs if questions provided
        answers_by_id = {}
        if questions:
            q_num_to_id = {str(q['question_number']): q['id'] for q in questions}
            
            for q_num, answer in answers_by_qnum.items():
                if q_num in q_num_to_id:
                    q_id = q_num_to_id[q_num]
                    answers_by_id[q_id] = answer
                    
                    # Save to database
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
        else:
            # If no questions provided, just use question numbers as IDs
            for q_num, answer in answers_by_qnum.items():
                answers_by_id[q_num] = answer
                
                # Save to database with question number as ID
                self.cursor.execute('''
                INSERT OR REPLACE INTO student_answers 
                (student_id, question_id, answer_text, page_numbers, processed_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    student_id,
                    q_num,  # Using question number as ID
                    answer,
                    json.dumps(question_page_map.get(q_num, [])),
                    datetime.now().isoformat()
                ))
        
        self.conn.commit()
        return answers_by_id
    
    def get_student_answer(self, student_id, question_id):
        """Retrieve a student's answer for a specific question"""
        self.cursor.execute('''
        SELECT answer_text FROM student_answers 
        WHERE student_id = ? AND question_id = ?
        ''', (student_id, question_id))
        
        result = self.cursor.fetchone()
        if result:
            return result[0]
        return None
    
    def get_student_answers(self, student_id):
        """Retrieve all answers for a student"""
        self.cursor.execute('''
        SELECT question_id, answer_text FROM student_answers 
        WHERE student_id = ?
        ''', (student_id,))
        
        return {str(row[0]): row[1] for row in self.cursor.fetchall()}
    
    def get_all_students(self):
        """Get list of all registered students"""
        self.cursor.execute('SELECT student_id, name FROM students')
        return self.cursor.fetchall()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()