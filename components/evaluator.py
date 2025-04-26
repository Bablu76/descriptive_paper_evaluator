# components/evaluator.py
import sqlite3
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

class AnswerEvaluator:
    def __init__(self, documents_db="./data/documents.db", 
                 model_answers_db="./data/model_answers.db",
                 student_db="./data/student_answers.db",
                 evaluation_db="./data/evaluations.db"):
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to the necessary databases
        self.docs_conn = sqlite3.connect(documents_db)
        self.docs_cursor = self.docs_conn.cursor()
        
        self.model_conn = sqlite3.connect(model_answers_db)
        self.model_cursor = self.model_conn.cursor()
        
        self.student_conn = sqlite3.connect(student_db)
        self.student_cursor = self.student_conn.cursor()
        
        # Create and connect to the evaluations database
        self.eval_conn = sqlite3.connect(evaluation_db)
        self.eval_cursor = self.eval_conn.cursor()
        
        # Create necessary tables
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables in the evaluations database"""
        self.eval_cursor.execute('''
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
        self.eval_conn.commit()
    
    def _extract_rubric_points(self, rubric):
        """Extract individual rubric points from various rubric formats"""
        rubric_points = []
        
        if isinstance(rubric, str):
            # Try to parse JSON string
            try:
                rubric = json.loads(rubric)
            except:
                # If not JSON, split by newlines or bullet points
                lines = rubric.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        # Remove bullet points
                        line = re.sub(r'^[\•\-\*\>\◦\■\□\○\●\★\✓]\s*', '', line)
                        rubric_points.append(line)
                return rubric_points
        
        if isinstance(rubric, list):
            # Already a list of points
            return rubric
        
        elif isinstance(rubric, dict):
            # Extract values or key-value pairs
            for key, value in rubric.items():
                if isinstance(value, str):
                    rubric_points.append(f"{key}: {value}")
                else:
                    rubric_points.append(key)
        
        return rubric_points
    
    def _segment_answer(self, answer_text, num_segments=5):
        """Split an answer into segments for more granular evaluation"""
        # Simple splitting by sentences (approximate)
        sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        
        # If not enough sentences, just return the original text
        if len(sentences) <= num_segments:
            return sentences
        
        # Calculate segment size
        segment_size = len(sentences) // num_segments
        
        # Create segments
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(sentences)
            segment = ' '.join(sentences[start_idx:end_idx])
            segments.append(segment)
        
        return segments
    
    def _calculate_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        if not text1 or not text2:
            return 0.0
            
        # Generate embeddings
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(emb1, emb2)  # Convert distance to similarity
        
        return max(0.0, min(1.0, similarity))  # Ensure in range [0, 1]
    
    def _check_plagiarism(self, student_answer, threshold=0.95):
        """Check for plagiarism against textbooks and other students"""
        # Check against textbook content
        self.docs_cursor.execute('SELECT id, book_title, chunk_text FROM textbook_chunks')
        textbook_chunks = self.docs_cursor.fetchall()
        
        highest_similarity = 0.0
        source = None
        
        # Split answer into segments for more accurate plagiarism detection
        segments = self._segment_answer(student_answer)
        
        for segment in segments:
            # Skip very short segments
            if len(segment.split()) < 10:
                continue
                
            # Check against textbooks
            for chunk_id, book_title, chunk_text in textbook_chunks:
                similarity = self._calculate_similarity(segment, chunk_text)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    source = f"Textbook: {book_title}"
        
        # Check against other students (simplified version)
        # In a real system, this would be more comprehensive
        self.student_cursor.execute('SELECT student_id, answer_text FROM student_answers')
        other_answers = self.student_cursor.fetchall()
        
        for other_student_id, other_answer in other_answers:
            for segment in segments:
                # Skip very short segments
                if len(segment.split()) < 10:
                    continue
                
                similarity = self._calculate_similarity(segment, other_answer)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    source = f"Student: {other_student_id}"
        
        # Return plagiarism score and source if above threshold
        if highest_similarity >= threshold:
            return highest_similarity, source
        
        return 0.0, None
    
    def evaluate_answer(self, student_id, question_id, question_data=None):
        """
        Evaluate a student's answer against the model answer
        
        Args:
            student_id: Student ID
            question_id: Question ID
            question_data: Optional dictionary with question information
                          (If not provided, will be fetched from the database)
        
        Returns:
            Evaluation results dictionary
        """
        # Get question data if not provided
        if not question_data:
            self.docs_cursor.execute(
                'SELECT question_number, question_text, marks, rubric FROM questions WHERE id = ?', 
                (question_id,)
            )
            question_row = self.docs_cursor.fetchone()
            
            if not question_row:
                return {
                    "error": f"Question with ID {question_id} not found in the database"
                }
                
            question_data = {
                "id": question_id,
                "question_number": question_row[0],
                "question_text": question_row[1],
                "marks": question_row[2],
                "rubric": question_row[3]
            }
        
        # Get the model answer
        self.model_cursor.execute(
            'SELECT model_answer FROM model_answers WHERE question_id = ?',
            (question_id,)
        )
        model_row = self.model_cursor.fetchone()
        
        if not model_row:
            return {
                "error": f"No model answer found for question {question_id}"
            }
            
        model_answer = model_row[0]
        
        # Get the student's answer
        self.student_cursor.execute(
            'SELECT answer_text FROM student_answers WHERE student_id = ? AND question_id = ?',
            (student_id, question_id)
        )
        student_row = self.student_cursor.fetchone()
        
        if not student_row:
            return {
                "error": f"No answer found for student {student_id}, question {question_id}"
            }
            
        student_answer = student_row[0]
        
        # Parse the rubric
        if isinstance(question_data.get('rubric'), str):
            try:
                rubric = json.loads(question_data['rubric'])
            except:
                rubric = question_data['rubric']
        else:
            rubric = question_data.get('rubric', [])
        
        # Extract rubric points
        rubric_points = self._extract_rubric_points(rubric)
        
        # Calculate overall semantic similarity
        overall_similarity = self._calculate_similarity(student_answer, model_answer)
        
        # Calculate similarity for each rubric point
        rubric_coverage = {}
        for point in rubric_points:
            # For each rubric point, find the most similar segment in the student's answer
            segments = self._segment_answer(student_answer)
            best_similarity = 0.0
            
            for segment in segments:
                similarity = self._calculate_similarity(point, segment)
                if similarity > best_similarity:
                    best_similarity = similarity
            
            rubric_coverage[point] = best_similarity
        
        # Calculate average rubric coverage
        avg_rubric_coverage = np.mean(list(rubric_coverage.values())) if rubric_coverage else 0.0
        
        # Check for plagiarism
        plagiarism_score, plagiarism_source = self._check_plagiarism(student_answer)
        
        # Calculate final marks
        max_marks = question_data.get('marks', 10)
        
        # Simple scoring formula
        # 60% weight to overall similarity, 40% to rubric coverage
        base_score = (overall_similarity * 0.6 + avg_rubric_coverage * 0.4) * max_marks
        
        # Apply plagiarism penalty if applicable
        if plagiarism_score > 0.95:
            penalty = min(base_score * 0.5, base_score)  # Up to 50% penalty
            final_marks = base_score - penalty
        else:
            final_marks = base_score
        
        # Ensure marks are within valid range
        final_marks = max(0.0, min(max_marks, final_marks))
        
        # Generate feedback based on evaluation
        feedback = self._generate_feedback(
            student_answer, 
            model_answer, 
            rubric_coverage, 
            overall_similarity, 
            plagiarism_score
        )
        
        # Store the evaluation results
        from datetime import datetime
        self.eval_cursor.execute('''
        INSERT OR REPLACE INTO evaluations 
        (student_id, question_id, marks, max_marks, similarity_score, 
         rubric_coverage, plagiarism_score, plagiarism_source, feedback, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_id,
            question_id,
            final_marks,
            max_marks,
            overall_similarity,
            json.dumps(rubric_coverage),
            plagiarism_score,
            plagiarism_source or "",
            feedback,
            datetime.now().isoformat()
        ))
        self.eval_conn.commit()
        
        # Prepare results
        results = {
            "student_id": student_id,
            "question_id": question_id,
            "marks": final_marks,
            "max_marks": max_marks,
            "similarity_score": overall_similarity,
            "rubric_coverage": rubric_coverage,
            "avg_rubric_coverage": avg_rubric_coverage,
            "plagiarism_detected": plagiarism_score > 0.95,
            "plagiarism_score": plagiarism_score,
            "plagiarism_source": plagiarism_source,
            "feedback": feedback
        }
        
        return results
    
    def _generate_feedback(self, student_answer, model_answer, rubric_coverage, similarity, plagiarism_score):
        """Generate personalized feedback based on evaluation results"""
        feedback_parts = []
        
        # Overall assessment
        if similarity > 0.9:
            feedback_parts.append("Excellent answer! Your response closely aligns with the expected answer.")
        elif similarity > 0.7:
            feedback_parts.append("Good answer. You've covered most of the key points.")
        elif similarity > 0.5:
            feedback_parts.append("Satisfactory answer. You've addressed some key points but missed others.")
        else:
            feedback_parts.append("Your answer needs improvement. It deviates significantly from the expected response.")
        
        # Rubric-specific feedback
        strengths = []
        weaknesses = []
        
        for point, score in rubric_coverage.items():
            if score > 0.7:
                strengths.append(f"✓ Strong coverage of: {point}")
            elif score < 0.4:
                weaknesses.append(f"✗ Limited coverage of: {point}")
        
        if strengths:
            feedback_parts.append("\nStrengths:")
            feedback_parts.extend(strengths)
        
        if weaknesses:
            feedback_parts.append("\nAreas for improvement:")
            feedback_parts.extend(weaknesses)
        
        # Plagiarism warning if applicable
        if plagiarism_score > 0.95:
            feedback_parts.append("\n⚠️ Warning: Your answer contains sections that appear to be directly copied from course materials or other sources. Please ensure you understand the concepts and express them in your own words.")
        
        return "\n".join(feedback_parts)
    
    def get_evaluation(self, student_id, question_id):
        """Retrieve a stored evaluation"""
        self.eval_cursor.execute('''
        SELECT marks, max_marks, similarity_score, rubric_coverage, 
               plagiarism_score, plagiarism_source, feedback 
        FROM evaluations 
        WHERE student_id = ? AND question_id = ?
        ''', (student_id, question_id))
        
        row = self.eval_cursor.fetchone()
        if not row:
            return None
            
        try:
            rubric_coverage = json.loads(row[3])
        except:
            rubric_coverage = {}
            
        return {
            "student_id": student_id,
            "question_id": question_id,
            "marks": row[0],
            "max_marks": row[1],
            "similarity_score": row[2],
            "rubric_coverage": rubric_coverage,
            "plagiarism_score": row[4],
            "plagiarism_source": row[5],
            "feedback": row[6]
        }
    
    def get_student_evaluations(self, student_id):
        """Get all evaluations for a student"""
        self.eval_cursor.execute('''
        SELECT question_id, marks, max_marks, feedback 
        FROM evaluations 
        WHERE student_id = ?
        ''', (student_id,))
        
        results = []
        for row in self.eval_cursor.fetchall():
            results.append({
                "question_id": row[0],
                "marks": row[1],
                "max_marks": row[2],
                "feedback": row[3]
            })
            
        return results
    
    def close(self):
        """Close all database connections"""
        for conn in [self.docs_conn, self.model_conn, self.student_conn, self.eval_conn]:
            if conn:
                conn.close()

