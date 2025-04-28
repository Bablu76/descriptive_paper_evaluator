
import sqlite3
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AnswerEvaluator:
    def __init__(
        self,
        db_path: str = "./data/main.db",
        embedding_model: str = "all-MiniLM-L6-v2",
        plagiarism_threshold: float = 0.95
    ):
        self.model = SentenceTransformer(embedding_model, device='cpu')
        self.plagiarism_threshold = plagiarism_threshold
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self._create_tables()
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _create_tables(self):
        """Create evaluations table."""
        try:
            self.cursor.execute('''
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
            self.conn.commit()
            logger.info("Created evaluations table")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
            raise

    def _extract_rubric_points(self, rubric: str | List | Dict) -> List[str]:
        """Extract individual rubric points."""
        rubric_points = []
        if isinstance(rubric, str):
            try:
                rubric = json.loads(rubric)
            except json.JSONDecodeError:
                lines = rubric.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        line = re.sub(r'^[\•\-\*\>\◦\■\□\○\●\★\✓]\s*', '', line)
                        rubric_points.append(line)
                return rubric_points
        if isinstance(rubric, list):
            return rubric
        if isinstance(rubric, dict):
            for key, value in rubric.items():
                if isinstance(value, str):
                    rubric_points.append(f"{key}: {value}")
                else:
                    rubric_points.append(key)
        return rubric_points

    def _segment_answer(self, answer_text: str, num_segments: int = 3) -> List[str]:
        """Split an answer into fewer segments for CPU efficiency."""
        sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
        if len(sentences) <= num_segments:
            return sentences
        segment_size = len(sentences) // num_segments
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(sentences)
            segment = ' '.join(sentences[start_idx:end_idx])
            if segment:
                segments.append(segment)
        return segments

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        try:
            emb1 = self.model.encode(text1, batch_size=1)
            emb2 = self.model.encode(text2, batch_size=1)
            similarity = 1 - cosine(emb1, emb2)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _keyword_overlap(self, text1: str, text2: str, min_words: int = 5) -> float:
        """Calculate keyword overlap as a pre-filter for plagiarism detection.

        Args:
            text1 (str): First text to compare.
            text2 (str): Second text to compare.
            min_words (int): Minimum number of words required in each text. Defaults to 5.

        Returns:
            float: Ratio of common words to the minimum set size, or 0.0 if conditions aren't met.
        """
        if not text1 or not text2:
            return 0.0
        
        # Extract words longer than 3 characters, converted to lowercase
        words1 = {word.lower() for word in text1.split() if len(word) > 3}
        words2 = {word.lower() for word in text2.split() if len(word) > 3}

        # Check if either set has fewer than min_words or is empty
        if len(words1) < min_words or len(words2) < min_words or not words1 or not words2:
            return 0.0

        # Calculate overlap
        common = len(words1 & words2)
        return common / min(len(words1), len(words2))

    def _check_plagiarism(self, student_answer: str, student_id: str) -> tuple[float, Optional[str]]:
        """Check for plagiarism with stricter pre-filter."""
        highest_similarity = 0.0
        source = None
        segments = self._segment_answer(student_answer)
        try:
            self.cursor.execute('SELECT id, book_title, chunk_text FROM textbook_chunks')
            textbook_chunks = self.cursor.fetchall()
            for segment in segments:
                if len(segment.split()) < 10:
                    continue
                for chunk_id, book_title, chunk_text in textbook_chunks:
                    if self._keyword_overlap(segment, chunk_text) < 0.4:  # Stricter threshold
                        continue
                    similarity = self._calculate_similarity(segment, chunk_text)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        source = f"Textbook: {book_title}"
        except sqlite3.Error as e:
            logger.error(f"Error checking textbook plagiarism: {e}")

        try:
            self.cursor.execute(
                'SELECT student_id, answer_text FROM student_answers WHERE student_id != ?',
                (student_id,)
            )
            other_answers = self.cursor.fetchall()
            for other_student_id, other_answer in other_answers:
                for segment in segments:
                    if len(segment.split()) < 10:
                        continue
                    if self._keyword_overlap(segment, other_answer) < 0.4:  # Stricter threshold
                        continue
                    similarity = self._calculate_similarity(segment, other_answer)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        source = f"Student: {other_student_id}"
        except sqlite3.Error as e:
            logger.error(f"Error checking student plagiarism: {e}")

        if highest_similarity >= self.plagiarism_threshold:
            return highest_similarity, source
        return 0.0, None

    def evaluate_answer(
        self,
        student_id: str,
        question_id: int,
        question_data: Optional[Dict] = None
    ) -> Dict:
        """Evaluate a student's answer."""
        if not question_data:
            try:
                self.cursor.execute(
                    'SELECT question_number, question_text, marks, rubric FROM questions WHERE id = ?',
                    (question_id,)
                )
                question_row = self.cursor.fetchone()
                if not question_row:
                    logger.error(f"Question ID {question_id} not found")
                    return {"error": f"Question with ID {question_id} not found"}
                question_data = {
                    "id": question_id,
                    "question_number": question_row[0],
                    "question_text": question_row[1],
                    "marks": question_row[2],
                    "rubric": question_row[3]
                }
            except sqlite3.Error as e:
                logger.error(f"Error fetching question data: {e}")
                return {"error": "Database error"}

        try:
            self.cursor.execute(
                'SELECT model_answer FROM model_answers WHERE question_id = ?',
                (question_id,)
            )
            model_row = self.cursor.fetchone()
            if not model_row:
                logger.error(f"No model answer for question {question_id}")
                return {"error": f"No model answer found for question {question_id}"}
            model_answer = model_row[0]
        except sqlite3.Error as e:
            logger.error(f"Error fetching model answer: {e}")
            return {"error": "Database error"}

        try:
            self.cursor.execute(
                'SELECT answer_text FROM student_answers WHERE student_id = ? AND question_id = ?',
                (student_id, question_id)
            )
            student_row = self.cursor.fetchone()
            if not student_row:
                logger.error(f"No answer for student {student_id}, question {question_id}")
                return {"error": f"No answer found for student {student_id}, question {question_id}"}
            student_answer = student_row[0]
        except sqlite3.Error as e:
            logger.error(f"Error fetching student answer: {e}")
            return {"error": "Database error"}

        rubric = question_data.get('rubric', [])
        if isinstance(rubric, str):
            try:
                rubric = json.loads(rubric)
            except json.JSONDecodeError:
                pass
        rubric_points = self._extract_rubric_points(rubric)

        overall_similarity = self._calculate_similarity(student_answer, model_answer)
        rubric_coverage = {}
        student_segments = self._segment_answer(student_answer)
        for point in rubric_points:
            best_similarity = 0.0
            for segment in student_segments:
                similarity = self._calculate_similarity(point, segment)
                best_similarity = max(best_similarity, similarity)
            rubric_coverage[point] = best_similarity

        avg_rubric_coverage = np.mean(list(rubric_coverage.values())) if rubric_coverage else 0.0
        plagiarism_score, plagiarism_source = self._check_plagiarism(student_answer, student_id)

        max_marks = question_data.get('marks', 10)
        base_score = (overall_similarity * 0.5 + avg_rubric_coverage * 0.5) * max_marks
        final_marks = base_score
        if plagiarism_score >= self.plagiarism_threshold:
            final_marks *= 0.5
        final_marks = max(0.0, min(max_marks, final_marks))

        feedback = self._generate_feedback(
            student_answer, model_answer, rubric_coverage, overall_similarity, plagiarism_score
        )

        try:
            self.cursor.execute('''
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
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error storing evaluation: {e}")
            return {"error": "Database error"}

        return {
            "student_id": student_id,
            "question_id": question_id,
            "marks": final_marks,
            "max_marks": max_marks,
            "similarity_score": overall_similarity,
            "rubric_coverage": rubric_coverage,
            "avg_rubric_coverage": avg_rubric_coverage,
            "plagiarism_detected": plagiarism_score >= self.plagiarism_threshold,
            "plagiarism_score": plagiarism_score,
            "plagiarism_source": plagiarism_source,
            "feedback": feedback
        }

    def _generate_feedback(
        self,
        student_answer: str,
        model_answer: str,
        rubric_coverage: Dict[str, float],
        similarity: float,
        plagiarism_score: float
    ) -> str:
        """Generate concise feedback."""
        feedback_parts = []
        if similarity > 0.9:
            feedback_parts.append("Excellent answer!")
        elif similarity > 0.7:
            feedback_parts.append("Good answer.")
        elif similarity > 0.5:
            feedback_parts.append("Satisfactory answer.")
        else:
            feedback_parts.append("Needs improvement.")

        if plagiarism_score >= self.plagiarism_threshold:
            feedback_parts.append("⚠️ High similarity detected.")

        return "\n".join(feedback_parts)

    def get_evaluation(self, student_id: str, question_id: int) -> Optional[Dict]:
        """Retrieve a stored evaluation."""
        try:
            self.cursor.execute('''
            SELECT marks, max_marks, similarity_score, rubric_coverage, 
                   plagiarism_score, plagiarism_source, feedback 
            FROM evaluations 
            WHERE student_id = ? AND question_id = ?
            ''', (student_id, question_id))
            row = self.cursor.fetchone()
            if not row:
                return None
            rubric_coverage = json.loads(row[3]) if row[3] else {}
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
        except sqlite3.Error as e:
            logger.error(f"Error retrieving evaluation: {e}")
            return None

    def get_student_evaluations(self, student_id: str) -> List[Dict]:
        """Get all evaluations for a student."""
        try:
            self.cursor.execute('''
            SELECT question_id, marks, max_marks, feedback 
            FROM evaluations 
            WHERE student_id = ?
            ''', (student_id,))
            return [
                {
                    "question_id": row[0],
                    "marks": row[1],
                    "max_marks": row[2],
                    "feedback": row[3]
                }
                for row in self.cursor.fetchall()
            ]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving student evaluations: {e}")
            return []

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            logger.info("Closed evaluator database connection")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
