import json
import logging
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    """
    Evaluates student answers against model answers using semantic similarity and keyword coverage.
    Results are stored in an SQLite database.
    """
    def __init__(self, db_path: str = "evaluation_results.db"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load sentence-transformer model for embeddings (semantic similarity)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize SQLite database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()
    
    def create_table(self):
        """
        Create evaluation table if not exists.
        """
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            student_id TEXT,
            question_id INTEGER,
            similarity REAL,
            coverage REAL,
            plagiarism_flag INTEGER
        )
        """)
        self.conn.commit()
    
    def compute_similarity(self, model_answer: str, student_answer: str) -> float:
        """
        Compute cosine similarity between model and student answer embeddings.
        """
        embeddings = self.model.encode([model_answer, student_answer])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)
    
    def compute_coverage(self, student_answer: str, keywords: list) -> float:
        """
        Fraction of rubric keywords found in the student answer.
        """
        if not keywords:
            return 0.0
        text = student_answer.lower()
        count = sum(1 for kw in keywords if kw.lower() in text)
        return count / len(keywords)
    
    def flag_plagiarism(self, similarity: float, threshold: float = 0.95) -> bool:
        """
        Flag potential plagiarism if similarity > threshold.
        """
        return similarity > threshold
    
    def evaluate(self, student_answers_path: str, model_answers_path: str):
        """
        Perform evaluation for all students/questions. Inserts into SQLite.
        """
        try:
            with open(student_answers_path, 'r', encoding='utf-8') as f:
                student_data = json.load(f)
            with open(model_answers_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON files: {e}")
            return
        
        model_answers = model_data.get("answers", [])
        students = student_data.get("students", [])
        
        self.logger.info("Starting evaluation of student answers.")
        for student in students:
            student_id = student.get("student_id")
            student_text = student.get("answer_text", "")
            for q_idx, model in enumerate(model_answers):
                keywords = model.get("keywords", [])
                model_text = model.get("model_answer", "")
                
                sim_score = self.compute_similarity(model_text, student_text)
                coverage = self.compute_coverage(student_text, keywords)
                plag_flag = self.flag_plagiarism(sim_score)
                
                self.cursor.execute(
                    "INSERT INTO evaluations (student_id, question_id, similarity, coverage, plagiarism_flag) VALUES (?, ?, ?, ?, ?)",
                    (student_id, q_idx, sim_score, coverage, int(plag_flag))
                )
        
        self.conn.commit()
        self.logger.info("Evaluation complete and results stored in database.")
    
    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate("student_answers.json", "model_answers.json")
