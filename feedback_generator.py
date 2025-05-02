import json
import logging
import sqlite3

class FeedbackGenerator:
    """
    Generates JSON feedback for each student and an instructor summary report.
    """
    def __init__(self, db_path: str = "evaluation_results.db"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def generate_feedback(self):
        """
        Compile evaluation data into JSON feedback files.
        """
        # Get list of students and questions
        self.cursor.execute("SELECT DISTINCT student_id FROM evaluations")
        students = [row[0] for row in self.cursor.fetchall()]
        self.cursor.execute("SELECT DISTINCT question_id FROM evaluations")
        questions = [row[0] for row in self.cursor.fetchall()]
        
        # Instructor summary
        instructor_summary = {"average_similarity_per_question": {}, 
                              "average_coverage_per_question": {}}
        for q in questions:
            self.cursor.execute("SELECT AVG(similarity), AVG(coverage) FROM evaluations WHERE question_id = ?", (q,))
            avg_sim, avg_cov = self.cursor.fetchone()
            instructor_summary["average_similarity_per_question"][q] = avg_sim
            instructor_summary["average_coverage_per_question"][q] = avg_cov
        
        try:
            with open("instructor_summary.json", 'w', encoding='utf-8') as f:
                json.dump(instructor_summary, f, indent=2)
            self.logger.info("Instructor summary saved to instructor_summary.json.")
        except Exception as e:
            self.logger.error(f"Failed to save instructor summary: {e}")
        
        # Per-student feedback
        for student in students:
            self.cursor.execute(
                "SELECT question_id, similarity, coverage, plagiarism_flag FROM evaluations WHERE student_id = ?", 
                (student,)
            )
            rows = self.cursor.fetchall()
            feedback = {"student_id": student, "questions": [], "average_similarity": None, "average_coverage": None}
            total_sim = total_cov = 0.0
            count = len(rows)
            for (q_id, sim, cov, plag) in rows:
                feedback["questions"].append({
                    "question_id": q_id,
                    "similarity": sim,
                    "coverage": cov,
                    "plagiarism_flag": bool(plag)
                })
                total_sim += sim
                total_cov += cov
            if count > 0:
                feedback["average_similarity"] = total_sim / count
                feedback["average_coverage"] = total_cov / count
            filename = f"feedback_{student}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(feedback, f, indent=2)
                self.logger.info(f"Feedback saved for student {student} in {filename}.")
            except Exception as e:
                self.logger.error(f"Failed to save feedback for {student}: {e}")
    
    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    fg = FeedbackGenerator()
    fg.generate_feedback()
