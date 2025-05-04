# --- START OF FILE feedback_generator.py ---

import json
import logging
import sqlite3
import os # Added import
from pathlib import Path # Use Pathlib for robust path handling

class FeedbackGenerator:
    """
    Generates JSON feedback for each student and an instructor summary report.
    Saves files to a specified output directory.
    """
    def __init__(self, db_path: str = "evaluation_results.db"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = db_path # Store db_path if needed later
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to database: {db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database {db_path}: {e}")
            self.conn = None # Mark connection as invalid
            self.cursor = None

    def generate_feedback(self, output_dir: str = "feedback_reports"): # Added output_dir parameter with default
        """
        Compile evaluation data into JSON feedback files saved in output_dir.
        """
        if not self.conn or not self.cursor:
             self.logger.error("Database connection not available. Cannot generate feedback.")
             return False # Indicate failure

        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
            self.logger.info(f"Ensured feedback output directory exists: {output_path}")
        except OSError as e:
             self.logger.error(f"Failed to create output directory {output_path}: {e}")
             return False # Indicate failure

        # --- Fetch data ---
        try:
            self.cursor.execute("SELECT DISTINCT student_id FROM evaluations")
            students = [row[0] for row in self.cursor.fetchall()]
            self.cursor.execute("SELECT DISTINCT question_id FROM evaluations")
            questions = sorted([row[0] for row in self.cursor.fetchall()]) # Sort questions for consistent order
            self.logger.info(f"Found {len(students)} students and {len(questions)} questions in evaluations.")
        except sqlite3.Error as e:
             self.logger.error(f"Error fetching student/question lists from database: {e}")
             return False

        if not students:
             self.logger.warning("No student data found in the database. No feedback generated.")
             # Optionally generate an empty summary? For now, just exit.
             return True # No error, just no data.

        # --- Instructor summary ---
        instructor_summary = {
            "total_students": len(students),
            "total_questions": len(questions),
            "average_similarity_per_question": {},
            "average_coverage_per_question": {},
            "plagiarism_flags_per_question": {}
        }
        overall_avg_sim = 0.0
        overall_avg_cov = 0.0
        overall_plag_flags = 0
        total_responses = 0

        self.logger.info("Generating instructor summary...")
        for q in questions:
            try:
                self.cursor.execute("""
                    SELECT AVG(similarity), AVG(coverage), SUM(CASE WHEN plagiarism_flag = 1 THEN 1 ELSE 0 END), COUNT(*)
                    FROM evaluations
                    WHERE question_id = ?
                """, (q,))
                # fetchone() might return None if no rows for a question_id (unlikely with DISTINCT query earlier)
                row = self.cursor.fetchone()
                if row:
                     avg_sim, avg_cov, plag_flags, count = row
                     instructor_summary["average_similarity_per_question"][q] = avg_sim if avg_sim is not None else 0.0
                     instructor_summary["average_coverage_per_question"][q] = avg_cov if avg_cov is not None else 0.0
                     instructor_summary["plagiarism_flags_per_question"][q] = plag_flags if plag_flags is not None else 0
                     # Accumulate for overall averages
                     if avg_sim is not None: overall_avg_sim += avg_sim * count
                     if avg_cov is not None: overall_avg_cov += avg_cov * count
                     if plag_flags is not None: overall_plag_flags += plag_flags
                     total_responses += count
                else:
                     instructor_summary["average_similarity_per_question"][q] = 0.0
                     instructor_summary["average_coverage_per_question"][q] = 0.0
                     instructor_summary["plagiarism_flags_per_question"][q] = 0
            except sqlite3.Error as e:
                 self.logger.error(f"Error calculating stats for question {q}: {e}")
                 instructor_summary["average_similarity_per_question"][q] = f"[Error: {e}]"
                 instructor_summary["average_coverage_per_question"][q] = f"[Error: {e}]"
                 instructor_summary["plagiarism_flags_per_question"][q] = f"[Error: {e}]"

        # Calculate overall averages
        if total_responses > 0:
            instructor_summary["overall_average_similarity"] = overall_avg_sim / total_responses
            instructor_summary["overall_average_coverage"] = overall_avg_cov / total_responses
        else:
            instructor_summary["overall_average_similarity"] = 0.0
            instructor_summary["overall_average_coverage"] = 0.0
        instructor_summary["overall_plagiarism_flags"] = overall_plag_flags


        summary_filename = output_path / "instructor_summary.json"
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(instructor_summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Instructor summary saved to {summary_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save instructor summary: {e}")
            # Continue to student feedback despite summary failure? Yes.

        # --- Per-student feedback ---
        self.logger.info("Generating per-student feedback...")
        files_generated = 0
        for student in students:
            try:
                self.cursor.execute("""
                    SELECT question_id, similarity, coverage, plagiarism_flag
                    FROM evaluations
                    WHERE student_id = ?
                    ORDER BY question_id
                """, (student,)) # Order for consistency
                rows = self.cursor.fetchall()
            except sqlite3.Error as e:
                 self.logger.error(f"Failed to fetch data for student {student}: {e}")
                 continue # Skip this student

            feedback = {"student_id": student, "questions": [], "average_similarity": None, "average_coverage": None}
            total_sim = 0.0
            total_cov = 0.0
            count = len(rows)

            for (q_id, sim, cov, plag) in rows:
                feedback["questions"].append({
                    "question_id": q_id,
                    "similarity": sim,
                    "coverage": cov,
                    "plagiarism_flag": bool(plag)
                })
                total_sim += sim if sim is not None else 0.0
                total_cov += cov if cov is not None else 0.0

            if count > 0:
                feedback["average_similarity"] = total_sim / count
                feedback["average_coverage"] = total_cov / count

            # Use Pathlib for robust path construction
            student_filename = output_path / f"feedback_{student}.json"
            try:
                with open(student_filename, 'w', encoding='utf-8') as f:
                    json.dump(feedback, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"Feedback saved for student {student} in {student_filename}")
                files_generated += 1
            except Exception as e:
                self.logger.error(f"Failed to save feedback for {student}: {e}")

        self.logger.info(f"Generated {files_generated} student feedback files in {output_path}.")
        return True # Indicate success


    def __del__(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")

# Example Usage (if needed)
if __name__ == "__main__":
     # Create dummy db for testing if needed
     if not os.path.exists("evaluation_results.db"):
          conn_test = sqlite3.connect("evaluation_results.db")
          cursor_test = conn_test.cursor()
          cursor_test.execute("""
          CREATE TABLE IF NOT EXISTS evaluations (
              student_id TEXT, question_id INTEGER, similarity REAL, coverage REAL, plagiarism_flag INTEGER)""")
          cursor_test.execute("INSERT INTO evaluations VALUES (?,?,?,?,?)", ('S101', 0, 0.85, 0.7, 0))
          cursor_test.execute("INSERT INTO evaluations VALUES (?,?,?,?,?)", ('S101', 1, 0.96, 0.9, 1))
          cursor_test.execute("INSERT INTO evaluations VALUES (?,?,?,?,?)", ('S102', 0, 0.70, 0.5, 0))
          conn_test.commit()
          conn_test.close()

     fg = FeedbackGenerator(db_path="evaluation_results.db")
     if fg.conn: # Check if connection succeeded
         fg.generate_feedback(output_dir="test_feedback") # Use a test directory
     else:
          print("Failed to initialize FeedbackGenerator due to DB connection error.")


# --- END OF FILE feedback_generator.py ---