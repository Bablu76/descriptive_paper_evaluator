import os
import json
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, db_path: str = "data/evaluation_results.db", model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the evaluator with a database path and embedding model.
        
        Args:
            db_path: Path to the SQLite database
            model_name: Name of the SentenceTransformer model to use
        """
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the embedding model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        
        # Cache for embeddings to avoid redundant computation
        self._embedding_cache = {}

    def _create_table(self):
        """Create the evaluations table in the database if it doesn't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                student_id TEXT,
                question_id INTEGER,
                similarity REAL,
                coverage REAL,
                plagiarism_flag INTEGER,
                rubric_scores TEXT,
                strengths TEXT,
                weaknesses TEXT,
                improvement_suggestions TEXT,
                total_marks REAL,
                max_marks INTEGER,
                PRIMARY KEY (student_id, question_id)
            )
        ''')
        self.conn.commit()
        logger.debug("Database table initialized")

    def _get_embedding(self, text: str):
        """Get embedding for a text string with caching."""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self._embedding_cache[text]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            return float(util.pytorch_cos_sim(emb1, emb2).item())
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def extract_keywords(self, text: str, expected_keywords: set = None) -> set:
        """
        Extract meaningful keywords from text.
        
        Args:
            text: Text to extract keywords from
            expected_keywords: Set of expected keywords to filter against
            
        Returns:
            Set of extracted keywords
        """
        # Simple tokenization and filtering
        words = set(word.lower() for word in text.replace('.', ' ').replace(',', ' ')
                    .replace('!', ' ').replace('?', ' ').replace('(', ' ')
                    .replace(')', ' ').split())
                    
        # Filter out common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of', 'from', 'up', 'down',
            'that', 'this', 'these', 'those', 'am', 'have', 'has', 'had', 'do', 'does', 'did',
            'not', 'no', 'nor', 'can', 'could', 'should', 'would', 'shall', 'will', 'may', 'might',
            'must', 'then', 'than', 'so', 'such', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'i', 'me', 'my',
            'you', 'your', 'him', 'his', 'her', 'she', 'he', 'what', 'which', 'who', 'whom', 'whose',
            'when', 'where', 'why', 'how', 'all', 'any', 'every', 'if', 'while', 'because', 'though'
        }
        words = words - stopwords
        
        # If expected keywords are provided, prioritize matches
        if expected_keywords:
            return words & expected_keywords
        
        return words

    def detect_peer_plagiarism(self, all_student_answers: Dict[str, List[str]]) -> Dict[Tuple[str, int], bool]:
        """
        Detect potential plagiarism between student answers.
        
        Args:
            all_student_answers: Dictionary mapping student IDs to their list of answers
            
        Returns:
            Dictionary mapping (student_id, question_index) to plagiarism flag
        """
        flagged = {}
        student_ids = list(all_student_answers.keys())
        
        # Check each pair of students exactly once
        for i, sid1 in enumerate(student_ids):
            for j in range(i + 1, len(student_ids)):
                sid2 = student_ids[j]
                
                # For each question answered by both students
                for q_idx, (ans1, ans2) in enumerate(zip(all_student_answers[sid1], all_student_answers[sid2])):
                    # Skip empty answers
                    if not ans1.strip() or not ans2.strip():
                        continue
                        
                    # Compute similarity threshold based on answer length
                    # Shorter answers naturally have higher similarity by chance
                    length_factor = min(1.0, max(0.7, 0.9 - 0.2 * (len(ans1) + len(ans2)) / 2000))
                    threshold = 0.8 * length_factor
                    
                    sim = self.compute_similarity(ans1, ans2)
                    
                    # Flag if similarity exceeds threshold
                    if sim > threshold:
                        logger.info(f"Potential plagiarism detected between students {sid1} and {sid2} for question {q_idx}")
                        flagged[(sid1, q_idx)] = True
                        flagged[(sid2, q_idx)] = True

        return flagged

    def _record_zero_score(self, student_id: str, q_idx: int, model: Dict[str, Any]):
        """Record zero score for empty or missing answers."""
        max_marks = model.get("marks", 0)
        self.cursor.execute("""
            INSERT OR REPLACE INTO evaluations (
                student_id, question_id, similarity, coverage, plagiarism_flag,
                rubric_scores, strengths, weaknesses, improvement_suggestions,
                total_marks, max_marks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id, q_idx, 0.0, 0.0, 0,
            json.dumps({}), json.dumps([]), json.dumps(["No answer provided"]), 
            json.dumps(["Submit a complete answer"]), 0.0, max_marks
        ))

    def evaluate(self, student_answers: Dict[str, List[str]], model_answers: List[Dict[str, Any]]):
        """
        Evaluate student answers against model answers and rubrics.
        
        Args:
            student_answers: Dictionary mapping student IDs to their list of answers
            model_answers: List of dictionaries containing model answers and rubrics
        """
        # First detect potential plagiarism across all submissions
        peer_flags = self.detect_peer_plagiarism(student_answers)
        
        # Process each student's answers
        for student_id, answers in student_answers.items():
            logger.info(f"Evaluating answers for student {student_id}")
            
            # Process each question
            for q_idx, student_text in enumerate(answers):
                # Skip if we don't have a model answer for this question
                if q_idx >= len(model_answers):
                    logger.warning(f"No model answer provided for question {q_idx}")
                    continue
                    
                model = model_answers[q_idx]
                model_text = model.get("model_answer", "")
                
                # Skip empty answers
                if not student_text.strip():
                    logger.warning(f"Empty answer from student {student_id} for question {q_idx}")
                    self._record_zero_score(student_id, q_idx, model)
                    continue
                
                # Calculate semantic similarity with model answer
                sim_score = self.compute_similarity(student_text, model_text)
                
                # Check keyword coverage
                model_keywords = set(model.get("keywords", []))
                if model_keywords:
                    student_keywords = self.extract_keywords(student_text, model_keywords)
                    matched_keywords = model_keywords & student_keywords
                    coverage = len(matched_keywords) / len(model_keywords) if model_keywords else 0.0
                else:
                    # Fallback if no keywords specified
                    coverage = sim_score
                
                # Check for plagiarism flag
                plag_flag = int(peer_flags.get((student_id, q_idx), False))
                
                # Evaluate against rubrics
                rubric_scores = {}
                total_scored = 0.0
                total_possible = 0.0
                
                for rubric in model.get("Rubrics", []):
                    crit = rubric.get("Criteria", "")
                    marks = rubric.get("Marks", 1)
                    
                    # Skip empty criteria
                    if not crit.strip():
                        continue
                    
                    # Calculate weighted similarity for this criteria
                    sim = self.compute_similarity(crit, student_text)
                    
                    # Apply non-linear scoring function for better discrimination
                    # This gives more weight to high-similarity matches
                    score = round((sim ** 1.5) * marks, 2)
                    
                    rubric_scores[crit] = score
                    total_scored += score
                    total_possible += marks
                
                # Calculate total rubric coverage
                rubric_coverage = total_scored / total_possible if total_possible > 0 else 0.0
                
                # Calculate final score with configurable weights
                alpha, beta, gamma = 0.5, 0.5, 0.3  # Weight parameters
                
                # Apply final scoring formula
                final_score = alpha * sim_score + beta * rubric_coverage
                
                # Apply plagiarism penalty if flagged
                if plag_flag:
                    final_score -= gamma
                
                # Ensure score is in valid range
                final_score = max(0.0, min(1.0, final_score))
                
                # Calculate marks based on question weight
                max_marks = model.get("marks", 0)
                awarded_marks = round(final_score * max_marks, 2)
                
                # Generate detailed feedback on strengths and weaknesses
                strengths = []
                weaknesses = []
                
                if rubric_scores:
                    # Sort rubrics by score
                    sorted_rubrics = sorted(rubric_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Select top strengths (rubrics with high scores)
                    for crit, score in sorted_rubrics:
                        normalized_score = score / model.get("Rubrics", [{"Marks": 1}])[0].get("Marks", 1)
                        if normalized_score > 0.7:
                            strengths.append(crit)
                        elif normalized_score < 0.4:
                            weaknesses.append(crit)
                
                # If no specific strengths identified, look at overall performance
                if not strengths and sim_score > 0.7:
                    strengths.append("Good overall understanding of concepts")
                
                # If no specific weaknesses identified but overall score is low
                if not weaknesses and sim_score < 0.4:
                    weaknesses.append("Needs to improve overall conceptual understanding")
                
                # Generate specific improvement suggestions based on weaknesses
                improvements = []
                for weakness in weaknesses:
                    # Parse the weakness to create an improvement suggestion
                    suggestion = f"Work on {weakness.lower()}"
                    improvements.append(suggestion)
                
                # Fallback if no specific improvements identified
                if not improvements and rubric_coverage < 0.6:
                    improvements.append("Review course materials to better address all rubric criteria")
                    
                # Store legacy best/worst rubric fields for backward compatibility
                best_rubric = strengths[0] if strengths else ""
                worst_rubric = weaknesses[0] if weaknesses else ""
                
                # Store evaluation results
                try:
                    self.cursor.execute("""
                        INSERT OR REPLACE INTO evaluations (
                            student_id, question_id, similarity, coverage, plagiarism_flag,
                            rubric_scores, strengths, weaknesses, improvement_suggestions,
                            total_marks, max_marks
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        student_id, q_idx, sim_score, coverage, plag_flag,
                        json.dumps(rubric_scores), json.dumps(strengths), 
                        json.dumps(weaknesses), json.dumps(improvements),
                        awarded_marks, max_marks
                    ))
                except sqlite3.Error as e:
                    logger.error(f"Database error: {e}")
                
        # Commit all changes to database
        self.conn.commit()
        logger.info("Evaluation completed for all students.")

    def get_student_results(self, student_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation results for a specific student.
        
        Args:
            student_id: ID of the student
            
        Returns:
            List of evaluation results for each question
        """
        self.cursor.execute("""
            SELECT question_id, similarity, coverage, plagiarism_flag, 
                   rubric_scores, strengths, weaknesses, improvement_suggestions,
                   total_marks, max_marks
            FROM evaluations 
            WHERE student_id = ?
            ORDER BY question_id
        """, (student_id,))
        
        results = []
        for row in self.cursor.fetchall():
            question_id, similarity, coverage, plagiarism_flag, \
            rubric_scores_json, strengths_json, weaknesses_json, improvements_json, \
            total_marks, max_marks = row
            
            # Parse JSON fields
            try:
                rubric_scores = json.loads(rubric_scores_json)
                strengths = json.loads(strengths_json)
                weaknesses = json.loads(weaknesses_json)
                improvements = json.loads(improvements_json)
            except (json.JSONDecodeError, TypeError):
                rubric_scores = {}
                strengths = []
                weaknesses = []
                improvements = []
            
            # Calculate percentage
            percentage = round((total_marks / max_marks) * 100, 1) if max_marks > 0 else 0
            
            results.append({
                "question_id": question_id,
                "similarity": similarity,
                "coverage": coverage,
                "plagiarism_flag": bool(plagiarism_flag),
                "rubric_scores": rubric_scores,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "improvement_suggestions": improvements,
                "total_marks": total_marks,
                "max_marks": max_marks,
                "percentage": percentage
            })
            
        return results

    def analyze_student_performance(self, student_id: str) -> Dict[str, Any]:
        """
        Analyze a student's overall performance.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Dictionary containing comprehensive performance analysis
        """
        results = self.get_student_results(student_id)
        
        if not results:
            return {
                "student_id": student_id,
                "overall_performance": "No data available",
                "total_score": 0,
                "total_possible": 0,
                "percentage": 0,
                "consistent_strengths": [],
                "consistent_weaknesses": [],
                "key_improvement_areas": [],
                "question_breakdown": []
            }
        
        # Calculate overall metrics
        total_score = sum(r["total_marks"] for r in results)
        total_possible = sum(r["max_marks"] for r in results)
        overall_percentage = round(total_score / total_possible * 100, 1) if total_possible > 0 else 0
        
        # Determine performance level
        if overall_percentage >= 85:
            performance_level = "Excellent"
        elif overall_percentage >= 70:
            performance_level = "Good"
        elif overall_percentage >= 55:
            performance_level = "Satisfactory"
        elif overall_percentage >= 40:
            performance_level = "Needs improvement"
        else:
            performance_level = "Unsatisfactory"
        
        # Aggregate strengths and weaknesses across all questions
        all_strengths = []
        all_weaknesses = []
        all_improvements = []
        
        for r in results:
            all_strengths.extend(r["strengths"])
            all_weaknesses.extend(r["weaknesses"])
            all_improvements.extend(r["improvement_suggestions"])
        
        # Count frequency of each strength/weakness
        strength_counter = Counter(all_strengths)
        weakness_counter = Counter(all_weaknesses)
        improvement_counter = Counter(all_improvements)
        
        # Get consistent patterns (appearing in multiple questions)
        consistent_strengths = [s for s, count in strength_counter.items() if count > 1]
        consistent_weaknesses = [w for w, count in weakness_counter.items() if count > 1]
        
        # Prioritize improvement areas
        key_improvements = [imp for imp, _ in improvement_counter.most_common(3)]
        
        # Create question breakdown
        question_breakdown = []
        for r in results:
            question_breakdown.append({
                "question_id": r["question_id"],
                "score": f"{r['total_marks']}/{r['max_marks']} ({r['percentage']}%)",
                "strengths": r["strengths"],
                "weaknesses": r["weaknesses"],
                "improvements": r["improvement_suggestions"]
            })
        
        # Generate personalized performance summary
        if len(consistent_strengths) > 0:
            strength_summary = f"Consistently demonstrates {', '.join(consistent_strengths[:2])}"
        else:
            unique_strengths = [s for s, _ in strength_counter.most_common(2)]
            strength_summary = f"Shows strengths in {', '.join(unique_strengths)}" if unique_strengths else "No consistent strengths identified"
        
        if len(consistent_weaknesses) > 0:
            weakness_summary = f"Consistently struggles with {', '.join(consistent_weaknesses[:2])}"
        else:
            unique_weaknesses = [w for w, _ in weakness_counter.most_common(2)]
            weakness_summary = f"Shows weaknesses in {', '.join(unique_weaknesses)}" if unique_weaknesses else "No consistent weaknesses identified"
        
        performance_summary = f"{performance_level} overall. {strength_summary}. {weakness_summary}."
        
        # Build complete analysis
        analysis = {
            "student_id": student_id,
            "overall_performance": performance_summary,
            "total_score": total_score,
            "total_possible": total_possible,
            "percentage": overall_percentage,
            "performance_level": performance_level,
            "consistent_strengths": consistent_strengths,
            "consistent_weaknesses": consistent_weaknesses,
            "key_improvement_areas": key_improvements,
            "question_breakdown": question_breakdown
        }
        
        return analysis

    def analyze_concept_mastery(self, student_id: str, concept_keywords: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Analyze a student's mastery of specific concepts based on their answers.
        
        Args:
            student_id: ID of the student
            concept_keywords: Dictionary mapping concept names to lists of related keywords
            
        Returns:
            Dictionary of concept mastery scores (0-1 scale)
        """
        # Get all student answers
        self.cursor.execute("""
            SELECT question_id, rubric_scores, similarity 
            FROM evaluations
            WHERE student_id = ?
        """, (student_id,))
        
        results = self.cursor.fetchall()
        if not results:
            logger.warning(f"No evaluation data found for student {student_id}")
            return {concept: 0.0 for concept in concept_keywords.keys()}
        
        # Calculate concept mastery based on rubric criteria and similarity scores
        concept_scores = {}
        
        for concept, keywords in concept_keywords.items():
            concept_matches = 0
            total_possible = 0
            
            for _, rubric_scores_json, similarity in results:
                try:
                    rubric_scores = json.loads(rubric_scores_json)
                except (json.JSONDecodeError, TypeError):
                    rubric_scores = {}
                
                # Check if any rubric criteria relate to this concept
                concept_related_criteria = []
                for criteria in rubric_scores.keys():
                    if any(kw.lower() in criteria.lower() for kw in keywords):
                        concept_related_criteria.append(criteria)
                
                if concept_related_criteria:
                    # Calculate average score for this concept's criteria
                    concept_score = sum(rubric_scores[c] for c in concept_related_criteria) / len(concept_related_criteria)
                    concept_matches += concept_score
                    total_possible += 1
            
            # Calculate mastery score
            concept_scores[concept] = round(concept_matches / total_possible, 2) if total_possible > 0 else 0.0
        
        return concept_scores
        
    def generate_personalized_feedback(self, student_id: str) -> str:
        """
        Generate personalized feedback for a student based on their performance analysis.
        
        Args:
            student_id: ID of the student
            
        Returns:
            String containing personalized feedback
        """
        analysis = self.analyze_student_performance(student_id)
        
        if "No data available" in analysis["overall_performance"]:
            return "No evaluation data available for this student."
        
        # Build personalized feedback message
        feedback = []
        
        # Overall assessment
        feedback.append(f"Overall Performance: {analysis['performance_level']} ({analysis['percentage']}%)")
        feedback.append(f"Total Score: {analysis['total_score']}/{analysis['total_possible']}")
        feedback.append("")
        
        # Strengths section
        feedback.append("Strengths:")
        if analysis["consistent_strengths"]:
            for strength in analysis["consistent_strengths"][:3]:
                feedback.append(f"- {strength}")
        else:
            feedback.append("- No consistent strengths identified across questions")
        feedback.append("")
        
        # Areas for improvement
        feedback.append("Areas for Improvement:")
        if analysis["key_improvement_areas"]:
            for area in analysis["key_improvement_areas"][:3]:
                feedback.append(f"- {area}")
        else:
            feedback.append("- No specific improvement areas identified")
        feedback.append("")
        
        # Question breakdown
        feedback.append("Question Breakdown:")
        for q in analysis["question_breakdown"]:
            feedback.append(f"Question {q['question_id']}: {q['score']}")
            if q["strengths"]:
                feedback.append(f"  Strengths: {', '.join(q['strengths'][:2])}")
            if q["weaknesses"]:
                feedback.append(f"  Needs work: {', '.join(q['weaknesses'][:2])}")
        
        # Concluding statement
        if analysis["percentage"] >= 70:
            feedback.append("\nKeep up the good work! Focus on the improvement areas to further enhance your understanding.")
        elif analysis["percentage"] >= 50:
            feedback.append("\nYou're on the right track. Dedicating more time to the improvement areas will strengthen your grasp of the material.")
        else:
            feedback.append("\nWith some focused effort on the identified improvement areas, you can significantly enhance your understanding and performance.")
        
        return "\n".join(feedback)
        
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            
    def __del__(self):
        """Destructor to ensure connection is closed when object is garbage collected."""
        self.close()