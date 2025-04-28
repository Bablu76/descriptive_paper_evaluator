import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from io import BytesIO
import base64
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeedbackGenerator:
    def __init__(self, db_path: str = "./data/main.db"):
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            os.makedirs("./data/reports", exist_ok=True)
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _fig_to_base64(self, fig):
        """Convert a matplotlib figure to base64 string."""
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            return img_base64
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""

    def _get_co_mapping(self) -> Dict:
        """Get the mapping of questions to course objectives."""
        try:
            self.cursor.execute('SELECT id, co_id FROM questions')
            return {row[0]: row[1] for row in self.cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error fetching CO mapping: {e}")
            return {}

    def _calculate_co_achievement(self, student_evaluations: List[Dict], co_mapping: Dict) -> tuple:
        """Calculate Course Objective achievement for a student."""
        co_scores = {}
        co_max = {}
        for eval_data in student_evaluations:
            question_id = eval_data['question_id']
            marks = eval_data['marks']
            max_marks = eval_data['max_marks']
            co_id = co_mapping.get(question_id, 'Unknown')
            if co_id not in co_scores:
                co_scores[co_id] = 0
                co_max[co_id] = 0
            co_scores[co_id] += marks
            co_max[co_id] += max_marks
        co_achievement = {
            co_id: (co_scores[co_id] / co_max[co_id] * 100) if co_max[co_id] > 0 else 0
            for co_id in co_scores
        }
        return co_achievement, co_scores, co_max

    def _calculate_class_statistics(self) -> pd.DataFrame:
        """Calculate class-wide statistics for all questions."""
        try:
            self.cursor.execute('''
            SELECT question_id, marks, max_marks FROM evaluations
            ''')
            evaluations = self.cursor.fetchall()
            df = pd.DataFrame(evaluations, columns=['question_id', 'marks', 'max_marks'])
            stats = df.groupby('question_id').agg({
                'marks': ['mean', 'median', 'min', 'max', 'std', 'count'],
                'max_marks': ['first']
            }).reset_index()
            stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
            stats['mean_percentage'] = (stats['marks_mean'] / stats['max_marks_first']) * 100
            return stats
        except sqlite3.Error as e:
            logger.error(f"Error calculating class statistics: {e}")
            return pd.DataFrame()

    def generate_instructor_report(self) -> Dict:
        """Generate a comprehensive report for the instructor."""
        try:
            stats = self._calculate_class_statistics()
            self.cursor.execute('SELECT student_id, name FROM students')
            students = self.cursor.fetchall()
            co_mapping = self._get_co_mapping()

            report = {
                "generated_at": datetime.now().isoformat(),
                "total_students": len(students),
                "question_statistics": stats.to_dict('records'),
                "student_reports": []
            }

            if not stats.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                stats = stats.sort_values('question_id_')
                x = range(len(stats))
                width = 0.35
                ax.bar([i - width/2 for i in x], stats['marks_mean'], width, label='Mean Score')
                ax.bar([i + width/2 for i in x], stats['max_marks_first'], width, label='Maximum Marks')
                ax.set_xlabel('Question ID')
                ax.set_ylabel('Marks')
                ax.set_title('Class Performance by Question')
                ax.set_xticks(x)
                ax.set_xticklabels(stats['question_id_'])
                ax.legend()
                fig.tight_layout()
                report["class_performance_plot"] = self._fig_to_base64(fig)

            co_achievement_total = {}
            co_max_total = {}
            for student_id, student_name in students:
                self.cursor.execute('''
                SELECT question_id, marks, max_marks, feedback FROM evaluations 
                WHERE student_id = ?
                ''', (student_id,))
                evaluations = [
                    {"question_id": row[0], "marks": row[1], "max_marks": row[2], "feedback": row[3]}
                    for row in self.cursor.fetchall()
                ]
                if not evaluations:
                    continue
                co_achievement, co_scores, co_max = self._calculate_co_achievement(evaluations, co_mapping)
                for co_id, score in co_scores.items():
                    if co_id not in co_achievement_total:
                        co_achievement_total[co_id] = 0
                        co_max_total[co_id] = 0
                    co_achievement_total[co_id] += score
                    co_max_total[co_id] += co_max[co_id]
                total_marks = sum(eval_data['marks'] for eval_data in evaluations)
                max_marks = sum(eval_data['max_marks'] for eval_data in evaluations)
                percentage = (total_marks / max_marks * 100) if max_marks > 0 else 0
                student_report = {
                    "student_id": student_id,
                    "name": student_name,
                    "total_marks": total_marks,
                    "max_marks": max_marks,
                    "percentage": percentage,
                    "co_achievement": co_achievement,
                    "evaluations": evaluations
                }
                report["student_reports"].append(student_report)

            with open("./data/reports/instructor_report.json", "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Generated instructor report")
            return report
        except Exception as e:
            logger.error(f"Error generating instructor report: {e}")
            return {}

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            logger.info("Closed feedback generator database connection")
        except Exception as e:
            logger.error(f"Error closing database: {e}")