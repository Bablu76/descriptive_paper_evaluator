# components/feedback_generator.py
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from io import BytesIO
import base64

class FeedbackGenerator:
    def __init__(self, docs_db="./data/documents.db", 
                student_db="./data/student_answers.db",
                evaluation_db="./data/evaluations.db"):
        # Connect to the necessary databases
        self.docs_conn = sqlite3.connect(docs_db)
        self.docs_cursor = self.docs_conn.cursor()
        
        self.student_conn = sqlite3.connect(student_db)
        self.student_cursor = self.student_conn.cursor()
        
        self.eval_conn = sqlite3.connect(evaluation_db)
        self.eval_cursor = self.eval_conn.cursor()
        
        # Ensure output directory exists
        os.makedirs("./data/reports", exist_ok=True)
    
    def _fig_to_base64(self, fig):
        """Convert a matplotlib figure to base64 string for embedding in HTML"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _get_co_mapping(self):
        """Get the mapping of questions to course objectives"""
        self.docs_cursor.execute('SELECT id, co_id FROM questions')
        co_mapping = {row[0]: row[1] for row in self.docs_cursor.fetchall()}
        return co_mapping
    
    def _calculate_co_achievement(self, student_evaluations, co_mapping):
        """Calculate Course Objective achievement for a student"""
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
        
        # Calculate percentages
        co_achievement = {}
        for co_id in co_scores:
            if co_max[co_id] > 0:
                co_achievement[co_id] = (co_scores[co_id] / co_max[co_id]) * 100
            else:
                co_achievement[co_id] = 0
                
        return co_achievement, co_scores, co_max
    
    def _calculate_class_statistics(self):
        """Calculate class-wide statistics for all questions"""
        # Get all evaluations
        self.eval_cursor.execute('''
        SELECT question_id, marks, max_marks FROM evaluations
        ''')
        
        evaluations = self.eval_cursor.fetchall()
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame(evaluations, columns=['question_id', 'marks', 'max_marks'])
        
        # Calculate statistics by question
        stats = df.groupby('question_id').agg({
            'marks': ['mean', 'median', 'min', 'max', 'std', 'count'],
            'max_marks': ['first']
        }).reset_index()
        
        # Flatten the column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        
        # Calculate percentage scores
        stats['mean_percentage'] = (stats['marks_mean'] / stats['max_marks_first']) * 100
        
        return stats
    
    def generate_instructor_report(self):
        """Generate a comprehensive report for the instructor"""
        # Get class statistics
        stats = self._calculate_class_statistics()
        
        # Get all students
        self.student_cursor.execute('SELECT student_id, name FROM students')
        students = self.student_cursor.fetchall()
        
        # Get CO mapping
        co_mapping = self._get_co_mapping()
        
        # Create overall report
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_students": len(students),
            "question_statistics": stats.to_dict('records'),
            "student_reports": []
        }
        
        # Generate visualization for class performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by question_id for consistent visualization
        stats = stats.sort_values('question_id_')
        
        x = np.arange(len(stats))
        width = 0.35
        
        ax.bar(x - width/2, stats['marks_mean'], width, label='Mean Score')
        ax.bar(x + width/2, stats['max_marks_first'], width, label='Maximum Marks')
        
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Marks')
        ax.set_title('Class Performance by Question')
        ax.set_xticks(x)
        ax.set_xticklabels(stats['question_id_'])
        ax.legend()
        
        fig.tight_layout()
        
        # Convert plot to base64 for embedding
        report["class_performance_plot"] = self._fig_to_base64(fig)
        
        # Calculate CO achievement across all students
        co_achievement_total = {}
        co_max_total = {}
        
        for student_id, student_name in students:
            # Get all evaluations for this student
            self.eval_cursor.execute('''
            SELECT question_id, marks, max_marks, feedback FROM evaluations 
            WHERE student_id = ?
            ''', (student_id,))
            
            evaluations = [
                {"question_id": row[0], "marks": row[1], "max_marks": row[2], "feedback": row[3]}
                for row in self.eval_cursor.fetchall()
            ]
            
            # Skip if no evaluations
            if not evaluations:
                continue
                
            # Calculate CO achievement
            co_achievement, co_scores, co_max = self._calculate_co_achievement(
                evaluations, co_mapping
            )
            
            # Add to totals
            for co_id, score in co_scores.items():
                if co_id not in co_achievement_total:
                    co_achievement_total[co_id] = 0
                    co_max_total[co_id] = 0
                    
                co_achievement_total[co_id] += score
                co_max_total[co_id] += co_max[co_id]
            
            # Calculate total marks
            total_marks = sum(eval_data['marks'] for eval_data in evaluations)
            max_marks = sum(eval_data['max_marks'] for eval_data in evaluations)
            percentage = (total_marks / max_marks * 100) if max_marks > 0 else 0
            
            # Add student report
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
