import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Visualization:
    @staticmethod
    def plot_class_performance(stats: pd.DataFrame) -> str:
        """Generate a bar plot for class performance by question."""
        try:
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
            
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            logger.info("Generated class performance plot")
            return img_base64
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            return ""