import os
import sqlite3
from datetime import datetime
import logging
from transformers import pipeline
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generate model answers using a lightweight LLM without RAG."""

    def __init__(
        self,
        db_path: str = "./data/main.db",
        hf_model: str = "google/flan-t5-base"
    ):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self._create_tables()
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        try:
            self.generator = pipeline(
                "text-generation",
                model=hf_model,
                tokenizer=hf_model,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.2,
                device=-1  # CPU
            )
            logger.info(f"Initialized {hf_model} for answer generation")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def _create_tables(self):
        """Create model answers table."""
        try:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_answers (
                question_id TEXT PRIMARY KEY,
                model_answer TEXT,
                word_count INTEGER,
                created_at TEXT
            )
            ''')
            self.conn.commit()
            logger.info("Created model answers table")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
            raise

    def _fetch_cached(self, qid: str) -> Optional[str]:
        """Fetch cached model answer."""
        try:
            self.cursor.execute(
                "SELECT model_answer FROM model_answers WHERE question_id = ?",
                (qid,)
            )
            row = self.cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching cached answer for {qid}: {e}")
            return None

    def _cache_answer(self, qid: str, answer: str):
        """Cache generated answer."""
        try:
            wc = len(answer.split())
            self.cursor.execute('''
            INSERT OR REPLACE INTO model_answers
            (question_id, model_answer, word_count, created_at)
            VALUES (?, ?, ?, ?)
            ''', (qid, answer, wc, datetime.utcnow().isoformat()))
            self.conn.commit()
            logger.info(f"Cached answer for question {qid}")
        except sqlite3.Error as e:
            logger.error(f"Error caching answer for {qid}: {e}")

    def _build_prompt(self, question_text: str, marks: int, co_id: str, rubric: List[str]) -> str:
        """Build prompt for model answer generation."""
        prompt = (
            f"Question ({marks} marks): {question_text}\n"
            f"Course Outcome: {co_id}\n"
            f"Rubric:\n" + "\n".join(f"- {r}" for r in rubric) + "\n"
            f"Generate a concise answer (~{marks*50} words) covering all rubric points."
        )
        return prompt

    def generate_model_answer(
        self,
        question_id: str,
        question_text: str,
        marks: int,
        co_id: str,
        rubric: List[str]
    ) -> Optional[str]:
        """Generate and cache a model answer."""
        try:
            cached = self._fetch_cached(question_id)
            if cached:
                logger.info(f"Using cached answer for question {question_id}")
                return cached

            prompt = self._build_prompt(question_text, marks, co_id, rubric)
            result = self.generator(prompt, max_new_tokens=512)[0]["generated_text"]
            answer = result[len(prompt):].strip() if result.startswith(prompt) else result.strip()

            self._cache_answer(question_id, answer)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer for {question_id}: {e}")
            return None

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            logger.info("Closed answer generator database connection")
        except Exception as e:
            logger.error(f"Error closing database: {e}")