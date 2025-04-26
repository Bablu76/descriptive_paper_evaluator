import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from transformers import pipeline

class AnswerGenerator:
    """Answer Generator using open-source LLMs with prompt-based RAG."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS model_answers (
        question_id TEXT PRIMARY KEY,
        model_answer TEXT,
        word_count INTEGER,
        created_at TEXT
    )
    """

    def __init__(
        self,
        model_answers_db="./data/model_answers.db",
        llm_backend="hf",
        hf_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    ):
        self.db_path = model_answers_db
        self.backend = llm_backend
        self.model_name = hf_model

        # HuggingFace local or downloaded model
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            device=-1  # CPU, change to 0 for GPU
        )

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _ensure_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self.CREATE_TABLE)
            conn.commit()

    def _fetch_cached(self, qid):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT model_answer FROM model_answers WHERE question_id=?", (qid,)
            )
            row = cur.fetchone()
        return row[0] if row else None

    def _cache_answer(self, qid, answer):
        wc = len(answer.split())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO model_answers
                   (question_id, model_answer, word_count, created_at)
                   VALUES (?, ?, ?, ?)""",
                (qid, answer, wc, datetime.utcnow().isoformat())
            )
            conn.commit()

    def _build_prompt(self, question_text, marks, co_id, rubric, context_chunks):
        prompt = (
            f"You are an academic assistant.\n\n"
            f"Course Outcome: {co_id}\n"
            f"Question ({marks} marks): {question_text}\n\n"
            f"Rubric:\n"
        )
        for r in rubric:
            prompt += f"- {r}\n"
        prompt += "\nRelevant Course Content:\n"
        for chunk in context_chunks:
            src = chunk["meta"]["source"]
            pg = chunk["meta"]["page"]
            txt = chunk["content"].strip().replace("\n", " ")
            prompt += f"[{src} p{pg}]: {txt[:300]}...\n"
        prompt += f"\nWrite a comprehensive answer (~{marks*150} words) covering all rubric points clearly and aligning with {co_id}.\n"
        return prompt

    def generate_model_answer(
        self,
        question_id: str,
        question_text: str,
        marks: int,
        co_id: str,
        rubric: list,
        retriever,
        top_k=5,
        llm_generate_func=None
    ) -> str:
        # Step 1: Check cache
        cached = self._fetch_cached(question_id)
        if cached:
            return cached

        # Step 2: Retrieve context
        chunks = retriever.search(question_text, k=top_k)

        # Step 3: Build prompt
        prompt = self._build_prompt(question_text, marks, co_id, rubric, chunks)

        # Step 4: Generate answer
        if llm_generate_func:
            result = llm_generate_func(prompt)
        else:
            result = self.generator(prompt, max_new_tokens=1024)[0]["generated_text"]

        # Step 5: Store and return
        self._cache_answer(question_id, result)
        return result
