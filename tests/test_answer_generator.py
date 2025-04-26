import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from components.answer_generator import ModelAnswerGenerator

# A dummy retriever for predictable snippets
class DummyRetriever:
    def search(self, query, k=5):
        return [
            {"content": f"Context about {query}", "meta": {"source": "TB.pdf", "page": 1}},
            {"content": f"More context about {query}", "meta": {"source": "TB.pdf", "page": 2}}
        ]

# A dummy LLM for override
def dummy_llm(prompt: str) -> str:
    return "DUMMY_ANSWER"

class TestModelAnswerGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use NamedTemporaryFile so Windows won't lock it
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        cls.db_path = tmp.name
        tmp.close()
        cls.mag = ModelAnswerGenerator(model_answers_db=cls.db_path, llm_backend="gemini")

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.db_path)
        except PermissionError:
            print("⚠️ Could not delete test DB file (still locked). Ignoring.")

    def test_generate_and_cache_with_override(self):
        qid = "Q_TEST"
        # first generation uses dummy_llm
        ans1 = self.mag.generate_model_answer(
            question_id=qid,
            question_text="Test question?",
            marks=2,
            co_id="CO1",
            rubric=["Point 1", "Point 2"],
            retriever=DummyRetriever(),
            llm_generate_func=dummy_llm
        )
        self.assertEqual(ans1, "DUMMY_ANSWER")

        # second call must hit cache, llm_generate_func not called
        ans2 = self.mag.generate_model_answer(
            question_id=qid,
            question_text="Changed text?",
            marks=2,
            co_id="CO1",
            rubric=["Point 1", "Point 2"],
            retriever=DummyRetriever(),
            llm_generate_func=lambda p: "WRONG"
        )
        self.assertEqual(ans2, "DUMMY_ANSWER")

    def test_sqlite_contains(self):
        # ensure the row was stored
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("SELECT question_id, model_answer FROM model_answers")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "Q_TEST")
        self.assertEqual(rows[0][1], "DUMMY_ANSWER")

    @patch.object(ModelAnswerGenerator, '_call_gemini', return_value="MOCK_GEMINI")
    def test_gemini_backend(self, mock_gemini):
        # clear DB manually
        os.remove(self.db_path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = tmp.name
        tmp.close()
        mag2 = ModelAnswerGenerator(model_answers_db=self.db_path, llm_backend="gemini")

        ans = mag2.generate_model_answer(
            question_id="Q2",
            question_text="Another test?",
            marks=1,
            co_id="CO1",
            rubric=["Point A"],
            retriever=DummyRetriever(),
            llm_generate_func=None  # triggers _call_gemini
        )
        self.assertEqual(ans, "MOCK_GEMINI")
        self.assertEqual(mock_gemini.call_count, 1)

    def test_gemini_api_diagnostic(self):
        from google import genai # google-genai client
        key = os.getenv("GEMINI_API_KEY")
        self.assertTrue(key, "GEMINI_API_KEY must be set in .env")
        self.assertTrue(hasattr(genai, "Client"), "google-genai must expose Client")
        try:
            client = genai.Client(api_key=key)
        except Exception as e:
            self.fail(f"Could not instantiate genai.Client: {e}")

if __name__ == "__main__":
    unittest.main()
