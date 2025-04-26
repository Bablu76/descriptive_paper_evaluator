import unittest
from components.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.dp = DocumentProcessor()

    def test_syllabus_parse(self):
        result = self.dp.parse_syllabus_json("data/syllabus.json")
        self.assertIn("course_info", result)
        self.assertTrue(result["outcomes"])

    def test_question_parse(self):
        questions = self.dp.parse_question_paper_json("data/questions.json")
        self.assertGreater(len(questions), 5)
        self.assertIn("question_text", questions[0])

    def test_pdf_extract_and_chunk(self):
        pages = self.dp.extract_text_from_pdf("data/textbook.pdf")
        self.assertGreater(len(pages), 0)
        index = self.dp.chunk_and_embed_texts(pages[:5])
        self.assertTrue(index.is_trained)
        self.assertGreater(len(self.dp.chunks), 0)

    def test_search_returns_results(self):
        pages = self.dp.extract_text_from_pdf("data/textbook.pdf")
        self.dp.chunk_and_embed_texts(pages[:5])
        results = self.dp.search("dynamic programming")
        self.assertGreater(len(results), 0)

if __name__ == "__main__":
    unittest.main()
