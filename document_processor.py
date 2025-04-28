import json
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", metric: str = "cosine"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.metric = metric
        self.questions = []
        self.syllabus = {}
        self.documents = []
        self.text_chunks = []
        self.chunk_sources = []
        self.index = None

    def load_questions(self, file_path: str):
        """Load questions from a JSON file."""
        try:
            with open(file_path, "r") as file:
                self.questions = json.load(file)
            logger.info(f"Loaded {len(self.questions)} questions from {file_path}")
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise

    def load_syllabus(self, file_path: str):
        """Load syllabus from a JSON file."""
        try:
            with open(file_path, "r") as file:
                self.syllabus = json.load(file)
            logger.info(f"Loaded syllabus from {file_path}")
        except Exception as e:
            logger.error(f"Error loading syllabus: {e}")
            raise

    def extract_text_from_pdfs(self, pdf_files: List[str]):
        """Extract text from PDF files using PyMuPDF."""
        self.documents = []
        for pdf_file in pdf_files:
            try:
                doc = fitz.open(pdf_file)
                pdf_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    pdf_text += page.get_text("text") + "\n"
                self.documents.append({"text": pdf_text, "source": pdf_file, "page": page_num + 1})
                doc.close()
                logger.info(f"Extracted text from {pdf_file}")
            except Exception as e:
                logger.error(f"Error extracting text from {pdf_file}: {e}")

    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50):
        """Chunk the extracted document text into smaller chunks."""
        self.text_chunks = []
        self.chunk_sources = []
        for doc in self.documents:
            text = doc["text"]
            source = doc["source"]
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    self.text_chunks.append(chunk)
                    self.chunk_sources.append({"source": source, "page": doc["page"]})
        logger.info(f"Created {len(self.text_chunks)} chunks")

    def embed_chunks(self) -> np.ndarray:
        """Embed the chunks of text into embeddings."""
        try:
            embeddings = self.embedding_model.encode(self.text_chunks, convert_to_numpy=True)
            logger.info(f"Embedded {len(self.text_chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            raise

    def build_faiss_index(self):
        """Build a FAISS index from the embeddings of text chunks."""
        try:
            embeddings = self.embed_chunks()
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            logger.info("Built FAISS index")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def save_index(self, index_file: str, source_file: str):
        """Save the FAISS index and chunk sources to files."""
        try:
            faiss.write_index(self.index, index_file)
            with open(source_file, "wb") as f:
                pickle.dump(self.chunk_sources, f)
            logger.info(f"Saved FAISS index to {index_file} and sources to {source_file}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def load_index(self, index_file: str, source_file: str):
        """Load a FAISS index and chunk sources from files."""
        try:
            self.index = faiss.read_index(index_file)
            with open(source_file, "rb") as f:
                self.chunk_sources = pickle.load(f)
            logger.info(f"Loaded FAISS index from {index_file} and sources from {source_file}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def analyze_co_distribution(self) -> Dict:
        """Analyze the distribution of questions across Course Outcomes (COs)."""
        try:
            co_counts = {}
            co_marks = {}
            co_questions = {}
            co_definitions = {
                f"co{i+1}": outcome 
                for i, outcome in enumerate(self.syllabus.get("Course Outcomes", []))
            }

            for q in self.questions:
                co = q.get("co_id", "").lower()
                if co:
                    co_counts[co] = co_counts.get(co, 0) + 1
                    co_marks[co] = co_marks.get(co, 0) + int(q.get("marks", 0))
                    if co not in co_questions:
                        co_questions[co] = []
                    co_questions[co].append(q["question_number"])

            total_questions = len(self.questions)
            total_marks = sum(int(q.get("marks", 0)) for q in self.questions)

            analysis = {
                "co_definitions": co_definitions,
                "co_counts": co_counts,
                "co_marks": co_marks,
                "co_questions": co_questions,
                "total_questions": total_questions,
                "total_marks": total_marks,
                "co_question_percentage": {
                    co: (count / total_questions) * 100 
                    for co, count in co_counts.items()
                },
                "co_marks_percentage": {
                    co: (marks / total_marks) * 100 
                    for co, marks in co_marks.items()
                }
            }
            logger.info("Completed CO distribution analysis")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing CO distribution: {e}")
            raise