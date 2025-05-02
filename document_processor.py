import os
import json
import logging
from typing import List
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    """
    Processes textbook PDF files to build a searchable FAISS index for course content.
    Loads the static syllabus (syllabus.json) and embeds text chunks with SentenceTransformer.
    """
    def __init__(self, textbook_dir: str = "textbooks", index_path: str = "corpus.index",
                 syllabus_path: str = "syllabus.json"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.textbook_dir = textbook_dir
        self.index_path = index_path
        self.syllabus_path = syllabus_path
        
        # Load static syllabus
        self.syllabus = self.load_syllabus()
        
        # Initialize embedding model for context retrieval
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.text_chunks = []  # maps index ids to text chunks
        
        # Build or load FAISS index
        if os.path.exists(self.index_path):
            self.logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            self.build_index()
    
    def load_syllabus(self):
        """
        Load syllabus information from a JSON file (static).
        """
        try:
            with open(self.syllabus_path, 'r', encoding='utf-8') as f:
                syllabus = json.load(f)
            self.logger.info("Syllabus loaded successfully.")
            return syllabus
        except Exception as e:
            self.logger.error(f"Failed to load syllabus: {e}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        """
        text = ""
        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f"Extracting text from {pdf_path}")
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, max_words: int = 200) -> List[str]:
        """
        Chunk large text into smaller pieces (~max_words each).
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks
    
    def build_index(self):
        """
        Build FAISS index: extract text from PDFs, chunk it, embed chunks, and index.
        """
        self.logger.info("Building FAISS index from textbook PDFs.")
        all_chunks = []
        for root, dirs, files in os.walk(self.textbook_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    text = self.extract_text_from_pdf(pdf_path)
                    chunks = self.chunk_text(text)
                    all_chunks.extend(chunks)
        
        self.text_chunks = all_chunks
        if not all_chunks:
            self.logger.warning("No text chunks found. FAISS index will be empty.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            return
        
        # Compute embeddings for all chunks
        self.logger.info("Computing embeddings for text chunks.")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        
        # Build FAISS index (L2 distance)
        self.logger.info("Creating FAISS index.")
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)
        self.index = index
        
        # Save index
        faiss.write_index(index, self.index_path)
        self.logger.info(f"FAISS index built and saved to {self.index_path}.")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k relevant text chunks from the FAISS index for a given query.
        """
        if self.index is None:
            self.logger.warning("FAISS index not found. No context retrieved.")
            return []
        query_vec = self.embedding_model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])
        return results

if __name__ == "__main__":
    # Example: build the index from PDFs in 'textbooks/'
    processor = DocumentProcessor()
    processor.build_index()
