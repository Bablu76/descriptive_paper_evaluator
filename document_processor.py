import os
import json
import logging
from typing import List
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DocumentProcessor:
    """
    Processes textbook PDF files to build a searchable FAISS index for course content.
    Loads the static syllabus (syllabus.json) and embeds text chunks with SentenceTransformer.
    """
    def __init__(self, textbook_dir: str = "textbooks", index_path: str = "corpus.index",
                 syllabus_path: str = "syllabus.json", index_type: str = "flat"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.textbook_dir = textbook_dir
        self.index_path = index_path
        self.index_type = index_type
        self.syllabus_path = syllabus_path
        
        # Load syllabus outcomes (static JSON)
        self.syllabus = self.load_syllabus()
        
        # Initialize SentenceTransformer (CPU only) for embeddings
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.text_chunks = []  # list of text chunks (strings)
        self.metadata = []     # list of dicts with metadata (file, page, chunk index)
        
        # Prepare cache directory
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Detect file changes: record current PDFs with their modification times
        state_file = os.path.join(self.cache_dir, "file_index.json")
        current_files = {}
        for root, dirs, files in os.walk(self.textbook_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    try:
                        current_files[pdf_path] = os.path.getmtime(pdf_path)
                    except OSError:
                        current_files[pdf_path] = None
        
        # Compare to saved state; if identical, skip rebuilding
        need_rebuild = True
        if os.path.exists(self.index_path) and os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                if saved_state.get("files") == current_files:
                    need_rebuild = False
                    self.logger.info("No changes in PDF files; loading existing index and cache.")
                    # Load existing FAISS index
                    self.index = faiss.read_index(self.index_path)
                    # Load cached chunks and metadata
                    try:
                        with open(os.path.join(self.cache_dir, "text_chunks.json"), 'r', encoding='utf-8') as f:
                            self.text_chunks = json.load(f)
                        with open(os.path.join(self.cache_dir, "metadata.json"), 'r', encoding='utf-8') as f:
                            self.metadata = json.load(f)
                    except Exception as e:
                        self.logger.error(f"Failed to load cached data: {e}")
                        need_rebuild = True
            except Exception as e:
                self.logger.warning(f"Could not read state file: {e}")
        
        # Save current file state for use after building
        self.current_files = current_files
        if need_rebuild:
            self.build_index()
    
    def load_syllabus(self):
        """
        Load syllabus (course outcomes) from JSON. Returns dict.
        """
        try:
            with open(self.syllabus_path, 'r', encoding='utf-8') as f:
                syllabus = json.load(f)
            self.logger.info("Syllabus loaded.")
            return syllabus
        except Exception as e:
            self.logger.error(f"Failed to load syllabus: {e}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract full text from a PDF using PyMuPDF.
        """
        text = ""
        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f"Extracting text from {pdf_path}")
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, max_words: int = 200) -> List[str]:
        """
        Split text into overlapping chunks of ~max_words.
        """
        words = text.split()
        if not words:
            return []
        overlap = max_words // 5  # 20% overlap
        step = max_words - overlap
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
            if i + max_words >= len(words):
                break
        return chunks
    
    def build_index(self):
        """
        Build the FAISS index: extract text from PDFs, chunk it, compute embeddings, and index.
        """
        self.logger.info("Building FAISS index from PDFs.")
        self.text_chunks = []
        self.metadata = []
        # Loop over PDFs and pages
        for root, dirs, files in os.walk(self.textbook_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    try:
                        doc = fitz.open(pdf_path)
                        self.logger.info(f"Processing {pdf_path}")
                        for page in doc:
                            page_text = page.get_text()
                            chunks = self.chunk_text(page_text, max_words=200)
                            for i, chunk in enumerate(chunks):
                                self.text_chunks.append(chunk)
                                self.metadata.append({
                                    "file": os.path.relpath(pdf_path, self.textbook_dir),
                                    "page": page.number,
                                    "chunk": i
                                })
                    except Exception as e:
                        self.logger.error(f"Error processing {pdf_path}: {e}")
        if not self.text_chunks:
            self.logger.warning("No text chunks extracted; creating empty index.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # Compute embeddings (parallel)
            self.logger.info("Encoding text chunks with SentenceTransformer.")
            try:
                pool = self.embedding_model.start_multi_process_pool()
                embeddings = self.embedding_model.encode_multi_process(
                    self.text_chunks, pool,
                    batch_size=64, show_progress_bar=True, device='cpu'
                )
                self.embedding_model.stop_multi_process_pool(pool)
            except Exception as e:
                self.logger.warning(f"Multi-process encoding failed ({e}); using single-process.")
                embeddings = self.embedding_model.encode(
                    self.text_chunks, batch_size=64, show_progress_bar=True, device='cpu'
                )
            embeddings = np.array(embeddings).astype("float32")
            
            # Build the specified FAISS index
            self.logger.info(f"Creating FAISS index of type '{self.index_type}'.")
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.embedding_dim)
                index.add(embeddings)
            elif self.index_type == "hnsw":
                M = 32
                index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
                index.add(embeddings)
            elif self.index_type == "ivf":
                nlist = max(1, min(100, len(embeddings)//10))
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
                self.logger.info(f"Training IVF index with nlist={nlist}.")
                index.train(embeddings)
                index.add(embeddings)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            self.index = index
        
        # Save the index to disk
        try:
            faiss.write_index(self.index, self.index_path)
            self.logger.info(f"FAISS index saved to {self.index_path}.")
        except Exception as e:
            self.logger.error(f"Failed to write FAISS index: {e}")
        
        # Cache chunks and metadata
        try:
            with open(os.path.join(self.cache_dir, "text_chunks.json"), 'w', encoding='utf-8') as f:
                json.dump(self.text_chunks, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.cache_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.cache_dir, "file_index.json"), 'w', encoding='utf-8') as f:
                json.dump({"files": self.current_files}, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the top-k most similar text chunks for a query using the FAISS index.
        """
        if self.index is None:
            self.logger.warning("FAISS index not available; cannot retrieve context.")
            return []
        if not self.text_chunks:
            self.logger.warning("No text chunks in memory; load the index or rebuild.")
            return []
        query_vec = self.embedding_model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build or query the textbook index")
    parser.add_argument("--build", action="store_true", help="Build/rebuild the FAISS index")
    parser.add_argument("--query", type=str, help="Query string to retrieve context")
    parser.add_argument("--topk", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--index_type", type=str, choices=["flat","hnsw","ivf"], default="flat",
                        help="Type of FAISS index to build")
    args = parser.parse_args()
    processor = DocumentProcessor(index_type=args.index_type)
    if args.build:
        processor.build_index()
    if args.query:
        results = processor.retrieve_context(args.query, top_k=args.topk)
        print(f"Top {len(results)} results for query '{args.query}':")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk[:200]}...")

# CLI Interface and Fault Tolerance
# A new command-line interface allows standalone operation. Using Python’s argparse, the script supports flags --build to (re)build the index and --query "some text" to retrieve the top-K relevant chunks. For example:
# bash
# Copy
# Edit
# python document_processor.py --build            # build or rebuild the index
# python document_processor.py --query "What is AI?" --topk 5