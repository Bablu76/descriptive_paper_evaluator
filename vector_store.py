import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.metadata = []

    def build_index(self, texts: List[str], metadata: List[Dict]):
        """Build FAISS index from text chunks."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            self.chunks = texts
            self.metadata = metadata
            logger.info(f"Built FAISS index with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for top-k similar chunks."""
        try:
            query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_emb, k)
            results = [
                {
                    "content": self.chunks[idx],
                    "meta": self.metadata[idx],
                    "distance": float(distances[0][i])
                }
                for i, idx in enumerate(indices[0]) if idx < len(self.chunks)
            ]
            logger.info(f"Performed search for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []

    def save(self, index_file: str, metadata_file: str):
        """Save FAISS index and metadata."""
        try:
            faiss.write_index(self.index, index_file)
            with open(metadata_file, "wb") as f:
                pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
            logger.info(f"Saved FAISS index to {index_file} and metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def load(self, index_file: str, metadata_file: str):
        """Load FAISS index and metadata."""
        try:
            self.index = faiss.read_index(index_file)
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.metadata = data["metadata"]
            logger.info(f"Loaded FAISS index from {index_file} and metadata from {metadata_file}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise