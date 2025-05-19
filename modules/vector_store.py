import os
import json
import faiss
import numpy as np
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import platform

# Determine default device
try:
    import torch
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEFAULT_DEVICE}")
except ImportError:
    DEFAULT_DEVICE = 'cpu'
    print("PyTorch not found. Using CPU.")

class VectorStore:
    """
    Manages text embedding, FAISS index creation, caching, and searching.
    Supports multiple index types (flat, hnsw, ivf) and metrics (L2, cosine).
    Automatically selects GPU if available, otherwise CPU.
    Handles empty inputs and logs all operations.
    """
    def __init__(
        self,
        model_name: str = 'all-mpnet-base-v2',
        index_type: str = 'ivf',
        metric: str = 'cosine',
        cache_dir: str = 'vector_store_cache'
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_name = model_name
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.use_cosine = self.metric == 'cosine'

        self.cache_dir = cache_dir
        self.device = DEFAULT_DEVICE
        self.index: faiss.Index = None
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        self.logger.info(f"Loading SentenceTransformer '{self.model_name}' on {self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            self.logger.error(f"Model load failed: {e}")
            raise

        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_path = os.path.join(self.cache_dir, f'faiss_{self.index_type}_{self.metric}.index')
        self.texts_path = os.path.join(self.cache_dir, 'texts.json')
        self.meta_path = os.path.join(self.cache_dir, 'metadata.json')

    def chunk_text(self, text: str, max_words: int = 200) -> List[str]:
        if not text or not isinstance(text, str):
            self.logger.warning("Empty or invalid text provided to chunk_text.")
            return []
        words = text.split()
        overlap = max_words // 5
        step = max(1, max_words - overlap)
        chunks, pos = [], 0
        while pos < len(words):
            end = min(pos + max_words, len(words))
            chunks.append(' '.join(words[pos:end]))
            if end == len(words): break
            pos += step
        return chunks

    def build_index(
        self,
        docs: List[str],
        doc_ids: List[str] = None,
        use_multiprocess: bool = True
    ):
        self.logger.info(f"Building index: {len(docs) if docs else 0} docs, multiprocess={use_multiprocess}")
        self.texts, self.metadata = [], []

        if not docs:
            self.logger.warning("No documents provided, initializing empty index.")
            self._init_empty_index()
            return

        if doc_ids and len(doc_ids) != len(docs):
            raise ValueError("docs and doc_ids length mismatch")

        # Chunk and record metadata
        for i, doc in enumerate(docs):
            chunks = self.chunk_text(doc)
            id_ = doc_ids[i] if doc_ids else f"doc_{i}"
            for j, chunk in enumerate(chunks):
                self.texts.append(chunk)
                self.metadata.append({'doc_id': id_, 'chunk_idx': j})

        if not self.texts:
            self.logger.warning("All documents produced no chunks, initializing empty index.")
            self._init_empty_index()
            return

        # Encode
        self.logger.info(f"Encoding {len(self.texts)} chunks on {self.device}")
        embeddings = None
        if use_multiprocess and self.device == 'cpu':
            try:
                with self.model.start_multi_process_pool() as pool:
                    embeddings = self.model.encode_multi_process(self.texts, pool, batch_size=64)
            except Exception as e:
                self.logger.warning(f"Multiprocess failed: {e}")
        if embeddings is None:
            embeddings = self.model.encode(self.texts, batch_size=128 if self.device=='cuda' else 64,
                                           show_progress_bar=True)
        embeddings = np.array(embeddings, dtype='float32')

        # Normalize for cosine
        if self.use_cosine:
            faiss.normalize_L2(embeddings)

        # Create index and train IVF on real embeddings
        self.index = self._create_faiss_index()
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.logger.info("Training IVF index on real embeddings.")
            self.index.train(embeddings)

        # Set defaults for search
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(self.index.nlist, 10)

        # Add embeddings
        self.index.add(embeddings)
        self.logger.info(f"Index built, total vectors: {self.index.ntotal}")

    def _init_empty_index(self):
        self.logger.warning("Initializing empty FAISS index.")
        if self.use_cosine:
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        else:
            quant = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quant, self.dim, 1, faiss.METRIC_L2)
            self.index.nprobe = 1

    def _create_faiss_index(self) -> faiss.Index:
        metric = faiss.METRIC_INNER_PRODUCT if self.use_cosine else faiss.METRIC_L2
        if self.index_type == 'flat':
            return faiss.IndexFlat(self.dim, metric)
        if self.index_type == 'hnsw':
            idx = faiss.IndexHNSWFlat(self.dim, 32, metric)
            return idx
        # IVF
        nlist = max(1, int(np.sqrt(len(self.texts))))
        idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dim), self.dim, nlist, metric)
        return idx

    def save(self) -> bool:
        if self.index is None:
            self.logger.error("No index to save.")
            return False
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.texts_path, 'w', encoding='utf-8') as f:
                json.dump(self.texts, f, ensure_ascii=False, indent=2)
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            self.logger.info("Index, texts, and metadata saved successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False

    def load(self) -> bool:
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, 'r', encoding='utf-8') as f:
                self.texts = json.load(f)
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.logger.info("Index, texts, and metadata loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return False

    def search(self, query: str, top_k: int = 5, ef_search: int = None, nprobe: int = None) -> List[Dict[str, Any]]:
        if self.index is None:
            self.logger.error("Search attempted on uninitialized index.")
            return []
        if self.index.ntotal == 0:
            self.logger.warning("Search on empty index.")
            return []

        vec = self.model.encode([query], show_progress_bar=False)
        vec = np.array(vec, dtype='float32')
        if self.use_cosine:
            faiss.normalize_L2(vec)

        if ef_search and hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef_search
        if nprobe and hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

        D, I = self.index.search(vec.reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            record = {'text': self.texts[idx], 'metadata': self.metadata[idx]}
            record['score' if self.use_cosine else 'distance'] = float(dist)
            results.append(record)
        return results

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        return [r['text'] for r in self.search(query, top_k=top_k)]
