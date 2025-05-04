# --- START OF FILE vector_store.py ---

import os
import json
import faiss
import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # For progress bars
import platform # To check OS potentially

# --- Check for PyTorch and CUDA availability ---
try:
    import torch
    if torch.cuda.is_available():
        # Consider setting device based on most free memory if multiple GPUs,
        # but for simplicity, using default cuda device (usually device 0)
        DEFAULT_DEVICE = 'cuda'
        print("CUDA (GPU) detected. Using GPU.")
    else:
        DEFAULT_DEVICE = 'cpu'
        print("CUDA not available. Using CPU.")
except ImportError:
    DEFAULT_DEVICE = 'cpu'
    print("PyTorch not found. Using CPU.")
# ----------------------------------------------


class VectorStore:
    """
    Manages text embedding, FAISS index creation, caching, and searching.
    Supports different index types and multi-process embedding.
    Automatically selects GPU if available, otherwise CPU.
    """
    def __init__(self,
                 model_name: str = 'all-mpnet-base-v2',
                 index_type: str = 'flat', # flat, hnsw, ivf
                 metric: str = 'l2', # l2 or cosine (IP)
                 cache_dir: str = 'vector_store_cache',
                 # Device selection is now automatic based on import checks
                 ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_name = model_name
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.use_cosine = (self.metric == 'cosine')

        self.cache_dir = cache_dir
        self.device = DEFAULT_DEVICE # Use the globally determined device
        self.index = None
        self.texts = []
        self.metadata = []

        self.logger.info(f"Initializing SentenceTransformer model '{self.model_name}' on device '{self.device}'")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            self.logger.error(f"Failed to initialize SentenceTransformer model: {e}")
            self.logger.error("Ensure the model name is correct and dependencies (like PyTorch) are installed.")
            raise # Re-raise exception as this is critical

        self.index_path = os.path.join(self.cache_dir, f'faiss_{self.index_type}_{self.metric}.index')
        self.texts_path = os.path.join(self.cache_dir, 'texts.json')
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')

        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"VectorStore initialized. Index type: {self.index_type}, Metric: {self.metric}, Device: {self.device}, Cache: {self.cache_dir}")

    def chunk_text(self, text: str, max_words: int = 200) -> List[str]:
        """Split text into potentially overlapping chunks."""
        # [Chunking logic remains the same - No changes needed here]
        words = text.split()
        if not words:
            return []
        overlap = max(0, max_words // 5)
        step = max(1, max_words - overlap)
        chunks = []
        current_pos = 0
        while current_pos < len(words):
            chunk_end = min(current_pos + max_words, len(words))
            chunk = " ".join(words[current_pos:chunk_end])
            chunks.append(chunk)
            if chunk_end == len(words): # Reached the end
                break
            current_pos += step
            if current_pos >= len(words): # Ensure step didn't overshoot the end entirely
                 break

        # More robust check for the absolute last words if missed
        # This is tricky with overlap. The loop above should handle it.
        # Verify if the last word of the original text is in the last chunk.
        # If not words[-1] in chunks[-1].split() if chunks else False: ...add last chunk

        return chunks


    def build_index(self, docs: List[str], doc_ids: List[str] = None, use_multi_process: bool = True):
        """
        Builds the FAISS index from scratch using provided documents.

        Args:
            docs (List[str]): List of raw text documents.
            doc_ids (List[str], optional): List of identifiers for each doc. Defaults to None.
            use_multi_process (bool): Whether to attempt multi-process embedding (recommended for CPU).
                                      Not applicable / potentially problematic on GPU depending on setup.
                                      If device is 'cuda', this might be ignored or forced off internally
                                      by SentenceTransformer or may cause issues. Best practice is often
                                      to use single process for GPU batching.
                                      Let's disable it automatically for GPU for safety.
        """
        self.logger.info(f"Starting index build. Docs: {len(docs)}, Requested multi-process: {use_multi_process}, Device: {self.device}")
        # Reset stored data
        self.texts = []
        self.metadata = []

        if doc_ids and len(docs) != len(doc_ids):
            raise ValueError("Length of 'docs' and 'doc_ids' must match.")

        # 1. Chunk all documents
        self.logger.info("Chunking documents...")
        # [Chunking loop remains the same]
        for i, doc_text in enumerate(tqdm(docs, desc="Chunking")):
            chunks = self.chunk_text(doc_text)
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            for chunk_idx, chunk in enumerate(chunks):
                self.texts.append(chunk)
                self.metadata.append({'doc_id': doc_id, 'chunk_idx': chunk_idx})


        if not self.texts:
            # [Empty index creation logic remains the same]
            self.logger.warning("No text chunks generated. Creating an empty index.")
            if self.use_cosine: self.index = faiss.IndexFlatIP(self.dim)
            elif self.index_type == "flat": self.index = faiss.IndexFlatL2(self.dim)
            elif self.index_type == "hnsw": self.index = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_L2)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dim); self.index = faiss.IndexIVFFlat(quantizer, self.dim, 1, faiss.METRIC_L2)
            else: self.index = faiss.IndexFlatL2(self.dim)
            return


        # 2. Compute embeddings
        self.logger.info(f"Encoding {len(self.texts)} text chunks using device: {self.device}")
        embeddings = None

        # --- Multi-process Strategy Update ---
        # Automatically disable multi-process if on GPU unless specifically handled/tested
        # Or if on Windows where it can be problematic sometimes
        can_use_multi_process = use_multi_process and self.device == 'cpu' #and platform.system() != "Windows"
        # Note: SentenceTransformer v3 might handle pool better on Windows. Re-evaluate if needed.
        self.logger.info(f"Multi-process encoding enabled: {can_use_multi_process}")


        if can_use_multi_process:
            try:
                self.logger.info("Attempting multi-process encoding (CPU)...")
                # Define pool context here for resource management
                with self.model.start_multi_process_pool() as pool:
                     embeddings = self.model.encode_multi_process(
                         self.texts, pool,
                         batch_size=64 # Adjust batch size based on available RAM per worker
                    )
                self.logger.info("Multi-process encoding successful.")
            except RuntimeError as e:
                 # Catch potential issues like 'cannot start new processes after Arora processes is True'
                 self.logger.warning(f"Multi-process encoding failed with RuntimeError ({e}). Falling back to single-process.")
                 embeddings = None
            except Exception as e:
                self.logger.warning(f"Multi-process encoding failed ({e}). Falling back to single-process.")
                embeddings = None

        if embeddings is None: # Fallback or if multi-process was disabled/failed
             self.logger.info(f"Using single-process encoding (Device: {self.device})...")
             embeddings = self.model.encode(
                 self.texts,
                 batch_size=128 if self.device == 'cuda' else 64, # Larger batch for GPU typical
                 show_progress_bar=True,
                 # Device is already set on the model, no need to pass here explicitly for self.model.encode
                 # device=self.device
            )
        # ---------------------------------------

        embeddings = np.array(embeddings, dtype='float32')
        self.logger.info(f"Embeddings computed, shape: {embeddings.shape}")

        # [Normalization and FAISS index creation logic remains the same - No changes needed here]
        # 3. Normalize for Cosine Similarity if needed
        if self.use_cosine:
            self.logger.info("Normalizing embeddings for Cosine Similarity (IP index).")
            faiss.normalize_L2(embeddings)

        # 4. Create and train/add to FAISS index
        index = None
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self.use_cosine else faiss.METRIC_L2

        if self.index_type == "flat":
            self.logger.info(f"Creating IndexFlat{'IP' if self.use_cosine else 'L2'}.")
            index = faiss.IndexFlat(self.dim, faiss_metric)

        elif self.index_type == "hnsw":
            M = 32; ef_construction = 200
            self.logger.info(f"Creating IndexHNSWFlat (M={M}, efConstruction={ef_construction}, Metric={faiss_metric}).")
            index = faiss.IndexHNSWFlat(self.dim, M, faiss_metric)
            index.hnsw.efConstruction = ef_construction

        elif self.index_type == "ivf":
             nlist = max(1, min(100, int(np.sqrt(len(self.texts)))))
             self.logger.info(f"Creating IndexIVFFlat (nlist={nlist}, Metric={faiss_metric}).")
             quantizer = faiss.IndexFlat(self.dim, faiss_metric)
             index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss_metric)
             if not index.is_trained and embeddings.shape[0] > 0:
                 self.logger.info(f"Training IVF index with {embeddings.shape[0]} vectors...")
                 index.train(embeddings)
             else: self.logger.info("Index already trained or no data to train on.")
             index.nprobe = min(nlist, 10)

        else: # Default fallback
             self.logger.warning(f"Unknown index_type '{self.index_type}'. Defaulting to IndexFlat{'IP' if self.use_cosine else 'L2'}.")
             index = faiss.IndexFlat(self.dim, faiss_metric)

        # Add embeddings to the index
        self.logger.info(f"Adding {embeddings.shape[0]} vectors to the index.")
        index.add(embeddings)
        self.index = index
        self.logger.info(f"FAISS index built successfully. Index is trained: {self.index.is_trained}, Total vectors: {self.index.ntotal}")

    def save(self):
        # [Save logic remains the same]
        if self.index is None: self.logger.error("Cannot save, index is not built."); return False
        self.logger.info(f"Saving index to {self.index_path}")
        try: faiss.write_index(self.index, self.index_path)
        except Exception as e: self.logger.error(f"Failed to write FAISS index: {e}"); return False
        self.logger.info(f"Saving texts ({len(self.texts)} chunks) to {self.texts_path}")
        try:
            with open(self.texts_path, 'w', encoding='utf-8') as f: json.dump(self.texts, f, ensure_ascii=False, indent=2)
        except Exception as e: self.logger.error(f"Failed to save text chunks: {e}"); return False
        self.logger.info(f"Saving metadata ({len(self.metadata)} items) to {self.metadata_path}")
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f: json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e: self.logger.error(f"Failed to save metadata: {e}"); return False
        self.logger.info("VectorStore saved successfully."); return True


    def load(self) -> bool:
        # [Load logic remains the same]
        if not os.path.exists(self.index_path): self.logger.error(f"Index file not found: {self.index_path}"); return False
        if not os.path.exists(self.texts_path): self.logger.error(f"Texts file not found: {self.texts_path}"); return False
        if not os.path.exists(self.metadata_path): self.logger.error(f"Metadata file not found: {self.metadata_path}"); return False
        self.logger.info(f"Loading index from {self.index_path}")
        try:
            self.index = faiss.read_index(self.index_path)
            self.logger.info(f"Index loaded. Type: {type(self.index)}, Trained: {self.index.is_trained}, Vectors: {self.index.ntotal}, Dim: {self.index.d}")
            if self.index.d != self.dim: self.logger.warning(f"Loaded index dim ({self.index.d}) mismatches model dim ({self.dim}).")
        except Exception as e: self.logger.error(f"Failed to load FAISS index: {e}"); self.index = None; return False
        self.logger.info(f"Loading texts from {self.texts_path}")
        try:
            with open(self.texts_path, 'r', encoding='utf-8') as f: self.texts = json.load(f)
        except Exception as e: self.logger.error(f"Failed to load text chunks: {e}"); self.texts = []; return False
        self.logger.info(f"Loading metadata from {self.metadata_path}")
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f: self.metadata = json.load(f)
        except Exception as e: self.logger.error(f"Failed to load metadata: {e}"); self.metadata = []; return False
        if len(self.texts) != self.index.ntotal or len(self.metadata) != self.index.ntotal:
             self.logger.warning(f"Inconsistency: Index ntotal ({self.index.ntotal}), texts ({len(self.texts)}), metadata ({len(self.metadata)}).")
        self.logger.info(f"VectorStore loaded successfully. Found {self.index.ntotal} items."); return True

    def search(self, query: str, top_k: int = 5, ef_search: int = 50, nprobe: int = 10) -> List[dict]:
        # [Search logic remains largely the same, ensures device is used for query embedding]
        if self.index is None: self.logger.error("Index not built/loaded."); return []
        if self.index.ntotal == 0: self.logger.warning("Search on empty index."); return []

        # 1. Embed the query using the correct device
        query_vec = self.model.encode([query], batch_size=1, show_progress_bar=False) # Model instance already knows its device
        query_vec = np.array(query_vec, dtype='float32')

        # 2. Normalize if needed
        if self.use_cosine: faiss.normalize_L2(query_vec)

        # 3. Set search params
        if hasattr(self.index, 'hnsw'): self.index.hnsw.efSearch = ef_search
        elif hasattr(self.index, 'nprobe'): self.index.nprobe = min(self.index.nlist, nprobe) if hasattr(self.index, 'nlist') else nprobe

        # 4. Search
        self.logger.debug(f"Searching index with top_k={top_k}")
        try: distances, indices = self.index.search(query_vec.reshape(1, -1), top_k) # Ensure query_vec is 2D
        except Exception as e: self.logger.error(f"FAISS search failed: {e}"); return []

        # 5. Format results
        results = []
        if len(indices) > 0:
            for i, idx in enumerate(indices[0]):
                if idx == -1: continue
                if 0 <= idx < len(self.texts): # Metadata bounds check implicitly covered if lengths match
                    score_label = 'score' if self.use_cosine else 'distance'
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {}, # Safety check for metadata
                        score_label: float(distances[0][i])
                    })
                else: self.logger.warning(f"Search returned out-of-bounds index {idx}")
        return results

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        results = self.search(query, top_k=top_k)
        return [r['text'] for r in results]

# # --- END OF FILE vector_store.py ---
# Start with: index_type='ivf', metric='cosine'. This offers a robust balance for typical textbook sizes.
# Monitor Performance: If search is too slow (unlikely with IVF unless N is huge), consider hnsw. If accuracy seems slightly off (rarely a major issue with reasonable nprobe), ensure nprobe is adequate (e.g., 10-20). If the dataset is small, you can simplify to flat.
# Always use: metric='cosine' for sentence embeddings.