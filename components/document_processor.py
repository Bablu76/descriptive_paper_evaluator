import os, json
import numpy as np
import fitz  # PyMuPDF

# Try smart splitter (LangChain)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    USE_SMART_SPLIT = True
except ImportError:
    USE_SMART_SPLIT = False

# Try sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("❌ Please install sentence-transformers to use DocumentProcessor.")

# Try FAISS
try:
    import faiss
except ImportError:
    raise ImportError("❌ Please install faiss-cpu or faiss-gpu to use semantic search.")

class DocumentProcessor:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.meta = []

        if USE_SMART_SPLIT:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )
        else:
            self.splitter = self._basic_splitter(chunk_size=800, overlap=100)

    def _basic_splitter(self, chunk_size=800, overlap=100):
        class BasicSplitter:
            def __init__(self): self.chunk_size, self.overlap = chunk_size, overlap
            def split_text(self, text):
                paras = text.split("\n\n")
                chunks = []
                for p in paras:
                    p = p.strip()
                    if not p: continue
                    while len(p) > chunk_size:
                        chunks.append(p[:chunk_size])
                        p = p[chunk_size - overlap:]
                    chunks.append(p)
                return chunks
        return BasicSplitter()

    def parse_syllabus_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "course_info": data.get("Course Information", {}),
            "objectives":  data.get("Course Objectives", []),
            "outcomes":    data.get("Course Outcomes", []),
            "units":       data.get("Unit Entries", [])
        }

    def parse_question_paper_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qs = []
        for q in data:
            qs.append({
                "question_number": q.get("Q_No"),
                "question_text":   q.get("Question"),
                "marks":           int(q.get("Marks", 0)),
                "co":              q.get("CO"),
                "rubric":          q.get("Rubric", [])
            })
        return qs

    def extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        doc = fitz.open(pdf_path)
        pages = []
        for i in range(len(doc)):
            txt = doc[i].get_text().strip()
            if txt:
                pages.append({
                    "source": os.path.basename(pdf_path),
                    "page":   i + 1,
                    "text":   txt
                })
        doc.close()
        return pages

    def chunk_and_embed_texts(self, pages):
        self.chunks = []
        self.meta = []

        for page in pages:
            for chunk in self.splitter.split_text(page["text"]):
                chunk = chunk.strip()
                if chunk:
                    self.chunks.append(chunk)
                    self.meta.append({
                        "source": page["source"],
                        "page": page["page"]
                    })

        if not self.chunks:
            raise RuntimeError("❌ No valid chunks generated from input.")

        embs = self.embedder.encode(self.chunks, convert_to_numpy=True).astype("float32")
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        return self.index

    def search(self, query, k=5):
        if not self.chunks or self.index is None:
            raise RuntimeError("🔍 No index available. Run chunk_and_embed_texts() first.")

        # Try semantic search
        try:
            q_emb = self.embedder.encode([query]).astype("float32")
            _, idxs = self.index.search(q_emb, min(k, self.index.ntotal))
            hits = []
            for idx in idxs[0]:
                if 0 <= idx < len(self.chunks):
                    hits.append({
                        "content": self.chunks[idx],
                        "meta": self.meta[idx]
                    })
            if hits:
                return hits
        except Exception as e:
            print("⚠️ FAISS search failed:", e)

        # Substring fallback
        low = query.lower()
        subs = []
        for txt, m in zip(self.chunks, self.meta):
            if low in txt.lower():
                subs.append({"content": txt, "meta": m})
                if len(subs) >= k:
                    break
        if subs:
            return subs

        # Final fallback
        return [
            {"content": txt, "meta": m}
            for txt, m in zip(self.chunks[:k], self.meta[:k])
        ]
