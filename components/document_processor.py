import json
import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, embedding_model_name="sentence-transformers/all-mpnet-base-v2", metric="cosine"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.metric = metric
        self.questions = []
        self.syllabus = {}
        self.documents = []
        self.text_chunks = []
        self.chunk_sources = []
        self.index = None

    def load_questions(self, file_path):
        """
        Load questions from a JSON file.
        """
        with open(file_path, "r") as file:
            self.questions = json.load(file)

    def load_syllabus(self, file_path):
        """
        Load syllabus from a JSON file.
        """
        with open(file_path, "r") as file:
            self.syllabus = json.load(file)

    def extract_text_from_pdfs(self, pdf_files):
        """
        Extract text from PDF files using PyMuPDF.
        """
        for pdf_file in pdf_files:
            doc = fitz.open(pdf_file)
            pdf_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pdf_text += page.get_text("text")
            self.documents.append({"text": pdf_text, "source": pdf_file})

    def chunk_documents(self, chunk_size=500, overlap=50):
        """
        Chunk the extracted document text into smaller chunks.
        """
        for doc in self.documents:
            text = doc["text"]
            source = doc["source"]
            # Basic chunking with overlap
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                self.text_chunks.append(chunk)
                self.chunk_sources.append(source)

    def embed_chunks(self):
        """
        Embed the chunks of text into embeddings using the SentenceTransformer.
        """
        embeddings = self.embedding_model.encode(self.text_chunks, convert_to_numpy=True)
        return embeddings

    def build_faiss_index(self):
        """
        Build a FAISS index from the embeddings of text chunks.
        """
        embeddings = self.embed_chunks()
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def save_index(self, index_file, source_file):
        """
        Save the FAISS index and chunk sources to files.
        """
        faiss.write_index(self.index, index_file)
        with open(source_file, "wb") as f:
            pickle.dump(self.chunk_sources, f)

    def load_index(self, index_file, source_file):
        """
        Load a FAISS index and chunk sources from files.
        """
        self.index = faiss.read_index(index_file)
        with open(source_file, "rb") as f:
            self.chunk_sources = pickle.load(f)

    def analyze_co_distribution(self):
        """
        Analyze the distribution of questions across Course Outcomes (COs).
        Returns a dictionary with CO stats and question counts.
        """
        co_counts = {}
        co_marks = {}
        co_questions = {}

        co_definitions = {
            f"co{i+1}": outcome 
            for i, outcome in enumerate(self.syllabus.get("Course Outcomes", []))
        }

        for q in self.questions:
            co = q.get("CO", "").lower()
            if co:
                co_counts[co] = co_counts.get(co, 0) + 1
                co_marks[co] = co_marks.get(co, 0) + int(q.get("Marks", 0))
                if co not in co_questions:
                    co_questions[co] = []
                co_questions[co].append(q["Q_No"])

        total_questions = len(self.questions)
        total_marks = sum(int(q.get("Marks", 0)) for q in self.questions)

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

        return analysis

    def match_questions_to_topics(self):
        """
        Match questions to relevant topics in the syllabus based on content similarity.
        Uses the search functionality to find the most relevant topics for each question.
        """
        topics = []
        for unit in self.syllabus.get("Unit Entries", []):
            unit_name = unit.get("Unit", "")
            topic_name = unit.get("Topic", "")
            for subtopic in unit.get("Subtopics", []):
                topics.append({
                    "unit": unit_name,
                    "topic": topic_name,
                    "subtopic": subtopic,
                    "text": f"{unit_name} {topic_name} {subtopic}"
                })

        question_topic_matches = []
        for question in self.questions:
            q_text = question.get("Question", "")

            if not hasattr(self, 'topic_embeddings'):
                topic_texts = [t["text"] for t in topics]
                self.topic_embeddings = self.embedding_model.encode(
                    topic_texts, 
                    convert_to_numpy=True
                )

            q_emb = self.embedding_model.encode([q_text], convert_to_numpy=True)

            if self.metric == "cosine":
                faiss.normalize_L2(q_emb)
                faiss.normalize_L2(self.topic_embeddings)
                similarities = q_emb @ self.topic_embeddings.T
            else:
                distances = np.sum((q_emb - self.topic_embeddings)**2, axis=1)
                similarities = 1 / (1 + distances) 

            top_indices = np.argsort(similarities[0])[-3:][::-1]

            question_topic_matches.append({
                "question_no": question.get("Q_No", ""),
                "question_text": q_text,
                "co": question.get("CO", ""),
                "matches": [
                    {
                        "unit": topics[idx]["unit"],
                        "topic": topics[idx]["topic"],
                        "subtopic": topics[idx]["subtopic"],
                        "similarity": float(similarities[0][idx])
                    }
                    for idx in top_indices
                ]
            })

        return question_topic_matches
