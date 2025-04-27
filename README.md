A system that works reliably with minimal technical complexity, while still achieving all functional requirements with modular approach .


✅1. document_processor.py
•	Purpose: Handles syllabus, textbook, and question paper ingestion.
•	Core Features:
o	PyMuPDF for PDF parsing.
o	FAISS for textbook vector storage.
o	Syllabus parsing from PDF/JSON.
o	Chunking with LangChain RecursiveCharacterTextSplitter.
o	Search via embedding similarity with optional book filtering.
o	CO analysis and question-topic matching.
________________________________________
✅ 2. answer_generator.py
•	Purpose: Generates and stores model answers using quadrant vector search.
•	Core Features:
o	Quadrant-based semantic retrieval from textbook chunks.
o	Structured prompt building based on question metadata + rubric + COs.
o	Compatible with pluggable llm_generate_func() to support Mixtral, GEMINI_API, etc.
o	Stores results in model_answers.db.
________________________________________
✅ 3. student_processor.py
•	Purpose: Parses student handwritten answer PDFs.
•	Core Features:
o	Uses PyMuPDF to extract text from scanned PDFs (no OCR).
o	Regex-based answer segmentation using visible question markers (e.g., Q1, 1., 1a).
o	Stores in student_answers.db.
o	Registers new students and tracks question-to-answer mappings.
________________________________________
✅ 4. evaluator.py
•	Purpose: Scores student answers against model answers.
•	Core Features:
o	Semantic similarity scoring (SentenceTransformer).
o	Rubric point matching via segment-wise similarity.
o	Plagiarism detection from textbooks and peer answers.
o	Custom scoring logic: weighted similarity + rubric + plagiarism penalties.
o	Writes results to evaluations.db.
________________________________________
✅ 5. feedback_generator.py
•	Purpose: Generates detailed feedback for instructors and students.
•	Core Features:
o	Aggregates student evaluation results.
o	Computes class-wide statistics.
o	Visualizes performance (matplotlib charts → base64 for UI/HTML).
o	Generates per-student feedback with CO-level analysis.
________________________________________
Everything is modular, local, resource-light, and matches your minimal-complexity design goals. You're in great shape to connect this into a LangChain pipeline + Streamlit UI for full integration.




	

