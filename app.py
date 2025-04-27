import streamlit as st
import json
from document_processor import DocumentProcessor
import os
import tempfile

# Set up Streamlit interface
st.title("Descriptive Paper Evaluator")

# Upload syllabus and question files
st.sidebar.header("Upload Files")
syllabus_file = st.sidebar.file_uploader("Upload Syllabus (JSON)", type="json")
questions_file = st.sidebar.file_uploader("Upload Questions (JSON)", type="json")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Initialize Document Processor
processor = DocumentProcessor()

# Process syllabus and questions when files are uploaded
if syllabus_file and questions_file:
    syllabus_data = json.load(syllabus_file)
    questions_data = json.load(questions_file)

    processor.syllabus = syllabus_data
    processor.questions = questions_data

    st.sidebar.success("Syllabus and Questions loaded successfully!")

# Extract text from PDFs and process them
if pdf_files:
    # Save uploaded PDFs temporarily
    pdf_paths = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_paths.append(temp_file.name)

    processor.extract_text_from_pdfs(pdf_paths)
    processor.chunk_documents()

    st.sidebar.success(f"Extracted text from {len(pdf_files)} PDFs.")

# Build FAISS index
if st.sidebar.button("Build FAISS Index"):
    processor.build_faiss_index()
    st.sidebar.success("FAISS Index built successfully!")

# Analyze CO distribution
if st.sidebar.button("Analyze CO Distribution"):
    analysis = processor.analyze_co_distribution()
    st.write("CO Distribution Analysis")
    st.json(analysis)

# Match Questions to Topics
if st.sidebar.button("Match Questions to Topics"):
    matches = processor.match_questions_to_topics()
    st.write("Question to Topic Matches")
    st.json(matches)

# Display documents (optional)
if pdf_files:
    st.write("Extracted PDF Text:")
    for doc in processor.documents:
        st.text(doc['text'])
