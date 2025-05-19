import os
import json
import sqlite3
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
import time # To prevent rapid re-runs if needed

# Import our refactored modules
from modules.document_processor import DocumentProcessor # Simplified version
from modules.vector_store import VectorStore # Advanced version
from modules.answer_generator import AnswerGenerator
from modules.evaluator import Evaluator
from modules.feedback_generator import FeedbackGenerator
from modules.student_processor import StudentProcessor

# --- Configuration ---
st.set_page_config(page_title="Assessment Evaluation System", page_icon="📝", layout="wide")
# Directory Paths
BASE_DIR = Path(__file__).resolve().parent  # Project root
DATA_DIR = BASE_DIR / "data"
TEXTBOOK_DIR = DATA_DIR / "textbooks"
STUDENT_DIR = DATA_DIR / "student_answers"
STUDENT_TEXT = DATA_DIR / "student_answers_txt"
VSTORE_CACHE = DATA_DIR / "vector_store_cache"
EXT_DIR = DATA_DIR / "Ext"  # For external outputs like processed student answers

# File Paths
SYLLABUS_PATH = DATA_DIR / "syllabus.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
MODEL_ANSWERS_PATH = DATA_DIR / "model_answers.json"
STUDENT_ANSWERS_PATH = EXT_DIR / "student_answers.json"
DB_PATH = DATA_DIR / "evaluation_results.db"
FILE_STATE_CACHE = DATA_DIR/VSTORE_CACHE / "file_index.json"

# Ensure directories exist
for path in [TEXTBOOK_DIR, STUDENT_DIR, STUDENT_TEXT, VSTORE_CACHE, EXT_DIR]:
    os.makedirs(path, exist_ok=True)

# --- Session State Initialization ---
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
    st.session_state['vector_store_config'] = {"index_type": "flat", "metric": "l2"} # Default config

# --- Helper Functions ---

def get_current_textbook_state(directory: Path) -> dict:
    """Gets a dictionary of PDF paths and their modification times."""
    current_files = {}
    if not directory.is_dir():
        st.warning(f"Textbook directory not found: {directory}")
        return {}
    for pdf_path in directory.glob("*.pdf"):
        try:
            current_files[str(pdf_path.resolve())] = pdf_path.stat().st_mtime
        except OSError as e:
            st.error(f"Could not access file {pdf_path}: {e}")
            current_files[str(pdf_path.resolve())] = None # Mark as inaccessible
    return current_files

def load_saved_file_state(cache_path: Path) -> dict:
    """Loads the previously saved file state from the JSON cache."""
    if not cache_path.is_file():
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            saved_state = json.load(f)
            # Ensure the 'files' key exists and is a dict
            if isinstance(saved_state.get("files"), dict):
                 return saved_state.get("files", {})
            else:
                 st.warning(f"Invalid format in {cache_path}. Ignoring saved state.")
                 return {}
    except (json.JSONDecodeError, OSError) as e:
        st.error(f"Could not read or parse state file {cache_path}: {e}")
        return {}

def save_file_state(state_data: dict, cache_path: Path):
    """Saves the current file state to the JSON cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"files": state_data}, f, indent=2)
        st.info(f"Saved file state to {cache_path}")
    except OSError as e:
        st.error(f"Failed to save file state cache {cache_path}: {e}")

def get_evaluation_stats():
    """Retrieves summary statistics from the evaluation database."""
    if not DB_PATH.is_file():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        # Use try-except for SQL queries in case table doesn't exist yet
        try:
            overall = pd.read_sql(
                "SELECT AVG(similarity) as avg_similarity, AVG(coverage) as avg_coverage, SUM(CASE WHEN plagiarism_flag = 1 THEN 1 ELSE 0 END) as plagiarism_flags FROM evaluations",
                conn
            ).iloc[0].to_dict()
        except pd.io.sql.DatabaseError:
             st.warning("Overall evaluations table might be empty or missing.")
             overall = {"avg_similarity": 0, "avg_coverage": 0, "plagiarism_flags": 0}
        except IndexError: # Handles case where query returns no rows
             overall = {"avg_similarity": 0, "avg_coverage": 0, "plagiarism_flags": 0}

        try:
             questions = pd.read_sql(
                 "SELECT question_id, AVG(similarity) as avg_similarity, AVG(coverage) as avg_coverage, COUNT(*) as num_responses FROM evaluations GROUP BY question_id",
                 conn
            )
        except pd.io.sql.DatabaseError:
             st.warning("Per-question evaluations table might be empty or missing.")
             questions = pd.DataFrame(columns=["question_id", "avg_similarity", "avg_coverage", "num_responses"])

        try:
             students = pd.read_sql(
                "SELECT student_id, AVG(similarity) as avg_similarity, AVG(coverage) as avg_coverage, SUM(CASE WHEN plagiarism_flag = 1 THEN 1 ELSE 0 END) as plagiarism_flags FROM evaluations GROUP BY student_id",
                 conn
            )
        except pd.io.sql.DatabaseError:
             st.warning("Per-student evaluations table might be empty or missing.")
             students = pd.DataFrame(columns=["student_id", "avg_similarity", "avg_coverage", "plagiarism_flags"])

        conn.close()
        return {"overall": overall, "questions": questions, "students": students}
    except sqlite3.Error as e:
        st.error(f"Database error while fetching stats: {e}")
        return None

def display_sidebar():
    """Sets up and displays the sidebar navigation and status."""
    st.sidebar.title("📚 Assessment System")
    page = st.sidebar.radio("Navigation", [
        "Dashboard",
        "Upload Syllabus",
        "Upload Questions",
        "Manage Textbooks & Index", # Renamed
        "Upload Student Answers",
        "Generate & View Results"
    ])
    st.sidebar.markdown("---")

    # Status indicators based on file existence or session state
    st.sidebar.write(f"{'✅' if SYLLABUS_PATH.is_file() else '❌'} Syllabus")
    st.sidebar.write(f"{'✅' if QUESTIONS_PATH.is_file() else '❌'} Questions")
    st.sidebar.write(f"{'✅' if any(TEXTBOOK_DIR.glob('*.pdf')) else '❌'} Textbooks Present")
    st.sidebar.write(f"{'✅' if any(STUDENT_DIR.glob('*.pdf')) else '❌'} Student PDFs Present")
    st.sidebar.write(f"{'✅' if MODEL_ANSWERS_PATH.is_file() else '❌'} Model Answers")
    st.sidebar.write(f"{'✅' if DB_PATH.is_file() else '❌'} Evaluation DB")

    # Vector Store Status Check - check session state
    if st.session_state.get('vector_store') is not None:
        vs = st.session_state.vector_store
        # Display loaded index config
        st.sidebar.write(f"✅ Index Loaded ({vs.index_type}/{vs.metric}, {vs.index.ntotal if vs.index else 0} items)")
    else:
         # Check if *any* index seems cached (less precise)
        if any(VSTORE_CACHE.glob('*.index')):
             st.sidebar.write(f"🟡 Index Cached (Not Loaded)")
        else:
             st.sidebar.write(f"❌ Index Not Built/Cached")

    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 AI Assessment Tools")
    return page


# --- Page Functions ---

def show_dashboard():
    """Displays the main dashboard with evaluation statistics."""
    st.title("📊 Dashboard")
    stats = get_evaluation_stats()

    if stats and stats["overall"]:
        overall = stats["overall"]
        q_df = stats["questions"]
        s_df = stats["students"]

        st.header("Overall Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Similarity", f"{overall.get('avg_similarity', 0):.2%}", help="Average semantic similarity score across all answers.")
        col2.metric("Avg. Coverage", f"{overall.get('avg_coverage', 0):.2%}", help="Average keyword coverage score across all answers.")
        col3.metric("Plagiarism Flags", f"{overall.get('plagiarism_flags', 0)}", help="Total count of answers flagged for high similarity.")

        st.header("Performance by Question")
        if not q_df.empty:
            st.dataframe(q_df.style.format({'avg_similarity': '{:.2%}', 'avg_coverage': '{:.2%}'}))
        else:
            st.info("No per-question data available yet.")

        st.header("Performance by Student")
        if not s_df.empty:
             # Add rank
            s_df['Similarity Rank'] = s_df['avg_similarity'].rank(method='dense', ascending=False).astype(int)
            st.dataframe(s_df.sort_values('Similarity Rank').style.format({'avg_similarity': '{:.2%}', 'avg_coverage': '{:.2%}'}))
        else:
             st.info("No per-student data available yet.")

    else:
        st.info("No evaluation data found. Process answers and generate results first.")


def upload_syllabus():
    """Handles uploading of the syllabus JSON file."""
    st.title("📄 Upload Syllabus JSON")
    st.markdown(f"Upload your course syllabus file (`{SYLLABUS_PATH.name}`). It should be a JSON containing course details, including 'Course Outcomes' if applicable.")
    uploaded = st.file_uploader("Select Syllabus JSON", type="json", key="syllabus_uploader")
    if uploaded:
        try:
            # Directly write the uploaded content
            with open(SYLLABUS_PATH, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"{SYLLABUS_PATH.name} uploaded successfully.")
            # Optionally try loading it to validate structure here
            syllabus_data = DocumentProcessor.load_syllabus(str(SYLLABUS_PATH))
            if not syllabus_data:
                 st.warning("Syllabus file uploaded, but failed basic load check. Ensure it's valid JSON.")
            elif "Course Outcomes" not in syllabus_data:
                 st.warning("Syllabus JSON uploaded, but missing 'Course Outcomes' key, which might be needed later.")
        except Exception as e:
            st.error(f"Failed to save or process uploaded syllabus: {e}")

def upload_questions():
    """Handles uploading of the questions JSON file."""
    st.title("❓ Upload Questions JSON")
    st.markdown(f"Upload your questions file (`{QUESTIONS_PATH.name}`). Expected format: a JSON object with a top-level key `\"questions\"` containing a list of question objects.")
    uploaded = st.file_uploader("Select Questions JSON", type="json", key="questions_uploader")
    if uploaded:
        try:
            # Save first
            with open(QUESTIONS_PATH, "wb") as f:
                 f.write(uploaded.getvalue())
            st.success(f"{QUESTIONS_PATH.name} uploaded successfully.")
             # Then validate structure
            questions_list = DocumentProcessor.load_questions(str(QUESTIONS_PATH))
            if not questions_list and QUESTIONS_PATH.is_file(): # File exists but loaded empty list
                 # Try reading raw to see if 'questions' key is missing
                try:
                    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f_raw:
                        raw_data = json.load(f_raw)
                        if 'questions' not in raw_data:
                             st.error("Validation Error: Uploaded JSON is valid, but missing the required top-level 'questions' key.")
                        elif not isinstance(raw_data.get('questions'), list):
                            st.error("Validation Error: 'questions' key found, but its value is not a list.")
                        else:
                             st.warning("Questions file uploaded, but DocumentProcessor reported issues loading questions (check logs or file content).")
                except Exception:
                     st.error("Validation Error: Could not read the uploaded file as valid JSON.")

        except Exception as e:
            st.error(f"Failed to save or process uploaded questions file: {e}")

def manage_textbooks_and_index():
    """Handles textbook uploads and building/loading the VectorStore index."""
    st.title("📚 Manage Textbooks & FAISS Index")

    st.subheader("Uploaded Textbooks")
    books = sorted(list(TEXTBOOK_DIR.glob("*.pdf")))
    if books:
        for b in books:
            st.write(f"- {b.name}")
    else:
        st.info("No PDF textbooks found in the 'textbooks' directory.")

    uploaded_files = st.file_uploader("Upload PDF textbooks", accept_multiple_files=True, type="pdf", key="textbook_uploader")
    uploaded_filenames = set() # Keep track to avoid double processing
    new_upload_detected = False
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in uploaded_filenames:
                 path = TEXTBOOK_DIR / file.name
                 try:
                     with open(path, "wb") as f:
                        f.write(file.getbuffer())
                     st.success(f"Uploaded {file.name}")
                     uploaded_filenames.add(file.name)
                     new_upload_detected = True
                 except Exception as e:
                     st.error(f"Failed to save {file.name}: {e}")
        if new_upload_detected:
            # Force session state clear to reflect new files for rebuild check? No, get_current works live.
            st.rerun() # Rerun to update the listed files and rebuild check logic

    st.markdown("---")
    st.subheader("FAISS Index Management")

    # --- Configuration Selection ---
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        # Use session state to remember the user's choice
        st.session_state.vector_store_config["index_type"] = st.selectbox(
            "Index Type",
            options=["flat", "hnsw", "ivf"],
            index=["flat", "hnsw", "ivf"].index(st.session_state.vector_store_config["index_type"]),
            help="Flat: Exact search (slow for large N). HNSW/IVF: Approximate search (faster). Choose based on corpus size and accuracy needs."
        )
    with col_cfg2:
        st.session_state.vector_store_config["metric"] = st.selectbox(
            "Distance Metric",
            options=["l2", "cosine"],
            index=["l2", "cosine"].index(st.session_state.vector_store_config["metric"]),
            help="L2 (Euclidean) or Cosine Similarity (Inner Product on normalized vectors). Affects search results relevance."
        )

    selected_index_type = st.session_state.vector_store_config["index_type"]
    selected_metric = st.session_state.vector_store_config["metric"]

    # --- Check if Index Needs Rebuild ---
    # Get current textbook state
    current_textbook_state = get_current_textbook_state(TEXTBOOK_DIR)
    saved_textbook_state = load_saved_file_state(FILE_STATE_CACHE)

    # Check if cache files for the *selected configuration* exist
    vs_check = VectorStore(cache_dir=str(VSTORE_CACHE), index_type=selected_index_type, metric=selected_metric)
    config_cache_exists = (
        Path(vs_check.index_path).is_file() and
        Path(vs_check.texts_path).is_file() and
        Path(vs_check.meta_path).is_file()
    )

    # Determine need for rebuild
    files_changed = (current_textbook_state != saved_textbook_state)
    need_rebuild = files_changed or not config_cache_exists or not current_textbook_state # Rebuild if no books

    # Display status
    if files_changed and config_cache_exists:
        st.warning("Textbook files have changed since the last index build for this configuration. Rebuilding is recommended.")
    elif not config_cache_exists and current_textbook_state:
         st.warning(f"Index cache for '{selected_index_type}/{selected_metric}' not found. Building is required.")
    elif not current_textbook_state:
         st.info("No textbooks found. Upload textbooks to build an index.")


    # --- Load/Build Buttons ---
    col_btn1, col_btn2 = st.columns(2)

    # Disable buttons if no textbooks present
    disable_buttons = not current_textbook_state

    # Only offer Load if cache exists *and* files haven't changed *and* not already loaded
    can_load = config_cache_exists and not files_changed and st.session_state.vector_store is None and not disable_buttons
    with col_btn1:
         if st.button("Load Existing Index", disabled=disable_buttons or not config_cache_exists or files_changed or st.session_state.vector_store is not None, help="Load the cached index if files haven't changed and config cache exists."):
             if not config_cache_exists:
                 st.error(f"Cannot load: Cache files for configuration '{selected_index_type}/{selected_metric}' not found.")
             elif files_changed:
                  st.error("Cannot load: Textbook files have changed. Please rebuild.")
             else:
                with st.spinner(f"Loading index ({selected_index_type}/{selected_metric})..."):
                    try:
                        vs = VectorStore(cache_dir=str(VSTORE_CACHE), index_type=selected_index_type, metric=selected_metric)
                        if vs.load():
                            st.session_state.vector_store = vs
                            # Also update session state config to match loaded one
                            st.session_state.vector_store_config["index_type"] = vs.index_type
                            st.session_state.vector_store_config["metric"] = vs.metric
                            st.success(f"FAISS index ({vs.index_type}/{vs.metric}, {vs.index.ntotal} items) loaded successfully from cache.")
                            time.sleep(1) # Short pause before potential auto-rerun
                            st.rerun()
                        else:
                            st.error("Failed to load index from cache. Files might be corrupted. Try rebuilding.")
                            st.session_state.vector_store = None # Ensure it's clear
                            need_rebuild = True # Force rebuild option
                    except Exception as e:
                        st.error(f"An unexpected error occurred during loading: {e}")
                        st.session_state.vector_store = None
                        need_rebuild = True

    # Offer Build if needed or forced by user
    can_build = current_textbook_state # Need books to build
    with col_btn2:
         build_button_label = "Rebuild Index" if config_cache_exists else "Build Index"
         build_help = "Extract text, create embeddings, and build/save the FAISS index. Overwrites existing cache for this configuration."
         if st.button(build_button_label, disabled=disable_buttons, help=build_help):
            with st.spinner(f"Building index ({selected_index_type}/{selected_metric}). This may take a while..."):
                 try:
                     # 1. Get list of book paths again
                     books_to_process = list(TEXTBOOK_DIR.glob("*.pdf"))
                     book_paths_str = [str(p.resolve()) for p in books_to_process]
                     doc_ids = [p.name for p in books_to_process] # Use filenames as IDs

                     # 2. Extract text using DocumentProcessor
                     st.write(f"Extracting text from {len(book_paths_str)} PDF(s)...")
                     # DocumentProcessor methods are static
                     texts = DocumentProcessor.extract_texts(book_paths_str)

                     # Check if extraction yielded any text
                     if not any(texts):
                          st.error("Text extraction failed for all documents or yielded no text. Index cannot be built.")
                     else:
                        # 3. Initialize VectorStore with selected config
                        vs = VectorStore(cache_dir=str(VSTORE_CACHE), index_type=selected_index_type, metric=selected_metric)

                        # 4. Build the index
                        st.write("Building FAISS index (computing embeddings)...")
                        vs.build_index(texts, doc_ids, use_multiprocess=True) # Allow multi-process

                        # 5. Save the index and cache data
                        st.write("Saving index and cache data...")
                        if vs.save():
                            # 6. Save the current file state
                            save_file_state(current_textbook_state, FILE_STATE_CACHE)
                            st.session_state.vector_store = vs
                            # Ensure session state config matches built one
                            st.session_state.vector_store_config["index_type"] = vs.index_type
                            st.session_state.vector_store_config["metric"] = vs.metric
                            st.success(f"FAISS index ({vs.index_type}/{vs.metric}, {vs.index.ntotal} items) built and saved successfully.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Index built, but failed to save cache files. Check logs and permissions.")
                            st.session_state.vector_store = None # Don't keep partially saved state

                 except Exception as e:
                    st.error(f"An error occurred during index building: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}") # More detail for debugging
                    st.session_state.vector_store = None

    # Display loaded status if applicable
    if st.session_state.vector_store is not None:
        vs = st.session_state.vector_store
        st.success(f"Index currently loaded: Type='{vs.index_type}', Metric='{vs.metric}', Items={vs.index.ntotal if vs.index else 'N/A'}")

def upload_student_answers():
    """Handles uploading and processing of student answer PDFs or plain-text files."""
    st.title("🧑‍🎓 Upload Student Answers")

    mode = st.radio("Select input mode", ("PDF/OCR", "Plain Text"), index=0)

    if mode == "PDF/OCR":
        st.subheader("Uploaded Student PDFs")
        pdfs = sorted(STUDENT_DIR.glob("*.pdf"))
        if pdfs:
            for pdf in pdfs:
                st.write(f"- {pdf.name}")
        else:
            st.info(f"No PDFs in `{STUDENT_DIR}`")

        up = st.file_uploader("Upload student answer PDFs", accept_multiple_files=True, type="pdf", key="stu_pdf")
        if up:
            for f in up:
                (STUDENT_DIR / f.name).write_bytes(f.getbuffer())
            st.success("PDFs uploaded.")
            st.rerun()

        st.markdown("---")
        if pdfs and st.button("Process PDFs via OCR"):
            with st.spinner("Running OCR on PDFs..."):
                try:
                    sp = StudentProcessor()
                    sp.process_directory(
                        folder=str(STUDENT_DIR),
                        output_path=str(STUDENT_ANSWERS_PATH)
                    )
                    st.success(f"Processed PDFs → `{STUDENT_ANSWERS_PATH.name}`")
                except Exception as e:
                    st.error(f"OCR processing failed: {e}")

    else:  # Plain Text mode
        st.subheader("Uploaded Student .txt Files")
        txts = sorted(STUDENT_TEXT.glob("*.txt"))
        if txts:
            for txt in txts:
                st.write(f"- {txt.name}")
        else:
            st.info(f"No .txt files in `{STUDENT_TEXT}`")

        up = st.file_uploader("Upload student answer .txt files", accept_multiple_files=True, type="txt", key="stu_txt")
        if up:
            for f in up:
                (STUDENT_TEXT / f.name).write_bytes(f.getbuffer())
            st.success(".txt files uploaded.")
            st.rerun()

        st.markdown("---")
        if txts and st.button("Process Plain-Text Files"):
            with st.spinner("Parsing .txt answers..."):
                try:
                    sp = StudentProcessor(delimiter=r"(\d+\.\d+[a-z]?)")
                    sp.process_directory(
                        folder=str(STUDENT_TEXT),
                        input_type='text',
                        output_path=str(STUDENT_ANSWERS_PATH)
                    )
                    st.success(f"Processed text → `{STUDENT_ANSWERS_PATH.name}`")
                except Exception as e:
                    st.error(f"Text processing failed: {e}")

def generate_view_results():
    """Handles generation of model answers, evaluation, feedback, and viewing results."""
    st.title("🚀 Generate & View Results")

    # --- Prerequisites Check ---
    missing_prereqs = []
    if not QUESTIONS_PATH.is_file(): missing_prereqs.append(f"`{QUESTIONS_PATH.name}`")
    if st.session_state.get('vector_store') is None: missing_prereqs.append("Loaded FAISS Index")
    if not STUDENT_ANSWERS_PATH.is_file(): missing_prereqs.append(f"`{STUDENT_ANSWERS_PATH.name}`")
    if not MODEL_ANSWERS_PATH.is_file() and not missing_prereqs: # Model answers needed for eval/feedback but can be generated
        # Check later if needed for evaluate/feedback
        pass

    if missing_prereqs:
        st.warning(f"Missing prerequisites for some actions: {', '.join(missing_prereqs)}. Please complete previous steps.")


    # --- Action Buttons ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Model Answers")
        if st.button("Generate Model Answers", help="Uses the loaded index and questions to generate model answers with an LLM.", disabled=(st.session_state.get('vector_store') is None or not QUESTIONS_PATH.is_file())):
            if st.session_state['vector_store'] is None:
                st.error("FAISS index not loaded. Please load or build it first.")
            elif not QUESTIONS_PATH.is_file():
                st.error(f"Questions file ({QUESTIONS_PATH.name}) not found.")
            else:
                with st.spinner("Generating model answers (may take time depending on model and number of questions)..."):
                    try:
                        # Assume syllabus provides context needed by AnswerGenerator
                        ag = AnswerGenerator(
                            vector_store=st.session_state['vector_store'],
                            syllabus_path=str(SYLLABUS_PATH),
                            fallback_model_id="google/flan-t5-base",  # Optional: customize fallback
                            device="cuda" if torch.cuda.is_available() else "cpu",  # Device control
                            context_k=3  # How many context chunks to use
                        )

                        output_path = ag.generate_all(questions_file=str(QUESTIONS_PATH), output_file=str(MODEL_ANSWERS_PATH))
                        if output_path and Path(output_path).is_file():
                            st.success(f"Model answers saved to `{MODEL_ANSWERS_PATH.name}`.")
                        else:
                            st.error("Model answer generation finished, but output file was not created.")
                    except Exception as e:
                        st.error(f"Error during model answer generation: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")

    with col2:
        st.subheader("2. Evaluate Answers")
        model_answers_exist = MODEL_ANSWERS_PATH.is_file()
        student_answers_exist = STUDENT_ANSWERS_PATH.is_file()
        if st.button("Evaluate Student Answers", help="Compares student answers to model answers (similarity, coverage). Stores results in the database.", disabled=(not model_answers_exist or not student_answers_exist)):
            if not model_answers_exist:
                st.error(f"Model answers file (`{MODEL_ANSWERS_PATH.name}`) not found. Generate them first.")
            if not student_answers_exist:
                st.error(f"Processed student answers file (`{STUDENT_ANSWERS_PATH.name}`) not found. Process student answers first.")

            if model_answers_exist and student_answers_exist:
                with st.spinner("Evaluating student answers against model answers..."):
                    try:
                        # Load model answers JSON
                        with open(MODEL_ANSWERS_PATH, 'r') as f:
                            model_answers = json.load(f)["answers"]
                        # Load student answers JSON
                        with open(STUDENT_ANSWERS_PATH, 'r') as f:
                            student_data = json.load(f)
                        student_answers = {
                            s['student_id']: list(s['answer'].values())
                            for s in student_data.get("students", [])
                        }

                        # Evaluate student answers
                        ev = Evaluator(db_path=str(DB_PATH))
                        ev.evaluate(student_answers, model_answers)
                        ev.close()

                        st.success(f"Evaluation complete. Results stored in `{DB_PATH.name}`.")
                    except Exception as e:
                        st.error(f"An error occurred during evaluation: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")

    with col3:
        st.subheader("3. Generate Feedback")
        db_exists = DB_PATH.is_file()
        if st.button("Generate Feedback Files", help="Creates individual feedback files for students and a summary report.", disabled=not db_exists):
            if not db_exists:
                st.error(f"Evaluation database (`{DB_PATH.name}`) not found. Run evaluation first.")
            else:
                with st.spinner("Generating feedback reports..."):
                    try:
                        feedback_dir = BASE_DIR / "data2/feedback_reports"
                        os.makedirs(feedback_dir, exist_ok=True)

                        fg = FeedbackGenerator(db_path=str(DB_PATH))
                        fg.generate_feedback(output_dir=str(feedback_dir))
                        st.success(f"Feedback reports generated in `{feedback_dir}`.")
                    except Exception as e:
                        st.error(f"Error generating feedback: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")

    # Optional: display summary stats after evaluation
    st.markdown("---")
    st.subheader("📊 Evaluation Summary")
    stats = get_evaluation_stats()
    if stats and stats["overall"] and stats["overall"].get('avg_similarity') is not None:
        st.metric("Avg. Similarity", f"{stats['overall']['avg_similarity']:.2%}")
        st.metric("Avg. Coverage", f"{stats['overall']['avg_coverage']:.2%}")
        st.metric("Plagiarism Flags", f"{stats['overall']['plagiarism_flags']}")
        if st.button("Go to Dashboard"):
            st.info("Use the sidebar to navigate to the Dashboard for more detailed analysis.")
    else:
        st.info("Run evaluation to see summary statistics.")
        
if __name__ == "__main__":
    page = display_sidebar()

    if page == "Dashboard":
        show_dashboard()
    elif page == "Upload Syllabus":
        upload_syllabus()
    elif page == "Upload Questions":
        upload_questions()
    elif page == "Manage Textbooks & Index":
        manage_textbooks_and_index()
    elif page == "Upload Student Answers":
        upload_student_answers()
    elif page == "Generate & View Results":
        generate_view_results()
