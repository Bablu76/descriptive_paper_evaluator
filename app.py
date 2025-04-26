import os, json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from components.document_processor import DocumentProcessor
from components.answer_generator import AnswerGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

st.title("📝 Descriptive Paper Evaluator")

# --- File Uploads ---
syllabus = st.file_uploader("Syllabus (JSON)", type="json")
textbooks = st.file_uploader("Textbooks / Class Notes (PDF)", type="pdf", accept_multiple_files=True)
questions = st.file_uploader("Questions (JSON)", type="json")

if st.button("Initialize & Generate Answers"):
    if not (syllabus and textbooks and questions):
        st.error("Please upload all three: syllabus, PDFs, and questions.")
        st.stop()

    os.makedirs("temp", exist_ok=True)
    dp = DocumentProcessor()
    all_pages = []

    # parse syllabus (if you need to show it)
    s_path = os.path.join("temp", syllabus.name)
    with open(s_path,"wb") as f: f.write(syllabus.read())
    meta = dp.parse_syllabus_json(s_path)
    st.success(f"Loaded course: {meta['course_info'].get('Course Name','–')}")

    # load all PDFs
    for pdf in textbooks:
        p = os.path.join("temp", pdf.name)
        with open(p,"wb") as f: f.write(pdf.read())
        st.write(f"• Parsed PDF: {pdf.name}")
        all_pages += dp.extract_text_from_pdf(p)

    # index
    dp.chunk_and_embed_texts(all_pages)
    st.success(f"Indexed {len(dp.chunks)} text chunks.")

    # parse questions
    q_path = os.path.join("temp", questions.name)
    with open(q_path,"wb") as f: f.write(questions.read())
    qs = dp.parse_question_paper_json(q_path)
    st.success(f"Loaded {len(qs)} questions.")

    # generate answers
    ag = AnswerGenerator()
    answers = {}
    bar = st.progress(0)
    for i, q in enumerate(qs, start=1):
        ans = ag.generate_model_answer(
            question_id   = q["question_number"],
            question_text = q["question_text"],
            marks         = q["marks"],
            co_id         = q["co"],
            rubric        = q["rubric"],
            retriever     = dp,
            top_k         = 5
        )
        answers[q["question_number"]] = ans
        bar.progress(i/len(qs))

    st.success("✅ Generated all model answers!")

    # show & download
    for qid, ans in answers.items():
        with st.expander(qid):
            st.write(ans)

    st.download_button(
        "Download All Answers as JSON",
        data=json.dumps(answers, indent=2),
        file_name="model_answers.json",
        mime="application/json"
    )
