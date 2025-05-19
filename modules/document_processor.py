import json
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles loading source data like PDF text and configuration JSON files,
    and normalizing question data. Does NOT handle embeddings or indexing.
    """

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """
        Extracts and returns all text from a single PDF file using PyMuPDF.

        Args:
            pdf_path (str): The full path to the PDF file.

        Returns:
            str: The extracted text, or an empty string if an error occurs.
        """
        if not isinstance(pdf_path, str) or not pdf_path:
            logger.error(f"Invalid pdf_path provided: {pdf_path!r}")
            return ""

        if not os.path.isfile(pdf_path):
            logger.error(f"PDF file not found at path: {pdf_path}")
            return ""

        text_chunks: List[str] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text().strip()
                        if page_text:
                            text_chunks.append(page_text)
                        else:
                            logger.debug(f"No text on page {page_num} of {pdf_path}, skipping.")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
                logger.info(f"Extracted text from {pdf_path} ({len(doc)} pages).")
        except Exception as e:
            logger.error(f"Failed to open or process PDF {pdf_path}: {e}")
            return ""

        return "\n".join(text_chunks)

    @staticmethod
    def extract_texts(pdf_paths: List[str]) -> List[str]:
        """
        Extracts text from a list of PDF paths.

        Args:
            pdf_paths (List[str]): Paths to PDF files.

        Returns:
            List[str]: Extracted text strings; empty string for failures.
        """
        if not isinstance(pdf_paths, list):
            logger.error("extract_texts expects a list of paths.")
            return []
        return [DocumentProcessor.extract_text(p) for p in pdf_paths]

    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """
        Loads data from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Dict[str, Any]: Loaded data, or empty dict on error.
        """
        if not isinstance(file_path, str) or not file_path:
            logger.error(f"Invalid file_path provided: {file_path!r}")
            return {}

        if not os.path.isfile(file_path):
            logger.error(f"JSON file not found at path: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON data from {file_path}.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
        return {}

    @staticmethod
    def load_questions(questions_path: str = "questions.json") -> List[Dict[str, Any]]:
        """
        Loads and normalizes questions from JSON.
        Supports both old key 'Rubric' (list of str) and new 'Rubrics' (list of dicts).

        Args:
            questions_path (str): Path to questions JSON file.

        Returns:
            List[Dict[str, Any]]: Normalized question dicts with keys:
                - Q_No: optional identifier
                - Question: question text
                - Marks: integer marks
                - CO: course outcome id (if present)
                - Rubrics: List[dict] each with 'Criteria' (str) and 'Marks' (int)
        """
        data = DocumentProcessor.load_json_file(questions_path)
        raw = data.get("questions")
        if not isinstance(raw, list):
            logger.warning(f"'questions' missing or not a list in {questions_path}.")
            return []

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict question at index {idx}.")
                continue

            # Preserve Q_No if present
            q_no = item.get("Q_No")

            # Check essential fields
            question_text = item.get("Question")
            marks = item.get("Marks")
            if not question_text or not isinstance(question_text, str):
                logger.warning(f"Skipping question at index {idx}: missing or invalid 'Question'.")
                continue
            if not isinstance(marks, int):
                logger.warning(f"Skipping question at index {idx}: missing or invalid 'Marks'.")
                continue

            # Normalize rubric key
            raw_rubrics = []
            if "Rubrics" in item and isinstance(item["Rubrics"], list):
                raw_rubrics = item["Rubrics"]
            elif "Rubric" in item and isinstance(item["Rubric"], list):
                raw_rubrics = item["Rubric"]
            else:
                logger.info(f"No rubrics found for question at index {idx}, defaulting to empty list.")

            clean_rubrics: List[Dict[str, Any]] = []
            for r_idx, r in enumerate(raw_rubrics):
                if isinstance(r, dict) and "Criteria" in r and "Marks" in r:
                    clean_rubrics.append({
                        "Criteria": str(r["Criteria"]),
                        "Marks": int(r["Marks"])
                    })
                elif isinstance(r, str):
                    # support old simple string rubric entries (no mark info)
                    clean_rubrics.append({"Criteria": r, "Marks": 0})
                else:
                    logger.warning(f"Skipping invalid rubric entry at question {idx}, rubric {r_idx}.")

            normalized.append({
                **({"Q_No": q_no} if q_no is not None else {}),
                "Question": question_text,
                "Marks": marks,
                **({"CO": item.get("CO")} if item.get("CO") else {}),
                "Rubrics": clean_rubrics
            })

        logger.info(f"Loaded and normalized {len(normalized)} questions from {questions_path}.")
        return normalized

    @staticmethod
    def load_syllabus(syllabus_path: str = "syllabus.json") -> Dict[str, Any]:
        """
        Loads the syllabus JSON file (e.g., course outcomes mapping).

        Args:
            syllabus_path (str): Path to the syllabus JSON file.

        Returns:
            Dict[str, Any]: Loaded syllabus data, or empty dict on error.
        """
        return DocumentProcessor.load_json_file(syllabus_path)
