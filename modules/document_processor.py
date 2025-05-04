import json
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any
import os # Needed for path operations if used within methods

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles loading source data like PDF text and configuration JSON files.
    It does NOT handle chunking, embedding, or indexing; that's VectorStore's job.
    Methods are static as they don't depend on instance state related to processing.
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
        text = ""
        logger.info(f"Attempting to extract text from: {pdf_path}")
        try:
            # Check if file exists before trying to open
            if not os.path.isfile(pdf_path):
                logger.error(f"PDF file not found at path: {pdf_path}")
                return ""

            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    try:
                        text += page.get_text() + "\n" # Add newline between pages
                    except Exception as page_e:
                        logger.warning(f"Error extracting text from page {page_num} of {pdf_path}: {page_e}")
                logger.info(f"Successfully extracted text from {pdf_path} ({len(doc)} pages).")
        except Exception as e:
            logger.error(f"Failed to open or process PDF {pdf_path}: {e}")
            return "" # Return empty string on failure
        return text.strip() # Remove leading/trailing whitespace from the combined text

    @staticmethod
    def extract_texts(pdf_paths: List[str]) -> List[str]:
        """
        Extracts text from a list of PDF paths.

        Args:
            pdf_paths (List[str]): A list of paths to PDF files.

        Returns:
            List[str]: A list containing the extracted text for each PDF.
                       Order matches the input list. Contains empty strings for failed extractions.
        """
        return [DocumentProcessor.extract_text(p) for p in pdf_paths]

    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """
        Loads data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            Dict[str, Any]: The loaded JSON data as a dictionary, or an empty dictionary on error.
        """
        logger.info(f"Attempting to load JSON data from: {file_path}")
        try:
             # Check if file exists
            if not os.path.isfile(file_path):
                logger.error(f"JSON file not found at path: {file_path}")
                return {}

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON from {file_path}.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to read or parse JSON file {file_path}: {e}")
            return {}

    @staticmethod
    def load_questions(questions_path: str = "questions.json") -> List[dict]:
        """
        Loads questions from the specified JSON file.
        Expects a structure like {"questions": [...]}.

        Args:
            questions_path (str): Path to the questions JSON file.

        Returns:
            List[dict]: The list of questions, or an empty list if loading fails
                        or the 'questions' key is missing/not a list.
        """
        data = DocumentProcessor.load_json_file(questions_path)
        questions = data.get('questions', [])

        if not isinstance(questions, list):
            logger.warning(f"'questions' key in {questions_path} is not a list. Returning empty list.")
            return []

        # Optional: Add validation for individual question structure if needed
        # for i, q in enumerate(questions):
        #     if not isinstance(q, dict) or 'Question' not in q:
        #         logger.warning(f"Invalid question format at index {i} in {questions_path}")
        #         # Decide how to handle: filter out, return empty, raise error?
        #         # For robustness, perhaps just log and keep valid ones, or filter later.

        return questions

    @staticmethod
    def load_syllabus(syllabus_path: str = "syllabus.json") -> dict:
        """
        Loads the syllabus JSON (e.g., course outcomes).

        Args:
            syllabus_path (str): Path to the syllabus JSON file.

        Returns:
            dict: The loaded syllabus data, or an empty dictionary on error.
        """
        # Assuming the entire file content is the syllabus structure
        return DocumentProcessor.load_json_file(syllabus_path)