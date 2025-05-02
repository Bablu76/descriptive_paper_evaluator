import os
import json
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

class StudentProcessor:
    """
    Processes student answer PDFs, extracting text and using OCR if needed.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a student PDF. If a page has no text, uses OCR (Tesseract).
        """
        self.logger.info(f"Processing student PDF: {pdf_path}")
        text_content = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text().strip()
                if page_text:
                    text_content += page_text + "\n"
                else:
                    # Page likely contains image; apply OCR
                    self.logger.info(f"No text on page, using OCR: Page {page.number}")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    text_content += ocr_text + "\n"
            return text_content
        except Exception as e:
            self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def process_directory(self, folder_path: str):
        """
        Process all PDF files in a directory. Saves output to 'student_answers.json'.
        """
        results = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    student_id = os.path.splitext(file)[0]
                    pdf_path = os.path.join(root, file)
                    text = self.extract_text_from_pdf(pdf_path)
                    results.append({"student_id": student_id, "answer_text": text})
        
        output_path = "student_answers.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"students": results}, f, indent=2)
            self.logger.info(f"Student answers saved to {output_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save student answers: {e}")

if __name__ == "__main__":
    # Example usage: process 'student_answers/' directory
    sp = StudentProcessor()
    student_folder = "student_answers"
    sp.process_directory(student_folder)
