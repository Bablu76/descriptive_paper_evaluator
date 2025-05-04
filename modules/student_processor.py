# --- START OF FILE student_processor.py ---

import os
import json
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pathlib import Path # Use Pathlib

class StudentProcessor:
    """
    Processes student answer PDFs, extracting text and using OCR if needed.
    Saves results to a specified output JSON file.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        # Optional: Check Tesseract installation here?
        try:
            pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version {pytesseract.get_tesseract_version()} detected.")
        except pytesseract.TesseractNotFoundError:
            self.logger.warning("Tesseract executable not found or not in PATH. OCR functionality will fail.")
        except Exception as e:
             self.logger.warning(f"Could not get Tesseract version: {e}. OCR might not work.")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a student PDF. If a page has no text, attempts OCR (Tesseract).
        """
        self.logger.info(f"Processing student PDF: {pdf_path}")
        text_content = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc): # Use enumerate for page numbers
                page_text = ""
                try:
                    page_text = page.get_text("text", sort=True).strip() # Get text, sort for reading order
                except Exception as text_extract_e:
                    self.logger.warning(f"Error extracting text from page {page_num} of {pdf_path}: {text_extract_e}. Attempting OCR.")
                    page_text = "" # Ensure it's empty if extraction failed

                if page_text:
                    text_content += page_text + "\n\n" # Double newline between pages maybe?
                else:
                    # Page likely contains image or text extraction failed; apply OCR
                    self.logger.info(f"No text found/extracted on page {page_num} of {pdf_path}, attempting OCR...")
                    try:
                        # Use higher DPI for potentially better OCR
                        pix = page.get_pixmap(dpi=300, alpha=False) # alpha=False for standard RGB
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        # Optional: Add language hint if known e.g., lang='eng'
                        ocr_text = pytesseract.image_to_string(img) # Add try-except?
                        if ocr_text.strip():
                            self.logger.info(f"OCR successful for page {page_num}.")
                            text_content += ocr_text.strip() + "\n\n"
                        else:
                            self.logger.warning(f"OCR on page {page_num} yielded no text.")
                    except pytesseract.TesseractNotFoundError:
                         self.logger.error("Tesseract not found. Cannot perform OCR. Install Tesseract and ensure it's in PATH.")
                         text_content += "[OCR FAILED: Tesseract not found]\n\n"
                         # Maybe break or continue? Continue processing other pages/files.
                    except Exception as ocr_e:
                         self.logger.error(f"OCR failed for page {page_num} of {pdf_path}: {ocr_e}")
                         text_content += f"[OCR FAILED: {ocr_e}]\n\n" # Include error message
            doc.close() # Explicitly close the document
            return text_content.strip() # Final strip
        except Exception as e:
            self.logger.error(f"Failed to open or process PDF {pdf_path}: {e}")
            return "[PDF Processing Error]"


    def process_directory(self, folder_path: str, output_path: str = "student_answers.json"): # Added output_path parameter
        """
        Process all PDF files in a directory. Saves output to the specified JSON file path.
        """
        input_dir = Path(folder_path)
        output_file = Path(output_path)

        if not input_dir.is_dir():
             self.logger.error(f"Input directory not found: {input_dir}")
             return None # Indicate failure

        self.logger.info(f"Processing student PDFs from directory: {input_dir}")
        results = []
        pdf_files = sorted(list(input_dir.glob("*.pdf"))) # Get list of PDFs
        self.logger.info(f"Found {len(pdf_files)} PDF files to process.")

        if not pdf_files:
             self.logger.warning("No PDF files found in the directory.")
             # Should we still write an empty JSON? Yes, consistent behavior.
        else:
             # Use tqdm if processing many files
             from tqdm import tqdm
             for pdf_path in tqdm(pdf_files, desc="Processing Student Answers"):
                # Extract a meaningful student ID (usually filename without extension)
                student_id = pdf_path.stem # Pathlib's stem gets filename without suffix
                text = self.extract_text_from_pdf(str(pdf_path))
                results.append({"student_id": student_id, "answer_text": text})

        # Save results to the specified output JSON file
        self.logger.info(f"Saving processed student answers to: {output_file}")
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Structure should be {"students": [...]}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"students": results}, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Successfully saved student answers to {output_file}")
            return str(output_file) # Return path on success
        except Exception as e:
            self.logger.error(f"Failed to save student answers JSON to {output_file}: {e}")
            return None # Indicate failure


# Example usage: process 'student_answers/' directory, save to 'processed_student_data.json'
if __name__ == "__main__":
     # Create dummy student answers dir if needed
    student_dir_test = "student_answers"
    os.makedirs(student_dir_test, exist_ok=True)
     # You might want to put actual (small) sample PDFs here for testing
     # e.g., copy a simple text PDF named S101.pdf into the folder

    sp = StudentProcessor()
    output_json_path = "processed_student_data.json" # Define output path
    sp.process_directory(student_dir_test, output_path=output_json_path)

    # Verify output (optional)
    if Path(output_json_path).is_file():
         print(f"Output generated at {output_json_path}")
    else:
         print("Output file was not generated.")


# --- END OF FILE student_processor.py ---