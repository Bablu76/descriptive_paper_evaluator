# --- START OF FILE student_processor.py ---

import os
import re
import json
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Dict, Any, List

class StudentProcessor:
    """
    Processes student answer PDFs (with OCR fallback) or plain-text files,
    and writes a JSON of the form:
    {
      "students": [
        {
          "student_id": "...",
          "answer": { "q1": "...", "q2": "...", ... }
        },
        ...
      ]
    }
    """

    def __init__(self, delimiter: str = r"(\d+\.\d+[a-z]?)"):
        """
        :param delimiter: regex for question IDs in text mode.
                          By default it matches patterns like "1.0a", "2.1b", etc.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.delim_pattern = delimiter

        # Check tesseract availability
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract v{version} detected.")
        except Exception as e:
            self.logger.warning(f"Tesseract unavailable: {e}. OCR will fail.")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from PDF; falls back to OCR on pages with no text."""
        self.logger.info(f"PDF mode: processing {pdf_path}")
        text_content = []
        try:
            doc = fitz.open(pdf_path)
            for pn, page in enumerate(doc):
                txt = ""
                try:
                    txt = page.get_text("text", sort=True).strip()
                except Exception:
                    self.logger.debug(f"Page {pn}: text extract failed, OCR next.")
                if not txt:
                    self.logger.info(f"Page {pn}: running OCR")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    txt = pytesseract.image_to_string(img).strip()
                text_content.append(txt)
            doc.close()
            return "\n\n".join(text_content).strip()
        except Exception as e:
            self.logger.error(f"Failed PDF processing for {pdf_path}: {e}")
            return ""

    def load_plain_text(self, txt_path: str) -> str:
        """Loads an entire .txt file into a string."""
        self.logger.info(f"Text mode: loading {txt_path}")
        try:
            return Path(txt_path).read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to read text file {txt_path}: {e}")
            return ""

    def parse_text_answers(self, text: str) -> Dict[str, str]:
        """
        Splits `text` on occurrences of the delimiter pattern,
        returning a dict: { "<qid>": "<answer>" }.
        """
        # include the delimiter in the split so we capture all parts
        parts = re.split(self.delim_pattern, text)
        # parts: [prelude, qid1, ans1, qid2, ans2, ...]
        answers: Dict[str, str] = {}
        for i in range(1, len(parts)-1, 2):
            qid = parts[i].strip()
            ans = parts[i+1].strip()
            answers[qid] = ans
        return answers

    def process_file(self, path: Path, input_type: str) -> Dict[str, Any]:
        """
        Process a single student file.
        :returns: dict with keys "student_id" and "answer" (the Q→A mapping)
        """
        sid = path.stem
        if input_type == 'pdf':
            raw = self.extract_text_from_pdf(str(path))
            # We assume the PDF has Q markers inline (like Q1, Q2) or you may
            # adapt `parse_text_answers` delim to look for "Q(\d+)" instead.
            answers = self.parse_text_answers(raw)
        elif input_type == 'text':
            raw = self.load_plain_text(str(path))
            answers = self.parse_text_answers(raw)
        else:
            raise ValueError("input_type must be 'pdf' or 'text'")
        return {"student_id": sid, "answer": answers}

    def process_directory(self,
                          folder: str,
                          input_type: str = 'pdf',
                          output_path: str = "student_answers.json"
                         ) -> str:
        """
        Walks `folder`, processes each .pdf or .txt (depending on mode),
        collects results under "students", writes JSON to `output_path`.
        """
        in_dir = Path(folder)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"{folder} not a dir")

        glob_ext = "*.pdf" if input_type=='pdf' else "*.txt"
        files = sorted(in_dir.glob(glob_ext))
        self.logger.info(f"Found {len(files)} {glob_ext} files")

        students: List[Dict[str, Any]] = []
        for f in files:
            rec = self.process_file(f, input_type)
            students.append(rec)

        out = {"students": students}
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)

        self.logger.info(f"Wrote {len(students)} records to {out_path}")
        return str(out_path)


# Example stand-alone test
if __name__ == "__main__":
    sp = StudentProcessor(delimiter=r"(\d+\.\d+[a-z]?)")
    # PDF mode:
    sp.process_directory("data/student_answers", input_type='pdf', output_path="out_pdf.json")
    # Text mode:
    sp.process_directory("data/student_answers", input_type='text', output_path="out_text.json")

# --- END OF FILE student_processor.py ---
