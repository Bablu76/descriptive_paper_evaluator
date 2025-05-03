import os
import json
import logging
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import InferenceApi

from document_processor import DocumentProcessor

class AnswerGenerator:
    """
    Generates model answers for given questions using a transformer model.
    Uses syllabus (syllabus.json) to include Course Outcome info, and 
    optionally retrieved context from textbooks via FAISS.
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.model_name = model_name
        self.device = device  # -1 for CPU, >=0 for GPU
        try:
            self.logger.info(f"Loading model {self.model_name}.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.generator = pipeline(
                "text2text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=self.device
            )
        except Exception as e:
            self.logger.error(f"Failed to load local model {self.model_name}: {e}")
            self.generator = None
        
        self.hf_inference = None
        try:
            if self.generator is None:
                self.logger.info("Using Hugging Face Inference API as fallback.")
                self.hf_inference = InferenceApi(
                    repo_id=self.model_name, 
                    token=os.getenv("HUGGINGFACE_API_TOKEN")
                )
        except Exception as e:
            self.logger.error(f"HF Inference API init failed: {e}")
            self.hf_inference = None
        
        # Load static syllabus and prepare CO map
        self.syllabus = self.load_syllabus()
    
    def load_syllabus(self):
        """
        Loads syllabus.json and returns a mapping of CO IDs (e.g., 'co1') to their descriptions.
        """
        try:
            with open("syllabus.json", 'r', encoding='utf-8') as f:
                syllabus = json.load(f)
            self.logger.info("Syllabus loaded.")
            # Convert course outcomes list to a map of co IDs to descriptions
            co_map = {}
            outcomes = syllabus.get("Course Outcomes", [])
            if isinstance(outcomes, list):
                for idx, desc in enumerate(outcomes, start=1):
                    co_key = f"co{idx}"
                    co_map[co_key] = desc
            # If Course Outcomes already a dict, use it directly
            if not co_map and "Course Outcomes" in syllabus and isinstance(syllabus["Course Outcomes"], dict):
                for key, desc in syllabus["Course Outcomes"].items():
                    co_map[key] = desc
            return co_map
        except Exception as e:
            self.logger.error(f"Syllabus load error: {e}")
            return {}
    
    def generate_answer(self, question: str, marks: int, course_outcome: str = None, context: List[str] = None) -> str:
        """
        Generate an answer given a question, marks, optional course outcome, and context.
        The prompt includes the CO description and context if available.
        """
        prompt = ""
        if course_outcome and course_outcome in self.syllabus:
            prompt += f"Course Outcome: {self.syllabus[course_outcome]}\n"
        if context:
            prompt += "Context: " + " ".join(context) + "\n"
        
        prompt += f"Question: {question}\nAnswer (in detail, for {marks} marks):"
        
        max_length = marks * 20  # ~20 tokens per mark
        try:
            if self.generator:
                outputs = self.generator(prompt, max_length=max_length, num_return_sequences=1)
                answer = outputs[0]['generated_text']
            elif self.hf_inference:
                result = self.hf_inference(prompt, parameters={"max_length": max_length})
                answer = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
            else:
                self.logger.error("No model available for generation.")
                return ""
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return ""
        
        return answer.strip()
    
    def generate_all(self, questions_path: str):
        """
        Load questions from JSON and generate answers for each.
        """
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = data.get("questions", [])
        except Exception as e:
            self.logger.error(f"Failed to load questions: {e}")
            return
        
        # Initialize document processor for context retrieval
        try:
            processor = DocumentProcessor(
                textbook_dir="textbooks",
                index_path="corpus.index",
                cache_dir=".cache",
                syllabus_path="syllabus.json",
                model_name="all-mpnet-base-v2",
                metric="cosine",
                index_type="hnsw"
            )
        except Exception as e:
            self.logger.error(f"DocumentProcessor init failed: {e}")
            processor = None
        
        answers = []
        for q in questions:
            question_text = q.get("question_text", "") or q.get("question", "")
            marks = q.get("marks", 1)
            co = q.get("co_id", None) or q.get("CO", None)
            keywords = q.get("keywords", [])
            rubric_items = q.get("rubric", [])
            self.logger.info(f"Question: {question_text[:30]}... Marks: {marks}, CO: {co}")
            
            # Retrieve relevant context from textbooks if available
            context_chunks = []
            if processor:
                try:
                    results = processor.search(question_text, k=3)
                    context_chunks = [r["text"] for r in results if "text" in r]
                except Exception as e:
                    self.logger.error(f"Context retrieval failed: {e}")
                    context_chunks = []
            
            # Generate initial answer
            answer_text = self.generate_answer(question_text, marks, course_outcome=co, context=context_chunks)
            
            # Ensure answer covers rubric items; regenerate up to max attempts if needed
            max_attempts = 3
            attempt = 1
            while attempt <= max_attempts:
                missing = []
                ans_lower = answer_text.lower()
                for item in rubric_items:
                    # Extract keywords by removing common verbs
                    kw = item.lower()
                    for verb in ["define", "explain", "provide", "mention", "the", "and", "for", "in", "with", "its", "key", "steps", "clear", "clearly"]:
                        kw = kw.replace(verb, "")
                    # Remove non-alphanumeric characters
                    kw = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in kw)
                    words = [w.strip() for w in kw.split() if w.strip()]
                    # If none of the keywords appear in answer, mark as missing
                    if words and not any(w in ans_lower for w in words):
                        missing.append(item)
                if not missing:
                    self.logger.info(f"All rubric items covered for question {question_text[:30]}...")
                    break
                if attempt < max_attempts:
                    self.logger.warning(f"Rubric items missing {missing}; regenerating answer (attempt {attempt+1})")
                    answer_text = self.generate_answer(question_text, marks, course_outcome=co, context=context_chunks)
                    attempt += 1
                else:
                    self.logger.error(f"Failed to cover rubric after {max_attempts} attempts. Missing: {missing}")
                    break
            
            answers.append({
                "question": question_text,
                "marks": marks,
                "CO": co,
                "keywords": keywords,
                "model_answer": answer_text
            })
        
        output_path = "model_answers.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"answers": answers}, f, indent=2)
            self.logger.info(f"Model answers saved to {output_path}.")
        except Exception as e:
            self.logger.error(f"Saving model answers failed: {e}")

def generate_answers(questions_path: str):
    gen = AnswerGenerator()
    gen.generate_all(questions_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python answer_generator.py <questions.json>")
    else:
        generate_answers(sys.argv[1])
