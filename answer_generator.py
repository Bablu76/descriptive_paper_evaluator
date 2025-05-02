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
    Uses static syllabus (syllabus.json) and question list (questions.json).
    Each answer may include retrieved context from textbooks.
    """
    def __init__(self, model_name: str = "google/flan-t5-small", device: int = -1):
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
        
        # Load static syllabus
        self.syllabus = self.load_syllabus()
    
    def load_syllabus(self):
        try:
            with open("syllabus.json", 'r', encoding='utf-8') as f:
                syllabus = json.load(f)
            self.logger.info("Syllabus loaded.")
            return syllabus
        except Exception as e:
            self.logger.error(f"Syllabus load error: {e}")
            return {}
    
    def generate_answer(self, question: str, marks: int, course_outcome: str = None, context: List[str] = None) -> str:
        """
        Generate an answer given a question, marks, optional course outcome, and context.
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
        processor = DocumentProcessor()
        
        answers = []
        for q in questions:
            question_text = q.get("question", "")
            marks = q.get("marks", 1)
            co = q.get("CO", None)
            keywords = q.get("keywords", [])
            self.logger.info(f"Question: {question_text[:30]}... Marks: {marks}, CO: {co}")
            
            # Retrieve relevant context from textbooks
            context_chunks = processor.retrieve_context(question_text)
            answer_text = self.generate_answer(question_text, marks, course_outcome=co, context=context_chunks)
            
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
