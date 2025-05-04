# --- START OF FILE answer_generator.py ---

import json
import re
import logging
import torch
import os
import time
from collections import Counter
from typing import List, Optional # Added Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
# from huggingface_hub.inference._generated.types import TextGenerationParameters
from huggingface_hub.utils import HfHubHTTPError # Keep this one from utils
from huggingface_hub.errors import InferenceTimeoutError # Import the error from its new location
import requests

# Import transformers components for the local fallback model
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as transformers_logging

# Local module imports
from document_processor import DocumentProcessor
from vector_store import VectorStore

# Suppress verbose transformers logging
transformers_logging.set_verbosity_error()

# Configure logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_API_TOKEN:
    logger.warning("HUGGINGFACE_API_TOKEN not found. API calls will fail, relying on local fallback.")
# -----------------------------------

# --- Configuration ---
DEFAULT_API_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_FALLBACK_MODEL_ID = "google/flan-t5-base" # Reasonable local fallback
# -----------------------------------

class AnswerGenerator:
    """
    Generates model answers using the Hugging Face Inference API primarily,
    with a fallback to a local Hugging Face model if the API fails.
    Uses API token from .env.
    """
    def __init__(self,
                 vector_store: VectorStore,
                 api_model_id: str = DEFAULT_API_MODEL_ID,
                 fallback_local_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
                 syllabus_path: str = 'syllabus.json',
                 device: str = 'cpu'): # Device for local model

        self.vector_store = vector_store
        self.api_model_id = api_model_id
        self.fallback_local_model_id = fallback_local_model_id
        self.device_setting = 0 if device=='cuda' and torch.cuda.is_available() else -1 # For pipeline device

        # --- Initialize HF Inference API Client ---
        logger.info(f"Attempting to initialize HF Inference Client for model '{self.api_model_id}'")
        self.hf_client: Optional[InferenceClient] = None # Type hint as Optional
        if HF_API_TOKEN:
            try:
                self.hf_client = InferenceClient(model=self.api_model_id, token=HF_API_TOKEN)
                # Consider adding a test ping here if needed
                logger.info(f"Hugging Face Inference Client initialized for API model: {self.api_model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize HF Inference Client (will rely on fallback): {e}")
                self.hf_client = None
        else:
             logger.warning("HF API Token missing. Relying solely on local fallback model.")
             self.hf_client = None # Explicitly set to None

        # --- Initialize Local Fallback Model ---
        self.local_generator = None
        if self.fallback_local_model_id:
            logger.info(f"Attempting to initialize local fallback model: '{self.fallback_local_model_id}' on device: {'cuda' if self.device_setting == 0 else 'cpu'}")
            try:
                 # Use pipeline for convenience
                 self.local_generator = pipeline(
                      "text2text-generation",
                      model=self.fallback_local_model_id,
                      tokenizer=self.fallback_local_model_id, # Explicitly state tokenizer
                      device=self.device_setting # Use -1 for CPU, 0 for first GPU
                 )
                 logger.info(f"Local fallback model '{self.fallback_local_model_id}' initialized successfully.")
            except Exception as e:
                 logger.error(f"Failed to initialize local fallback model '{self.fallback_local_model_id}': {e}")
                 logger.error("Local fallback will not be available.")
                 self.local_generator = None
        else:
             logger.warning("No fallback_local_model_id specified. No local fallback available.")
             self.local_generator = None


        # --- Load Syllabus (remains the same) ---
        logger.info(f"Loading syllabus from {syllabus_path}")
        syllabus = DocumentProcessor.load_syllabus(syllabus_path)
        if syllabus:
            self.co_map = {str(key).lower(): desc for key, desc in syllabus.get("Course Outcomes Mapping", {}).items()}
            if not self.co_map:
                self.co_map = {f"co{i+1}".lower(): desc for i, desc in enumerate(syllabus.get("Course Outcomes", []))}
            logger.info(f"Loaded {len(self.co_map)} Course Outcomes from syllabus.")
        else:
            self.co_map = {}; logger.warning(f"Syllabus {syllabus_path} failed to load or is empty.")
        # ----------------------------------------

    def _generate_with_api(self, prompt: str, marks: int) -> Optional[str]:
         """Internal method to attempt generation using the HF Inference API."""
         if self.hf_client is None:
             logger.info("Skipping API attempt: HF Client not available.")
             return None # Indicate API is not available

         retries = 3
         delay = 5 # seconds
         last_error = None

         for attempt in range(retries):
              try:
                  logger.info(f"Sending request to HF Inference API (Attempt {attempt + 1}/{retries})...")
                  response_text = self.hf_client.text_generation(prompt=prompt, max_new_tokens=768,
                                do_sample=False,
                                return_full_text=False,
                                stop_sequences=["[/INST]", "</s>"])
                  logger.info(f"API Call successful (Attempt {attempt + 1})")
                  logger.debug(f"Raw answer received: {response_text[:150]}...")
                  return response_text.strip() # Success

              except InferenceTimeoutError as e:
                  last_error = e; logger.warning(f"API Timeout (Attempt {attempt + 1}/{retries}). Retrying..."); time.sleep(delay); delay *= 2; continue
              except HfHubHTTPError as e:
                   last_error = e; status = e.response.status_code if hasattr(e, 'response') else 'Unknown'
                   logger.warning(f"API HTTP Error (Status: {status}, Attempt {attempt + 1}/{retries}): {e}")
                   if status == 401: logger.error("API Unauthorized (401). Cannot use API."); return None # Fatal
                   if status in [429, 503]: logger.info("Retrying potentially transient API error..."); time.sleep(delay); delay *= 2; continue
                   logger.error("Non-retryable API HTTP error."); return None # Fatal for API attempt
              except requests.exceptions.RequestException as e:
                  last_error = e; logger.warning(f"Network error during API call (Attempt {attempt+1}/{retries}): {e}. Retrying..."); time.sleep(delay); delay *= 2; continue
              except Exception as e:
                  last_error = e; logger.error(f"Unexpected error during API call (Attempt {attempt+1}/{retries}): {e}"); import traceback; logger.error(f"Traceback: {traceback.format_exc()}"); return None # Fatal for API attempt

         logger.error(f"API generation failed after {retries} attempts. Last error: {last_error}")
         return None # API attempt failed after retries

    def _generate_with_local(self, prompt: str, marks: int) -> Optional[str]:
        """Internal method to attempt generation using the local fallback model."""
        if self.local_generator is None:
            logger.warning("Skipping local fallback: local generator not available.")
            return None

        # Parameters for local pipeline (adjust as needed for T5 etc.)
        local_params = {
            "max_length": 512,          # Max sequence length (prompt+answer) for T5
            "min_length": max(15, marks * 5), # T5 often needs min_length
            "num_beams": 4,             # Beam search often improves quality
            "early_stopping": True,
            "no_repeat_ngram_size": 2   # Prevent repetitive loops
        }

        try:
             logger.info(f"Attempting generation with local model: '{self.fallback_local_model_id}'")
             # Note: Pipeline takes care of tokenization etc.
             output = self.local_generator(prompt, **local_params)
             if output and isinstance(output, list) and 'generated_text' in output[0]:
                 generated_text = output[0]['generated_text']
                 logger.info("Local generation successful.")
                 logger.debug(f"Local raw answer: {generated_text[:150]}...")
                 return generated_text.strip()
             else:
                 logger.error(f"Local generator returned unexpected output format: {output}")
                 return None
        except Exception as e:
             logger.error(f"Error during local model generation: {e}")
             import traceback
             logger.error(f"Traceback: {traceback.format_exc()}")
             return None


    def generate_answer(self, question: str, marks: int = 0, co_id: str = None, rubric: List[str] = None) -> str:
        """
        Generates a single model answer, trying the API first and falling back to local model if needed.
        """
        rubric = rubric or []
        # --- Get Context (same logic) ---
        context_text = "" ; context_chunks = []
        if self.vector_store:
            try: context_chunks = self.vector_store.retrieve_context(question, top_k=3); context_text = "\n".join(context_chunks) if context_chunks else ""; logger.debug(f"Retrieved {len(context_chunks)} context chunks.")
            except Exception as e: logger.error(f"Failed to retrieve context: {e}"); context_text = "[Context retrieval failed]"
        else: logger.warning("Vector store not provided."); context_text = ""

        # --- Prepare Prompt Components (same logic) ---
        co_key = str(co_id).lower() if co_id else None
        co_text = self.co_map.get(co_key, "") if co_key else ""
        rubric_text = "\n".join(f"- {point.strip()}" for point in rubric if point.strip()) if rubric else "No specific rubric points provided."
        ctx_block = f"Relevant Background Context from Textbook:\n---\n{context_text}\n---\n\n" if context_text and context_text != "[Context retrieval failed]" else ""

        # --- API Prompt (Mistral Instruct format) ---
        api_prompt = f"[INST] You are an expert university teaching assistant generating a high-quality model answer for an exam question worth {marks} marks.\n"
        # ... (rest of the detailed API prompt instructions as defined before) ...
        api_prompt += f"Generate the Model Answer now: [/INST]"

        # --- Try API Generation ---
        api_result = self._generate_with_api(api_prompt, marks)

        if api_result is not None:
             # API succeeded (or returned a non-None error string which we treat as the result)
             # Let's ensure it's not an explicit error message we want to bypass
             if not api_result.startswith("[Error:"):
                 logger.info("Using result from Hugging Face Inference API.")
                 return api_result
             else:
                  logger.warning(f"API attempt returned an error message: {api_result}. Proceeding to fallback.")
                  # Fall through to local fallback

        # --- API Failed or Unavailable - Try Local Fallback ---
        logger.info("API failed or unavailable. Attempting local fallback generation...")

        # --- Local Prompt (Simpler format for models like T5) ---
        local_prompt = f"Generate a model answer for the following university exam question."
        local_prompt += f" The question is worth {marks} marks."
        if co_text: local_prompt += f" Address Course Outcome: {co_id} - {co_text}."
        local_prompt += f"\n\nQuestion:\n{question}"
        if ctx_block: local_prompt += f"\n\nRelevant Context:\n{context_text}"
        local_prompt += f"\n\nKey Rubric Points:\n{rubric_text}"
        local_prompt += f"\n\nAnswer:"
        #------------------------------

        local_result = self._generate_with_local(local_prompt, marks)

        if local_result is not None:
             logger.info(f"Using result from local fallback model '{self.fallback_local_model_id}'.")
             return local_result
        else:
             logger.error("Both API and local fallback generation failed.")
             # Return a consolidated error message
             api_status = "Unavailable/Failed" if api_result is None or api_result.startswith("[Error:") else "OK but rejected?"
             local_status = "Unavailable/Failed"
             return f"[Error: Generation Failed (API Status: {api_status}, Fallback Status: {local_status})]"


    def extract_keywords(self, answer: str) -> List[str]:
        # [No changes needed here]
        if not isinstance(answer, str) or not answer: return []
        try:
            words = re.findall(r'\b[A-Z][a-zA-Z-]*[a-zA-Z]\b|\b[a-z]{3,}\b', answer)
            stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but','is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had','do', 'does', 'did', 'will', 'would', 'should', 'can', 'could', 'may', 'might','that', 'this', 'these', 'those', 'it', 'its', 'with', 'as', 'by', 'from','about', 'above', 'below', 'into', 'out', 'over', 'under', 'again', 'further','then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any','both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor','not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'answer','question', 'context', 'following', 'point', 'points', 'exam', 'model', 'generate', 'using', 'based', 'also'}
            keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
            if not keywords: return []
            keyword_counts = Counter(kw.lower() for kw in keywords)
            return [word for word, _ in keyword_counts.most_common(7)]
        except Exception as e: logger.error(f"Error extracting keywords: {e}"); return []


    def generate_all(self, questions_file: str, output_file: str):
         """
         Generates answers for all questions (trying API, then fallback) and saves to output_file.
         """
         # Check if *any* generator is available
         if self.hf_client is None and self.local_generator is None:
              logger.error("FATAL: Both API client and local generator failed to initialize. Cannot generate answers.")
              return None

         logger.info(f"Starting generation of all model answers from {questions_file} (API Preferred: {self.api_model_id}, Fallback: {self.fallback_local_model_id or 'None'})")
         try: questions = DocumentProcessor.load_questions(questions_file)
         except Exception as e: logger.error(f"Error loading questions file {questions_file}: {e}"); return None
         if not questions: logger.error(f"No questions loaded from {questions_file}. Aborting."); return None

         results = []
         logger.info(f"Generating answers for {len(questions)} questions...")
         # Import tqdm here if desired for generate_all
         try: from tqdm import tqdm
         except ImportError: tqdm = lambda x, **kwargs: x # No progress bar if tqdm not installed

         for q in tqdm(questions, desc="Generating Model Answers"):
             question_text = q.get('Question')
             if not question_text: logger.warning(f"Skipping entry missing 'Question': {q}"); continue

             marks = q.get('Marks', 0); co_id = q.get('CO'); rubric = q.get('Rubric', [])
             if not isinstance(rubric, list): logger.warning(f"Rubric for Q '{question_text[:30]}...' not a list."); rubric = []

             # generate_answer now contains the API/fallback logic
             model_ans = self.generate_answer(question_text, marks, co_id, rubric)
             keywords = self.extract_keywords(model_ans)

             results.append({'question': question_text,'marks': marks,'CO': co_id,'keywords': keywords,'model_answer': model_ans})

         logger.info(f"Saving {len(results)} generated answers to {output_file}")
         try:
             output_dir = os.path.dirname(output_file)
             if output_dir: os.makedirs(output_dir, exist_ok=True)
             with open(output_file, 'w', encoding='utf-8') as f:
                 json.dump({'answers': results}, f, indent=2, ensure_ascii=False)
             logger.info(f"Model answers successfully saved to {output_file}.")
             return output_file
         except Exception as e: logger.error(f"Failed to save model answers to {output_file}: {e}"); return None

# Example usage (if run directly)
if __name__ == '__main__':
     # Uses HF_API_TOKEN from .env if available, otherwise relies on local flan-t5-base
     logger.info("Running AnswerGenerator standalone example with API/Local Fallback.")

     class DummyVectorStore:
         def retrieve_context(self, query, top_k): return ["Dummy context 1.", "Dummy context 2."]
     dummy_vs = DummyVectorStore()

     if not os.path.exists("syllabus.json"):
         with open("syllabus.json", "w") as f: json.dump({"Course Outcomes": ["Understand basic AI.", "Apply algorithms."]}, f)
     if not os.path.exists("questions.json"):
         with open("questions.json", "w") as f: json.dump({"questions": [{"Question": "Explain the concept of Retrieval-Augmented Generation (RAG).", "Marks": 4, "CO": "CO1", "Rubric": ["Define RAG", "Retriever Role", "Generator Role"]}]}, f)

     # Initialize with defaults (Mistral API, Flan-T5 Base fallback)
     generator = AnswerGenerator(vector_store=dummy_vs, syllabus_path="syllabus.json")

     if generator.hf_client or generator.local_generator: # Check if at least one worked
        output = generator.generate_all(questions_file="questions.json", output_file="test_fallback_api_answers.json")
        if output:
             logger.info(f"Example generation complete. Output: {output}")
             with open(output, 'r') as f:
                 data = json.load(f);
                 if data.get('answers'):
                     print("\n--- Example Generated Answer ---")
                     print(f"Q: {data['answers'][0].get('question')}")
                     print(f"A: {data['answers'][0].get('model_answer')}")
                     print("------------------------------\n")
        else:
             logger.error("Example generation failed.")
     else:
         logger.error("Cannot run example: Both API client and local generator failed to initialize.")


# --- END OF FILE answer_generator.py ---