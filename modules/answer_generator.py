# --- START OF FILE answer_generator.py ---

import json
import re
import logging
import torch
import os
import time
import traceback
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
from transformers import logging as transformers_logging

from .document_processor import DocumentProcessor
from .vector_store import VectorStore

transformers_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# -----------------------------------

# --- Configuration ---
DEFAULT_API_MODEL_ID = "gemini-2.0-flash"
DEFAULT_FALLBACK_MODEL_ID = "google/flan-t5-base"
# -----------------------------------

class AnswerGenerator:
    def __init__(self,
                 vector_store: VectorStore,
                 fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
                 syllabus_path: str = 'syllabus.json',
                 device: str = 'cpu',
                 context_k: int = 3):

        self.vector_store = vector_store
        self.fallback_model_id = fallback_model_id
        self.device_setting = 0 if device == 'cuda' and torch.cuda.is_available() else -1
        self.context_k = context_k

        self.use_gemini = bool(GEMINI_API_KEY)
        logger.info(f"Using GEMINI API: {self.use_gemini}")

        self.local_generator = None
        if self.fallback_model_id:
            try:
                logger.info(f"Initializing local fallback model: '{self.fallback_model_id}'")
                self.local_generator = pipeline(
                    "text2text-generation",
                    model=self.fallback_model_id,
                    tokenizer=self.fallback_model_id,
                    device=self.device_setting
                )
            except Exception as e:
                logger.error(f"Failed to initialize local fallback model '{self.fallback_model_id}': {e}")
                self.local_generator = None

        logger.info(f"Loading syllabus from {syllabus_path}")
        syllabus = DocumentProcessor.load_syllabus(syllabus_path)
        if syllabus:
            self.co_map = {str(key).lower(): desc for key, desc in syllabus.get("Course Outcomes Mapping", {}).items()}
            if not self.co_map:
                self.co_map = {f"co{i+1}".lower(): desc for i, desc in enumerate(syllabus.get("Course Outcomes", []))}
            logger.info(f"Loaded {len(self.co_map)} Course Outcomes from syllabus.")
        else:
            self.co_map = {}
            logger.warning(f"Syllabus {syllabus_path} failed to load or is empty.")

    def _generate_with_gemini(self, prompt: str, max_tokens: int) -> Optional[str]:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found. Cannot use Gemini API.")
            return None

        try:
            model = genai.GenerativeModel(DEFAULT_API_MODEL_ID)  # or 'gemini-pro' if preferred
            response = model.generate_content(prompt)
            return response.text.strip() if hasattr(response, 'text') else None
        except Exception as e:
            logger.error(f"Gemini SDK call failed: {e}")
            logger.debug(traceback.format_exc())
            return None


    def _generate_with_local(self, prompt: str, max_tokens: int) -> Optional[str]:
        if self.local_generator is None:
            logger.warning("Skipping local fallback: local generator not available.")
            return None

        local_params = {
            "max_length": max(512, max_tokens * 2),
            "min_length": max(15, max_tokens // 10),
            "num_beams": 4,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        }

        try:
            logger.info(f"Attempting generation with local model: '{self.fallback_model_id}'")
            output = self.local_generator(prompt, **local_params)
            if output and isinstance(output, list) and 'generated_text' in output[0]:
                return output[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Error during local model generation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    def _extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        if not text:
            return []

        prompt = f"Extract the main keywords from the following text and return them as a comma-separated list: {text[:1000]}"
        keywords = self._generate_with_gemini(prompt, 100)
        if keywords:
            return [k.strip() for k in keywords.split(',') if k.strip()]

        local_result = self._generate_with_local(prompt, 100)
        if local_result:
            return [k.strip() for k in local_result.split(',') if k.strip()]

        try:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            stopwords = {"the", "and", "for", "with", "from", "that", "this", "have", "what", "when", "where", "why", "how"}
            word_counts = Counter(w for w in words if w not in stopwords)
            return [word for word, _ in word_counts.most_common(top_k)]
        except Exception as e:
            logger.error(f"Keyword extraction fallback failed: {e}")
            return []

    def generate_answer(self,
                        question: str,
                        marks: int = 0,
                        co_id: str = None,
                        rubric: List[str] = None,
                        rubrics: List[Dict[str, Any]] = None) -> Tuple[str, List[str]]:

        if rubrics and isinstance(rubrics, list) and rubrics and isinstance(rubrics[0], dict):
            rubric = [f"{r.get('Criteria')} ({r.get('Marks', 1)} mark{'s' if r.get('Marks', 1) > 1 else ''})"
                      for r in rubrics if r.get('Criteria')]
        else:
            rubric = rubric or []

        context_chunks = []
        if self.vector_store:
            try:
                stopwords_pattern = r'\b(the|a|an|in|of|for|to|and|or|is|are|with|by|on|at|what|when|where|why|how|define|explain|discuss|describe|analyze|evaluate|compare|contrast|list|identify|enumerate|outline|illustrate|demonstrate)\b'
                search_query = re.sub(stopwords_pattern, '', question.lower())
                search_query = re.sub(r'\s+', ' ', search_query).strip()
                context_chunks = self.vector_store.retrieve_context(search_query, top_k=self.context_k)
            except Exception as e:
                logger.error(f"Failed to retrieve context: {e}")
                context_chunks = []

        tokens_per_mark = 60
        approx_length = marks * tokens_per_mark
        co_text = self.co_map.get(str(co_id).lower(), "") if co_id else ""
        rubric_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric)) if rubric else "No specific rubric points provided."

        if context_chunks:
            ctx_text = "\n".join(f"- {c[:200]}..." if len(c) > 200 else f"- {c}" for c in context_chunks)
            ctx_block = f"Relevant context (use if helpful):\n{ctx_text}\n\n"
        else:
            ctx_block = ""

        base_prompt = (
            f"You are an expert exam answer writer. Write a model answer for the question below, "
            f"targeted to be worth {marks} marks (approx. {approx_length} tokens).\n\n"
        )
        if co_text:
            base_prompt += f"Course Outcome to address: {co_text}\n\n"

        base_prompt += ctx_block
        base_prompt += f"Question: {question}\n\n"
        base_prompt += (
            "Make sure your answer thoroughly covers all of these key points:\n"
            f"{rubric_text}\n\n"
            "Write in a structured, academic tone with:\n"
            "1. Clear explanation of each concept\n"
            "2. Examples where appropriate (especially for higher-mark questions)\n"
            "3. Proper introduction and conclusion\n"
            "Avoid explicitly mentioning or referencing the marks (e.g., '(2 marks)') in the answer.\n\n"
            "Answer:\n"
        )

        result = self._generate_with_gemini(base_prompt, approx_length)
        if result:
            logger.info("Using result from Gemini API.")
            return result, self._extract_keywords(result)

        logger.info("Gemini API failed. Attempting local fallback generation...")
        local_result = self._generate_with_local(base_prompt, approx_length)
        if local_result:
            logger.info("Using result from local fallback model.")
            return local_result, self._extract_keywords(local_result)

        return "[Error: Generation Failed (API Status: GEMINI Failed, Fallback Status: Unavailable/Failed)]", []

    def generate_all(self, questions_file: str, output_file: str):
        if not self.use_gemini and self.local_generator is None:
            logger.error("No generation method available.")
            return None

        questions = DocumentProcessor.load_questions(questions_file)
        if not questions:
            logger.error("No questions found.")
            return None

        results = []
        for q in questions:
            text = q.get("Question")
            if not text:
                continue

            marks = q.get("Marks", 0)
            co = q.get("CO")
            rubric = q.get("Rubric", [])
            rubrics = q.get("Rubrics", [])

            ans, keywords = self.generate_answer(text, marks, co, rubric, rubrics)
            results.append({
                "question": text,
                "marks": marks,
                "CO": co,
                "keywords": keywords,
                "model_answer": ans
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"answers": results}, f, indent=2)

        logger.info(f"Saved answers to {output_file}")
        return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate model answers from questions JSON.")
    parser.add_argument("--questions", required=True, help="Path to questions.json")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--cache_dir", default="vector_cache", help="Cache directory for VectorStore")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="SentenceTransformer model for embeddings")
    parser.add_argument("--fallback_model", default=DEFAULT_FALLBACK_MODEL_ID, help="Local fallback model")
    parser.add_argument("--index_type", default="ivf", choices=["flat", "hnsw", "ivf"], help="FAISS index type")
    parser.add_argument("--metric", default="cosine", choices=["l2", "cosine"], help="FAISS metric")
    parser.add_argument("--syllabus", default="syllabus.json", help="Path to syllabus.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for local model")
    parser.add_argument("--context_k", type=int, default=3, help="Number of context chunks to retrieve")

    args = parser.parse_args()

    from .vector_store import VectorStore
    vs = VectorStore(
        model_name=args.model,
        index_type=args.index_type,
        metric=args.metric,
        cache_dir=args.cache_dir
    )

    if not vs.load():
        logger.warning("No existing index found; context retrieval may not work properly.")

    generator = AnswerGenerator(
        vector_store=vs,
        fallback_model_id=args.fallback_model,
        syllabus_path=args.syllabus,
        device=args.device,
        context_k=args.context_k
    )

    output_path = generator.generate_all(args.questions, args.output)
    if output_path:
        logger.info(f"Answers saved to {output_path}")
        with open(output_path, 'r') as f:
            data = json.load(f)
            if data.get('answers'):
                print("\n--- Example Generated Answer ---")
                print(f"Q: {data['answers'][0].get('question')}")
                print(f"A: {data['answers'][0].get('model_answer')}")
                print("------------------------------\n")
    else:
        logger.error("Failed to generate answers.")

# --- END OF FILE answer_generator.py ---
