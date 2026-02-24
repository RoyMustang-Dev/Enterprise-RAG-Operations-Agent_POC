"""
Core Synthesis Engine (Llama 70B)

This is the primary mathematical reasoning brain of the Enterprise RAG System.
It explicitly accepts the Top-5 strictly reranked Vector DB chunks and forces the
model to synthesize an answer bounded entirely by those chunks.

It explicitly prevents hallucination via aggressive system prompt engineering.
"""
import os
import json
import logging
import requests
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SynthesisEngine:
    """
    Executes the dense grounding logic utilizing Groq's largest available reasoning model.
    """
    
    def __init__(self, model_override: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override
        
        # The rigid System Prompt instructed by the Enterprise Blueprint
        self.system_prompt = '''SYSTEM: You are the enterprise-grade reasoning brain. Your responsibility is to synthesize final answers using ONLY the provided CONTEXT. Follow these rules EXACTLY.

1) INPUTS available: 
   - USER_PROMPT: "<user text>"
   - CONTEXT_CHUNKS: [ {text, source_meta...}, ... ]

2) OUTPUT format (Output exactly one JSON object):
   - "answer": <concise human-readable result, <= 400 words>
   - "provenance": [ {"source_id":"", "quote":"<=40 words"} ... ]
   - "confidence": 0.00-1.00

3) RULES:
   - Use only facts present in CONTEXT_CHUNKS. If you must hypothesize, clearly label as "hypothesis".
   - For each factual claim, attach a provenance entry.
   - If you cannot answer, respond: "I don't know based on the provided documents."
   - Do NOT include chain-of-thought in the final visible output.
'''

    def synthesize(self, user_prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes the final conversational output.
        
        Args:
            user_prompt (str): The raw inquiry.
            context_chunks (List[dict]): The pre-filtered, strictly ranked Top 5 data mappings.
            
        Returns:
            dict: The parsed JSON response containing the answer and raw provenance array.
        """
        if not self.api_key:
             # Fast failure fallback during scaffolding
             return {
                 "answer": "Error: Groq API Key missing. Core Reasoning Brain offline.", 
                 "provenance": [], 
                 "confidence": 0.0
             }
             
        # Flatten the Top-5 chunks into a raw string mapping for the LLM context window
        context_block = "\n\n---\n\n".join([
            f"SOURCE_ID: {c.get('source', 'Unknown')} | CONTENT: {c.get('page_content', '')}"
            for c in context_chunks
        ])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        user_payload = f"USER_PROMPT: {user_prompt}\n\nCONTEXT_CHUNKS:\n{context_block}"
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_payload}
            ],
            # Low temperature enforces Grounding over Creativity
            "temperature": 0.1,
            "max_completion_tokens": 8192,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info(f"[SYNTHESIS] Executing 70B Grounded Generation bounds against {len(context_chunks)} chunks...")
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"[SYNTHESIS PAYLOAD DUMP] {json.dumps(payload, indent=2)}")
                logger.error(f"[SYNTHESIS ERROR DUMP] {response.text}")
                
            response.raise_for_status()
            
            raw_content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(raw_content)
            
            logger.info(f"[SYNTHESIS] Generation Complete. Confidence: {result.get('confidence', 0)}")
            return result
            
        except Exception as e:
             logger.error(f"[SYNTHESIS] Dense generation crashed: {e}")
             return {
                 "answer": "I encountered an internal error trying to process the documents.", 
                 "provenance": [], 
                 "confidence": 0.0
             }
