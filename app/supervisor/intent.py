"""
ReAct Supervisor Intent Classifier

This module acts as the "Traffic Cop" for the entire architecture.
Before we waste computing power executing a dense 70B RAG pipeline, we pass 
the user's string to a hyper-fast 8B model (`llama-3.1-8b-instant`).

By forcing the 8B model to return strict JSON, we can programmatically fork 
the execution graph (e.g., Short-circuit "Hello" directly to a fast response, 
saving massive compute costs).
"""
import os
import json
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Executes a <400ms inference call establishing the conversational trajectory.
    """
    
    def __init__(self, model_override: str = "llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override
        
        # The rigid JSON schema instructed in the rag-implementation blueprint
        self.system_prompt = '''SYSTEM: You are a high-precision intent classifier. Given the user message, classify into one of: ["greeting","smalltalk","out_of_scope","rag_question","code_request","analytics_request","multimodal_audio","other"]. Output exactly one compact JSON:

{"intent": "<one of the labels above>", "confidence": 0.00-1.00, "route": "<target_agent_name>", "notes": "<optional 1-sentence signal words>"}

Rules:
- If user asks for anything illegal, set intent="out_of_scope".
- If user references files, docs, data, or asks about specific people/entities (e.g. HR reviews, employee data) -> prefer "rag_question".
- If user asks to write/execute code -> "code_request".
- If not confident (<0.5), choose "other" and include notes.
'''

    def classify(self, user_prompt: str) -> Dict[str, Any]:
        """
        Calculates the explicit logical intent of the user.
        
        Args:
            user_prompt (str): The raw string extracted from the HTTP payload.
            
        Returns:
            dict: The dictionary containing the exact routing path to follow.
        """
        if not self.api_key:
             # Scaffolding fallback preventing 500 crashes during Phase 2/3 migration
             logger.warning("[SUPERVISOR] GROQ_API_KEY missing. Defaulting to 'rag_question'.")
             return {
                 "intent": "rag_question", 
                 "confidence": 0.0, 
                 "route": "rag_agent", 
                 "notes": "API Key missing. Forced fallback."
             }
             
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Near determinism. We don't want the router 'hallucinating' paths.
            "temperature": 0.1,
            "max_tokens": 120,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info(f"[SUPERVISOR] Classifying Intent via {self.model_id}...")
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=5)
            response.raise_for_status()
            
            raw_content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(raw_content)
            
            logger.info(f"[SUPERVISOR] Detected Intent -> {result.get('intent')} (Conf: {result.get('confidence')})")
            return result
            
        except requests.exceptions.Timeout:
             logger.error("[SUPERVISOR] Timeout reaching Groq classification API.")
             # Fall-Open to the core RAG processor if routing fails
             return {"intent": "rag_question", "route": "rag_agent", "confidence": 0.1}
        except Exception as e:
             logger.error(f"[SUPERVISOR] Critical LLM execution failure: {e}")
             return {"intent": "rag_question", "route": "rag_agent", "confidence": 0.1}
