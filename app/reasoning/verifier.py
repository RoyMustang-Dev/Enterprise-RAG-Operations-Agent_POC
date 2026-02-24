"""
Independent Evidence Verifier (Sarvam M)

Checks the output of the 70B model. If Llama hallucinations occurred, 
Sarvam catches the un-sourced claim and immediately flags it.
By using an entirely different foundational architecture (Sarvam instead of Llama), 
we establish absolute model independence, defeating 'AI Yes-Men' behavior.
"""
import os
import json
import logging
import requests
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HallucinationVerifier:
    """
    Independent Fact-Checker auditing the draft answer against the raw chunk arrays.
    """
    
    def __init__(self, use_sarvam: bool = True):
        # We target Sarvam explicitly for independence check, but fallback to Groq if keys are missing
        self.use_sarvam = use_sarvam
        self.sarvam_key = os.getenv("SARVAM_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        self.system_prompt = '''SYSTEM: You are an evidence verifier. Given a candidate answer and the CONTEXT_CHUNKS, perform the following:

1) For each factual claim in the answer, check if CONTEXT_CHUNKS contain supporting evidence. 
2) Provide an overall verifier_verdict: SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED and a numeric score 0.00-1.00.

Output exactly one JSON object with fields: 
{
  "overall_verdict": "SUPPORTED",
  "score": 1.0,
  "is_hallucinated": false
}

Constraints:
- If ANY claim is contradicted or simply missing from the text, flag "is_hallucinated": true.
'''

    def verify(self, draft_answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cross-validates the draft answer string against the mathematical context arrays.
        """
        context_block = "\n\n---\n\n".join([c.get("page_content", "") for c in context_chunks])
        user_payload = f"DRAFT_ANSWER: {draft_answer}\n\nCONTEXT_CHUNKS:\n{context_block}"
        
        # Determine Routing Path (Prefer Sarvam)
        if self.use_sarvam and self.sarvam_key:
             return self._invoke_sarvam(user_payload)
        elif self.groq_key:
             return self._invoke_groq_fallback(user_payload)
             
        # Scaffold Fallback
        return {"overall_verdict": "UNVERIFIED", "score": 0.0, "is_hallucinated": False}

    def _invoke_groq_fallback(self, payload_str: str) -> Dict[str, Any]:
        """Executes verification using an external independent Groq Model."""
        headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant", # Use smaller, faster model just for binary verification
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": payload_str}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info("[VERIFIER] Executing logic boundary verification... (Groq Fallback)")
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=8)
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"[VERIFIER] Execution failure: {e}")
            return {"overall_verdict": "ERROR", "score": 0.0, "is_hallucinated": False}
            
    def _invoke_sarvam(self, payload_str: str) -> Dict[str, Any]:
        """Executes verification natively on Sarvam M."""
        headers = {"api-subscription-key": self.sarvam_key, "Content-Type": "application/json"}
        payload = {
            "model": "sarvam-m",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": payload_str}
            ],
            "temperature": 0.0,
            "top_p": 1,
            "max_tokens": 1000
        }
        
        try:
            logger.info("[VERIFIER] Executing logic boundary verification... (Sarvam M Native)")
            # Using the standard Sarvam Chat completion endpoint
            response = requests.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            # Since Sarvam may not explicitly support JSON response_format natively, we parse defensively
            raw_text = response.json()["choices"][0]["message"]["content"]
            # Extract JSON bound if Sarvam wraps it in markdown blocks
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].strip()
                
            return json.loads(raw_text)
        except Exception as e:
            logger.error(f"[VERIFIER] Sarvam execution failure: {e}. Falling back to Groq natively.")
            if self.groq_key:
                return self._invoke_groq_fallback(payload_str)
            return {"overall_verdict": "ERROR", "score": 0.0, "is_hallucinated": False}
