"""
Output Formatter

This module sits at the very end of the routing pipeline just before the 
FastAPI framework ships the HTTP Response back to the Client UI.

It ensures that the mathematical provenance arrays returned by the 70B model
are injected as Markdown `[Citation: ID]` inline links within the raw text.
"""
import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Standardizes the LLM JSON Outputs into clean Markdown syntax for Streamlit/React.
    """
    
    @staticmethod
    def format_output(answer: str, provenance: List[Dict[str, str]]) -> str:
        """
        Appends the structured citation bounding box to the end of the semantic answer.
        
        Args:
            answer (str): The raw paragraph emitted by Llama-70B.
            provenance (List[dict]): The array of utilized source IDs.
            
        Returns:
            str: The Enterprise-ready markdown block.
        """
        if not provenance:
            logger.info("[FORMATTER] No provenance array detected. Bypassing citation injection.")
            return answer
            
        formatted_str = f"{answer}\n\n### 📑 Enterprise Verification Sources\n\n"
        
        # Deduplicate sources while preserving order inherently
        seen_sources = set()
        for idx, item in enumerate(provenance, 1):
             source_id = item.get("source_id", "Unknown")
             if source_id not in seen_sources:
                  seen_sources.add(source_id)
                  quote = item.get("quote", "").replace('\n', ' ').strip()
                  # Limit quote length visually 
                  if len(quote) > 100:
                      quote = quote[:97] + "..."
                      
                  formatted_str += f"- **[{idx}] {source_id}**: _{quote}_\n"
                  
        return formatted_str
        
    @staticmethod
    def construct_final_response(state: Dict[str, Any]) -> Dict[str, Any]:
         """
         Prepares the final dictionary for the FastAPI Route layer schema `ChatResponse`.
         """
         formatted_answer = ResponseFormatter.format_output(state.get("answer", ""), state.get("sources", []))
         
         # Append visual warning if the Independent Verifier caught a logic flaw
         if state.get("is_hallucinated", False):
              logger.warning("[FORMATTER] Injecting Hallucination Warning into final payload.")
              formatted_answer = f"⚠️ **Enterprise Guard Warning: The logic engine has flagged portions of this response as unverified or contradicted by the Source Data.**\n\n{formatted_answer}"
              
         return {
             "answer": formatted_answer,
             "sources": state.get("sources", []),
             "confidence": state.get("confidence", 0.0),
             "verifier_verdict": state.get("verifier_verdict", "UNVERIFIED"),
             "is_hallucinated": state.get("is_hallucinated", False),
             "optimizations": state.get("optimizations", {})
         }
