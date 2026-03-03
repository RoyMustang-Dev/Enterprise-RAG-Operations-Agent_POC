"""
Smalltalk Agent (Llama 8B)

This node catches general chatter, intent=greeting, and explicit questions 
about the AI's identity. It strictly queries the LLM combined with the 
Global Agent Bootstrapper Persona so it correctly identifies itself according
to user specs, instead of hardcoded dev strings.
"""
import os
import json
import logging
import requests
from typing import Dict, Any

from app.prompt_engine.groq_prompts.config import get_compiled_prompt, PersonaCacheManager

logger = logging.getLogger(__name__)

class SmalltalkAgent:
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = "llama-3.1-8b-instant" # Fast, cheap model for routing smalltalk
        
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the LLM specifically for persona-driven smalltalk and greeting workflows.
        """
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        
        # Pull Persona details to add to state
        cache = PersonaCacheManager()
        persona = cache.get_persona()
        bot_name = persona.get("bot_name", "Enterprise RAG System") if persona else "Enterprise RAG System"
        state["active_persona"] = bot_name
        
        # Compile System Prompt explicitly using the multimodal/conversational template 
        # (This injects the ReAct persona + strict bounds)
        system_prompt = get_compiled_prompt("multimodal_voice", self.model_id)
        
        # Build messages payload including history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if available, limit to last 4 to keep context clean
        for turn in chat_history[-4:]:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
            
        messages.append({"role": "user", "content": query})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.5, # Let it be slightly creative for smalltalk
            "max_tokens": 512
        }
        
        state["optimizations"]["agent_routed"] = "smalltalk"
        
        try:
            logger.info(f"[SMALLTALK] Sending conversation payload to {self.model_id}")
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            answer = response.json().get("choices")[0].get("message").get("content")
            state["answer"] = answer
            state["confidence"] = 0.99
            
        except Exception as e:
            logger.error(f"[SMALLTALK] LLM Execution failed: {e}")
            state["answer"] = f"Hello! I am {bot_name}. I'm currently experiencing a network interruption."
            state["confidence"] = 0.5
            
        return state
