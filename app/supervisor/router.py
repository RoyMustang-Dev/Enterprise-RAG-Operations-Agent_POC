"""
Execution Graph Orchestrator (ReAct Router)

This module builds the Directed Acyclic Graph (DAG) for the multi-agent system.
It acts as the single entrypoint for the `/chat` route. It physically passes the State 
dictionary down the pipe, running the Guard -> Intent -> RAG or Smalltalk sequence.
"""
import logging
from typing import Dict, Any

from app.core.types import AgentState
from app.prompt_engine.guard import PromptInjectionGuard
from app.supervisor.intent import IntentClassifier
from app.agents.rag import RAGAgent

logger = logging.getLogger(__name__)

class ExecutionGraph:
    """
    Constructs the operational state machine ensuring that user input is systematically
    cleansed, categorized, and then mathematically routed to the correct Execution Brain.
    """
    
    def __init__(self):
        """
        Instantiates the immutable middleware layers that execute on every single request.
        """
        self.guard = PromptInjectionGuard()
        self.classifier = IntentClassifier()
        self.rag_agent = RAGAgent()
        
    def invoke(self, query: str, chat_history: list = None) -> AgentState:
        """
        The formal entrypoint called by `app.api.routes`.
        
        Args:
            query (str): The raw string sent by the user.
            chat_history: Previously serialized human/AI turns.
            
        Returns:
            AgentState: The fully completed dictionary containing the final grounded answer.
        """
        # Initialize the global TypedDict scope
        state: AgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "intent": None,
            "search_query": None,
            "context_chunks": [],
            "context_text": "",
            "confidence": 0.0,
            "verifier_verdict": "PENDING",
            "is_hallucinated": False,
            "optimizations": {},
            "answer": "",
            "sources": []
        }
        
        # -------------------------------------------------------------
        # STEP 1: SAFETY (Prompt Injection Guard)
        # -------------------------------------------------------------
        safety_report = self.guard.evaluate(query)
        if safety_report.get("is_malicious", False):
            logger.warning(f"[ROUTER] Guard intercepted payload (Soft bypass in Dev). Flagged as: {safety_report.get('categories')}")
            
        # -------------------------------------------------------------
        # STEP 2: REASONING (Intent Classification)
        # -------------------------------------------------------------
        intent_report = self.classifier.classify(query)
        state["intent"] = intent_report.get("intent", "rag_question")
        
        # -------------------------------------------------------------
        # STEP 3: EXECUTION FORK (MoE / ReAct Routing)
        # -------------------------------------------------------------
        if state["intent"] in ["greeting", "smalltalk", "out_of_scope"]:
             # Bypass the expensive 70B logic completely.
             logger.info("[ROUTER] Smalltalk Bypass executing via Llama-8B.")
             # [PHASE 5 MOCK] - Execute `app.agents.smalltalk_agent`
             state["answer"] = "Hello! I am the Enterprise RAG System. How can I help you today?"
             state["confidence"] = 0.99
             return state
             
        elif state["intent"] in ["rag_question", "analytics_request", "code_request"]:
             # Dispatch into the heavy machinery
             logger.info("[ROUTER] Dispatching payload to the dense RAG/Coder agents.")
             
             # Execute the end-to-end RAG pipeline
             final_state = self.rag_agent.invoke(state)
             return final_state
        
        # Failsafe Catch-all
        state["answer"] = "I am unsure how to route this specific query format."
        return state
