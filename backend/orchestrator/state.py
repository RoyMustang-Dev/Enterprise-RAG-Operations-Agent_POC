from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    The shared state of the conversation graph.
    All agents read from and mutate this state.
    """
    # Inputs
    query: str
    chat_history: List[Dict[str, str]]  # Format: [{"role": "user", "content": "..."}, ...]
    
    # Routing & Context
    intent: Optional[str]               # e.g., "greeting", "factual", "analytical"
    search_query: Optional[str]         # Rewritten query if applicable
    context_chunks: List[Dict[str, Any]]# The raw chunks from the vector DB
    context_text: str                   # The formatted string of context
    
    # Outputs
    answer: str
    sources: List[Dict[str, Any]]
    
    # Traceability
    confidence: float
    verifier_verdict: str
    is_hallucinated: bool
    optimizations: Dict[str, Any]
    
    # Internal Pipeline State
    current_agent: Optional[str]
    streaming_callback: Optional[Any]   # Pass stream handler through state
