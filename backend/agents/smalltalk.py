from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState

class SmalltalkAgent(BaseAgent):
    """
    The Smalltalk/Greeting Agent.
    Bypasses deep retrieval pipelines to offer instant persona responses.
    """
    
    def execute(self, state: AgentState) -> dict:
        print("Smalltalk Agent: Executing instant bypass...")
        answer = "Hello — I’m Galactus. I help answer questions using your uploaded documents and connected knowledge sources.\nYou can ask me about your files, compare documents, extract information, or reason across sources."
        
        # Stream it instantly if callback exists
        streaming_callback = state.get("streaming_callback")
        if streaming_callback:
            import time
            for word in answer.split():
                streaming_callback(word + " ")
                time.sleep(0.01)
                
        return {
            "answer": answer,
            "sources": [],
            "context_chunks": [],
            "context_text": "",
            "confidence": 1.0,
            "verifier_verdict": "SUPPORTED",
            "is_hallucinated": False,
            "optimizations": {
                "short_circuited": True,
                "temperature": 0.2,
                "reasoning_effort": "low",
                "agent_routed": "SmalltalkAgent"
            }
        }
