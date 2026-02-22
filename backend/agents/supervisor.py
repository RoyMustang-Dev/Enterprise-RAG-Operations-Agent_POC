from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState

class SupervisorAgent(BaseAgent):
    """
    The Orchestrator / Semantic Router.
    Analyzes the user's query and routes them to the correct sub-agent.
    """
    
    def execute(self, state: AgentState) -> dict:
        query = state.get("query", "")
        
        # We use a strict prompt to force categorical routing
        supervisor_prompt = """You are the Semantic Router for an Enterprise Knowledge Assistant.
Analyze the user's query and categorize their exact intent into exactly ONE of the following three categories.

CATEGORIES:
1. "smalltalk" : The user is saying hello, greeting, asking how you are, thanking you, or engaging in general chit-chat unrelated to documents. (e.g. "hi", "how are you", "thanks", "hhi", "hellow")
2. "analytical" : The user is asking to compare data, calculate something, extract a specific deeply nested fact, or requesting you to "reason" across the documents. (e.g. "Compare X and Y", "Is Aditya eligible based on criteria X")
3. "rag" : The user is asking a standard informational question about the documents, requesting summaries, explanations, or facts. (e.g. "What is the policy on X?", "Explain document Y")

OUTPUT INSTRUCTIONS:
Return ONLY the exact category name ("smalltalk", "analytical", or "rag"). Do not include any other text or punctuation.
"""
        try:
            # We enforce low temperature for deterministic routing
            intent = self.llm_client.generate(
                prompt=query,
                system_prompt=supervisor_prompt,
                temperature=0.0
            ).strip().lower()
            
            # Clean up potential LLM hallucination in the routing format
            if "smalltalk" in intent or "greet" in intent or "hello" in intent:
                intent = "smalltalk"
            elif "analytical" in intent or "compare" in intent or "reason" in intent:
                intent = "analytical"
            else:
                intent = "rag"
                
            return {"intent": intent}
        except Exception as e:
            print(f"Supervisor Error: {e}")
            # Safe fallback if the LLM router goes down
            return {"intent": "rag"}
