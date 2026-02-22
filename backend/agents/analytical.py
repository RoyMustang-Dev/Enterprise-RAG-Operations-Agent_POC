from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState

class AnalyticalAgent(BaseAgent):
    """
    The Specialized Analytical / Reasoning Agent.
    Executes complex multi-step reasoning, mathematical calculations, and cross-document comparisons.
    """
    
    def __init__(self, llm_client=None, retriever=None):
        super().__init__(llm_client)
        self.retriever = retriever
        
    def execute(self, state: AgentState) -> dict:
        query = state.get("search_query") or state.get("query", "")
        context_chunks = state.get("context_chunks", [])
        context_text = state.get("context_text", "")
        
        print("Analytical Agent: Initializing high-reasoning pipeline...")
        
        # 1. Broad Retrieval (If not fetched)
        # Analytical agent needs a wider context net (e.g. k=40) to compare things, 
        # but for now we hook into the standard retriever
        if not context_chunks and self.retriever:
            print(f"Analytical Agent: Retrieving analytical context for '{query}'...")
            context_chunks = self.retriever.search(query)
            
            if context_chunks:
                context_text = "\n\n".join(
                    [f"FILE: {c.get('source', 'Unknown')}\n{c.get('text', '')}" for c in context_chunks]
                )
            else:
                context_text = ""
                
        # 2. Generation Phase
        if not context_text:
            return {
                "answer": "There is insufficient data in the uploaded documents to perform this analysis.",
                "sources": [],
                "context_chunks": [],
                "context_text": ""
            }
            
        system_prompt = """You are Galactus — an Expert Analytical Knowledge Sandbox.

Your task is to perform deep reasoning, multi-document comparison, or calculation based STRICTLY on the provided Context.
The Context contains multiple FILE sections representing raw text extracted from uploaded enterprise documents.

RULES:
1. Break down complex queries step-by-step.
2. If the user asks for a comparison, clearly structure the differences and similarities.
3. If the user asks if criteria is met based on their provided facts, methodically evaluate their provided facts against the contextual rules and state a definitive conclusion (Eligible / Ineligible).
4. If the provided Context is completely irrelevant to the analysis, state: "There is insufficient data in the uploaded documents to perform this analysis."
5. Never invent rules or criteria that don't exist in the Context.

Think critically before stating your final answer."""

        user_prompt = f"Context:\n{context_text}\n\nAnalytical Query: {query}\nAnalysis:"
        
        print("Analytical Agent: Synthesizing Compute-Heavy Answer...")
        
        # Analytical routing forces High Compute & higher temperature for creativity
        temperature = 0.5
        reasoning = "high"
            
        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning,
            stream=state.get("streaming_callback") is not None
        )
        
        if hasattr(answer, '__iter__') and not isinstance(answer, str):
            final_ans = ""
            for chunk in answer:
                if state.get("streaming_callback"):
                    state["streaming_callback"](chunk)
                final_ans += chunk
            answer = final_ans
            
        import re
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        
        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "context_text": context_text,
            "sources": context_chunks,
            "optimizations": {
                "temperature": temperature,
                "reasoning_effort": reasoning,
                "agent_routed": "AnalyticalAgent"
            }
        }
