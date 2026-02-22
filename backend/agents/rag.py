from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState
import logging

class RAGAgent(BaseAgent):
    """
    The core Document Retrieval & Q&A Agent.
    Executes grounded RAG generation using retrieved semantic context.
    """
    
    def __init__(self, llm_client=None, retriever=None):
        super().__init__(llm_client)
        self.retriever = retriever
        
    def execute(self, state: AgentState) -> dict:
        query = state.get("search_query") or state.get("query", "")
        context_chunks = state.get("context_chunks", [])
        context_text = state.get("context_text", "")
        
        # 1. Retrieval (If not already fetched by a pre-processor)
        if not context_chunks and self.retriever:
            print(f"RAG Agent: Retrieving context for '{query}'...")
            context_chunks = self.retriever.search(query)
            
            # Format context text
            if context_chunks:
                context_text = "\n\n".join(
                    [f"FILE: {c.get('source', 'Unknown')}\n{c.get('text', '')}" for c in context_chunks]
                )
            else:
                context_text = ""
                
        # 2. Generation Phase
        if not context_text:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "context_chunks": [],
                "context_text": ""
            }
            
        system_prompt = """You are Galactus — a grounded enterprise knowledge assistant.

Your mandate is to answer the user's Question using strictly and ONLY the provided Context chunks.
The Context contains multiple FILE sections representing raw text extracted from uploaded enterprise documents.

RULES:
1. Read the Context carefully.
2. If the user's question cannot be answered using the Context, explicitly state: "I don't know based on the provided context." DO NOT improvise or use outside knowledge.
3. If the answer is in the Context, provide a clear, concise, and direct response.
4. When you provide facts from the context, naturally weave the filename into your sentence (e.g., "According to 'Annual_Report.pdf', the revenue was..."). DO NOT output raw formatting like `FILE: ...`. Treat each FILE independently.

Accuracy is more important than completeness."""

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        
        print("RAG Agent: Synthesizing Answer...")
        
        # Determine Temperature Mode
        temperature = 0.4
        reasoning = "low"
        
        # Check if analytical routing accidentally fell through
        if state.get("intent") == "analytical":
            temperature = 0.7
            reasoning = "high"
            
        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning,
            stream=state.get("streaming_callback") is not None
        )
        
        # If streaming is enabled, `generate` returns a generator (or we handled the callback inside if it's the old API style, let's assume we build the final string string either way in our client wrapper)
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
            "sources": context_chunks
        }
