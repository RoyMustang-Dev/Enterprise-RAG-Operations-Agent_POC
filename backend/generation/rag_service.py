from typing import List, Dict, Tuple
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore
from backend.generation.llm_provider import OllamaClient
from sentence_transformers import CrossEncoder

class RAGService:
    """
    Orchestrates Retrieval-Augmented Generation (RAG) with a Governance Layer.
    1. Query Classifier
    2. Source Filter
    3. Retrieval (FAISS)
    4. Chunk Reranker (Cross-Encoder)
    5. Context Limiter
    6. Structured Prompt & Generation
    """
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSStore()
        self.llm_client = OllamaClient()
        # Initialize CrossEncoder for Reranking
        print("Loading CrossEncoder for Reranking...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        print("CrossEncoder loaded successfully.")

    def _query_classifier(self, query: str) -> str:
        """Classifies intent: single-source, comparison, multi-source."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "both"]):
            return "comparison"
        elif ".pdf" in query_lower or ".txt" in query_lower or "file" in query_lower:
            return "single-source" # Likely asking about a specific file
        return "multi-source"

    def _source_filter(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Filters chunks to only include mentioned sources if a specific file is queried."""
        query_lower = query.lower()
        # Basic heuristic: Check if any known document source is explicitly named in the query
        all_sources = self.vector_store.get_all_documents()
        mentioned_sources = [src for src in all_sources if src.lower() in query_lower]
        
        if mentioned_sources:
            filtered = [chunk for chunk in retrieved_chunks if chunk.get('source') in mentioned_sources]
            if filtered:
                return filtered
        return retrieved_chunks

    def _chunk_reranker(self, query: str, chunks: List[Dict], top_n: int = 5) -> List[Dict]:
        """Reranks retrieved chunks using a CrossEncoder."""
        if not chunks:
            return []
        
        # Prepare pairs for cross-encoder scoring
        pairs = [[query, chunk.get('text', '')] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        
        # Add scores to chunks and sort descending
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])
            
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_n]

    def _context_limiter(self, chunks: List[Dict], max_tokens: int = 3000) -> List[Dict]:
        """Limits the context size to prevent overflow. Simple word count approximation."""
        limited_chunks = []
        current_words = 0
        for chunk in chunks:
            words = len(chunk.get('text', '').split())
            if current_words + words > max_tokens:
                break
            limited_chunks.append(chunk)
            current_words += words
        return limited_chunks

    def answer_query(self, query: str, initial_k: int = 20, final_k: int = 5) -> Dict:
        """
        End-to-end RAG pipeline with Governance Layer.
        """
        # 1. Query Classifier (Diagnostic logging)
        intent = self._query_classifier(query)
        print(f"Query Intent: {intent}")

        # 2. Initial Retrieval (Broad Net)
        print(f"Retrieving initial context for: {query}")
        query_embedding = self.embedding_model.generate_embedding(query)
        initial_results = self.vector_store.search(query_embedding, k=initial_k)
        
        if not initial_results:
            return {
                "answer": "I don't know based on the provided context.",
                "sources": []
            }

        # 3. Source Filter (Narrow by filename if mentioned)
        filtered_results = self._source_filter(query, initial_results)

        # 4. Chunk Reranker (Semantic refinement)
        reranked_results = self._chunk_reranker(query, filtered_results, top_n=final_k)

        # 5. Context Limiter (Size safety)
        final_context = self._context_limiter(reranked_results)
            
        # 6. Construct Context String securely
        context_text = ""
        unique_sources = []
        seen_files = set()
        
        for chunk in final_context:
            src = chunk.get('source', 'Unknown')
            context_text += f"--- FILE: {src} ---\n{chunk.get('text', '')}\n\n"
            if src not in seen_files:
                unique_sources.append(chunk)
                seen_files.add(src)
        
        # 7. Structured Prompt
        system_prompt = """You are a Retrieval-Augmented Generation assistant.
You MUST follow these rules strictly:

================ CORE PRINCIPLES ================
1. You may ONLY use information explicitly present in the provided Context.
2. Never use outside knowledge.
3. Never guess.
4. Never fabricate facts.
5. Never merge unrelated sources.
6. Every answer must be grounded in the Context.

If the answer is not present, respond:
"I don't know based on the provided context."

================ SOURCE HANDLING ================
The Context contains multiple FILE sections.
Each section represents a different source.

Rules:
- Treat each FILE independently.
- Do NOT blend information across files unless the user explicitly asks for comparison or reasoning across multiple sources.
- If the user mentions a specific filename:
    - ONLY use that file.
    - If that file does not appear in Context, respond:
      "I do not have access to [filename]. It may not be ingested yet."

================ REASONING MODE ================
Before answering, silently perform these steps:
1. Identify the user intent.
2. Determine which FILE(s) are relevant.
3. Extract exact facts from those FILE(s).
4. If multiple files are required, combine them logically.
5. If files conflict, prefer the most recent or explicitly stated information.
6. If insufficient information exists, say so.

================ QUESTION TYPES ================
Support ALL question types including but not limited to:
- factual lookup
- summaries
- comparisons
- eligibility checks
- profile matching
- reasoning across documents
- extracting structured data
- identifying updates or changes
- answering arbitrary user queries

================ ELIGIBILITY / MATCHING ================
For evaluation or eligibility questions:
- Extract requirements from one source.
- Extract candidate/profile details from another.
- Compare step-by-step.
- Clearly conclude YES / NO / INSUFFICIENT DATA.
- Briefly justify.

================ CONFLICT HANDLING ================
If two sources disagree:
- State the conflict.
- Prefer newer or explicitly labeled information.
- If unclear, say ambiguity exists.

================ OUTPUT RULES ================
- Be concise.
- Be factual.
- No speculation.
- No filler.
- No apologies unless necessary.
- Never mention internal reasoning steps.

If Context is empty or irrelevant:
"I don't know based on the provided context."
================================================="""

        user_prompt = f"""Context:
{context_text}

Question: {query}
Answer:"""

        # 8. Generate
        print("Detailed Log: Calling LLM to Generate Answer...")
        try:
            answer = self.llm_client.generate(user_prompt, system_prompt=system_prompt)
        except Exception as e:
            print(f"Detailed Log: LLM Generation Failed: {e}")
            answer = "Sorry, I encountered an error while generating the response. Please check the backend logs."

        if not answer:
            answer = "The LLM returned an empty response."
        
        return {
            "answer": answer,
            "sources": unique_sources
        }

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    rag = RAGService()
    response = rag.answer_query("What about Akobot?")
    print("\n=== ANSWER ===\n", response["answer"])
