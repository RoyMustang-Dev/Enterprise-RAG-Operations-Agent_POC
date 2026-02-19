from typing import List, Dict
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore
from backend.generation.llm_provider import OllamaClient

class RAGService:
    """
    Orchestrates Retrieval-Augmented Generation (RAG).
    1. Embeds user query.
    2. Retrieves top-k relevant chunks from Vector Store.
    3. Constructs augmented prompt.
    4. Generates answer using LLM.
    """
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSStore()
        self.llm_client = OllamaClient()

    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieves relevant context chunks for a query.
        """
        query_embedding = self.embedding_model.generate_embedding(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results

    def answer_query(self, query: str, k: int = 3) -> Dict:
        """
        End-to-end RAG pipeline.
        
        Args:
            query (str): The user's question.
            k (int): Number of context chunks to retrieve.
            
        Returns:
            Dict: {
                "answer": str,
                "sources": List[Dict] (metadata of retrieved chunks)
            }
        """
        # 1. Retrieve
        print(f"Retrieving context for: {query}")
        context_results = self.retrieve_context(query, k=k)
        
        if not context_results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "sources": []
            }
            
        # 2. Construct Context String
        # Note: FAISSStore returns a flat dict with metadata + text + score
        context_text = "\n\n".join([f"Source ({r.get('source', 'Unknown')}): {r.get('text', '')}" for r in context_results])
        
        # 3. Construct Prompt
        system_prompt = """You are a helpful AI assistant for an Enterprise RAG system. 
Use the provided context to answer the user's question accurately.
If the answer is not in the context, say you don't know.
Do not hallucinate content not present in the context sources.
"""
        user_prompt = f"""Context information is below:
---------------------
{context_text}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""

        # 4. Generate
        print("Detailed Log: Calling LLM to Generate Answer...")
        try:
            answer = self.llm_client.generate(user_prompt, system_prompt=system_prompt)
            print(f"Detailed Log: LLM Response Received: {answer[:50]}...") # Log first 50 chars
        except Exception as e:
            print(f"Detailed Log: LLM Generation Failed: {e}")
            answer = "Sorry, I encountered an error while generating the response. Please key checking the backend logs."

        if not answer:
            answer = "The LLM returned an empty response. It might be overloaded or crashed."
        
        return {
            "answer": answer,
            "sources": context_results
        }

if __name__ == "__main__":
    # Test RAG Service
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    rag = RAGService()
    response = rag.answer_query("Who is Amir Khan?")
    print("\n=== ANSWER ===")
    print(response["answer"])
    print("\n=== SOURCES ===")
    for source in response["sources"]:
        print(source.get('source'))
