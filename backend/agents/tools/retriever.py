import os
import json
from typing import List, Dict, Any
import logging

class RetrieverTool:
    """
    Standalone module responsible for Hybrid Hybrid / Vector search.
    Abstracts FAISS operations from the orchestration layer.
    """
    
    def __init__(self, faiss_store, embedding_model, top_k=20, final_k=5):
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.final_k = final_k
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes semantic search over the vector store.
        (BM25 Hybrid logic will be injected here later in Phase 3)
        """
        try:
            logging.info(f"RetrieverTool: Generating embedding for '{query}'")
            query_embedding = self.embedding_model.generate_embedding(query)
            
            # 1. Base Retrieval
            initial_results = self.faiss_store.search(query_embedding, k=self.top_k)
            if not initial_results:
                return []
                
            # 2. Source Filtering (Filename matching heuristic)
            mentioned_files = [f for f in ["AAICLAS1161721280920.pdf", "Adi_Resume_N.pdf", "content.txt"] if f.lower() in query.lower()]
            filtered_results = initial_results
            
            if mentioned_files:
                logging.info(f"RetrieverTool: Filtering results to explicitly mentioned files: {mentioned_files}")
                filtered_results = [r for r in initial_results if r.get("source", "") in mentioned_files]
                if not filtered_results:
                     logging.warning(f"RetrieverTool: No chunks found in {mentioned_files}, falling back to all.")
                     filtered_results = initial_results
                     
            # 3. Context Limiter (Take top K to prevent context overflow)
            final_chunks = filtered_results[:self.final_k]
            
            # We strip embeddings visually from the returned state chunks
            for c in final_chunks:
                if "embedding" in c:
                    del c["embedding"]
                    
            return final_chunks
            
        except Exception as e:
            logging.error(f"RetrieverTool failed: {e}")
            return []
