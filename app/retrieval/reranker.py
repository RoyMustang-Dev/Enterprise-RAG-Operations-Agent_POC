"""
Cross-Encoder Semantic Reranker

This module solves the 'Lost in the Middle' problem prevalent in LLMs.
Instead of sending 30 loosely-matched chunks to the expensive 70B model (causing hallucinations/distraction), 
we retrieve 30 chunks rapidly via Vector Math, and then execute a slow, precise Cross-Encoder model locally 
to strictly score the relevance from 0-1, explicitly isolating only the TOP 5 chunks.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Applies BGE-Reranker matrices against candidate pairs.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        Instantiates the PyTorch model onto the local GPU dynamically.
        """
        self.model_name = model_name
        self._model = None
        
        # In a physical deployment, this checks `app.infra.hardware.HardwareProbe`
        # and binds to 'cuda' or 'mps' immediately upon instantiation.
        logger.info(f"[RERANKER] Scaffolded {self.model_name}. Model will lazy-load on first execution.")
        
    @property
    def model(self):
        """Lazy-loads the huggingface Cross-Encoder model directly into active hardware limits."""
        if self._model is None:
            logger.info(f"[RERANKER] Physically allocating memory for Cross-Encoder: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                
                # Check hardware configuration to optimize deployment
                from app.infra.hardware import HardwareProbe
                hw = HardwareProbe.detect_environment()
                device = hw.get("primary_device", "cpu")
                
                self._model = CrossEncoder(self.model_name, device=device)
                logger.info(f"[RERANKER] Successfully bound {self.model_name} to {device} tensor mapping.")
            except ImportError:
                logger.error("[RERANKER] sentence_transformers not installed. Reranking will universally fail.")
                raise
        return self._model
        
    def rerank(self, query: str, context_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Evaluates the linguistic overlap between the exact query and every individual candidate chunk.
        
        Args:
            query (str): The rewritten user prompt.
            context_chunks (List[dict]): The pre-filtered list of up to 30 chunks pulled from Qdrant.
            top_k (int): Exact integer limit for final output delivery to the 70B Synthesis Engine.
            
        Returns:
            List[dict]: The tightly sorted subset of contexts matching exact semantic relevance.
        """
        if not context_chunks:
            return []
            
        # We must align the payloads as a 2D Array mapping [ [Query, Chunk1], [Query, Chunk2] ]
        pairs = [[query, chunk.get("page_content", "")] for chunk in context_chunks]
        
        try:
            # Predict produces an explicit array of numeric values corresponding accurately to input positions
            logger.info(f"[RERANKER] Computing heavy cross-encoder matrix for {len(pairs)} candidate arrays...")
            scores = self.model.predict(pairs)
            
            # Reattach the mathematical floats aggressively to the primary dictionaries
            for i, chunk in enumerate(context_chunks):
                chunk["rerank_score"] = float(scores[i])
                
            # Filter mathematically mapping highest arrays to index 0
            sorted_chunks = sorted(context_chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            # Truncate
            final_chunks = sorted_chunks[:top_k]
            logger.info(f"[RERANKER] Narrowed search space from {len(context_chunks)} -> {len(final_chunks)} exact chunks.")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"[RERANKER] Cross-Encoder execution physically crashed: {e}")
            # Failsafe: Revert to the raw retrieved chunks (already sorted by Cosine Distance)
            logger.warning("[RERANKER] Bypassing re-rank mechanism due to error.")
            return context_chunks[:top_k]
