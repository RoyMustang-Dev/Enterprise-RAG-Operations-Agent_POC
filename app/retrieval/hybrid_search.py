"""
Hybrid Search (Dense + Lexical Rerank)

Feature-flagged lexical reranking on top of dense retrieval results.
This is a lightweight hybrid strategy without external dependencies.
"""
import re
from typing import List, Dict, Any


class HybridReranker:
    """
    Applies a simple lexical overlap score to rerank dense results.
    """

    def __init__(self):
        pass

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return chunks

        for chunk in chunks:
            text = chunk.get("page_content", "")
            c_tokens = self._tokenize(text)
            overlap = len(q_tokens.intersection(c_tokens))
            chunk["lexical_score"] = overlap
            # Combine with dense score if present
            dense_score = float(chunk.get("score", 0.0))
            chunk["hybrid_score"] = dense_score + (overlap * 0.01)

        return sorted(chunks, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    def _tokenize(self, text: str) -> set:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return set(tokens)
