"""
Query Complexity Classifier

Uses a low-latency 8B model to score query complexity from 0.0 to 1.0.
Falls back to lightweight heuristics if API keys are missing.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


class ComplexityClassifier:
    """
    Returns a strict float score in [0.0, 1.0] representing query complexity.
    """

    def __init__(self, model_override: str = "llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override

        self.system_prompt = """SYSTEM: You are a strict query complexity scorer.
Return EXACTLY one JSON object:
{
  "score": 0.0-1.0,
  "reason": "brief rationale"
}

Scoring guidance:
- 0.0-0.3: short, direct, single-fact queries
- 0.4-0.6: multi-clause or requires synthesis across multiple facts
- 0.7-1.0: multi-hop, comparison, analysis, or ambiguous requests requiring reasoning

Be deterministic. Do not return any extra text."""

    async def score(self, query: str) -> float:
        """
        Returns a strict float score between 0.0 and 1.0.
        """
        if not query:
            return 0.0

        # Fallback heuristic if API key is missing
        if not self.api_key:
            return self._heuristic_score(query)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_completion_tokens": 120,
            "response_format": {"type": "json_object"},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=8,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    raw = data["choices"][0]["message"]["content"]
                    parsed = json.loads(raw)
                    return self._clamp_score(parsed.get("score", 0.0))
        except Exception as e:
            logger.warning(f"[COMPLEXITY] Scoring failed, using heuristic fallback: {e}")
            return self._heuristic_score(query)

    def _heuristic_score(self, query: str) -> float:
        """Simple heuristic fallback based on length and multi-hop cues."""
        q = query.lower()
        word_count = len(q.split())
        multi_hop_flags = ["compare", "contrast", "difference", "analyze", "resolve", "tradeoff", "versus"]
        has_multi_hop = any(flag in q for flag in multi_hop_flags)

        score = 0.2
        if word_count > 25:
            score += 0.2
        if word_count > 50:
            score += 0.2
        if has_multi_hop:
            score += 0.3
        return self._clamp_score(score)

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))
