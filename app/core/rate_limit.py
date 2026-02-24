"""
Token Bucket Rate Limiter Middleware

Enterprise applications must prevent abusive users from rapidly draining the API limits.
This module checks incoming requests against an in-memory or Redis-backed bucket.
"""
import time
import logging
from typing import Dict, Tuple
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class TokenBucketRateLimiter:
    """
    Standard implementation of the Token Bucket algorithm.
    Each unique user/IP is mapped to a bucket filled with N tokens.
    Generating a response costs 1 token. Tokens refill linearly over time.
    """
    
    def __init__(self, capacity: int = 10, refill_rate_per_second: float = 0.5):
        """
        Args:
            capacity (int): Maximum burst amount of requests per client.
            refill_rate_per_second (float): How many tokens generate per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate_per_second
        
        # State: { "client_id": (tokens_remaining, last_refill_timestamp) }
        # WARNING: In production, switch this Dict to a distributed Redis Hash immediately.
        self._buckets: Dict[str, Tuple[float, float]] = {}
        
    def _refill(self, client_id: str):
        """Calculates time elapsed and adds tokens mathematically."""
        now = time.time()
        
        # New client initialization
        if client_id not in self._buckets:
            self._buckets[client_id] = (self.capacity, now)
            return
            
        tokens, last_refill = self._buckets[client_id]
        elapsed = now - last_refill
        
        # Generate new tokens, capping at the physical `capacity` limit
        new_tokens = int(elapsed * self.refill_rate)
        if new_tokens > 0:
            tokens = min(self.capacity, tokens + new_tokens)
            self._buckets[client_id] = (tokens, now)
            
    def consume(self, client_id: str, cost: int = 1) -> bool:
        """
        Attempts to subtract a specific amount of tokens from the bucket.
        
        Args:
            client_id (str): The identifier (e.g. Session ID or IP address).
            cost (int): How much weight to pull from the bucket.
            
        Raises:
            HTTPException (429): If the client is depleted.
            
        Returns:
            bool: True if authorized to proceed.
        """
        # 1. Update the bucket math natively
        self._refill(client_id)
        
        # 2. Extract state
        tokens, last_refill = self._buckets[client_id]
        
        # 3. Validation Logic
        if tokens >= cost:
            self._buckets[client_id] = (tokens - cost, last_refill)
            return True
            
        logger.warning(f"[RATE LIMIT] Client {client_id} exhausted bucket. Returning 429.")
        raise HTTPException(
            status_code=429, 
            detail="You're moving too fast. Generating enterprise responses requires heavy compute. Try again shortly."
        )
