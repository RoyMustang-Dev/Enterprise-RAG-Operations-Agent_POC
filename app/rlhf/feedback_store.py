"""
RLHF Telemetry Storage Engine

This module is responsible for physically capturing human feedback loops (Thumbs Up / Thumbs Down + Text)
and saving them securely into physical disk mappings.
This data is used later to actively fine-tune the RAG logic or prompt structures.
"""
import os
import json
import logging
from datetime import datetime
from threading import Thread
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FeedbackStore:
    """
    Singleton-pattern class managing non-blocking writes to the `rlhf_feedback.jsonl` pipeline.
    """
    _instance = None
    
    def __new__(cls, log_path: str = "data/audit/rlhf_feedback.jsonl"):
        """
        Instantiates exactly one writer queue per Uvicorn worker to prevent file-locking.
        """
        if cls._instance is None:
            cls._instance = super(FeedbackStore, cls).__new__(cls)
            cls._instance._initialize(log_path)
        return cls._instance
        
    def _initialize(self, log_path: str):
        self.log_path = os.path.abspath(log_path)
        log_dir = os.path.dirname(self.log_path)
        
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"FATAL: Feedback store failed to create directory -> {e}")
                
        # Ensure the file exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'a'): pass
            
    def record_feedback(self, session_id: str, rating: str, feedback_text: str = "", metadata: Dict[str, Any] = None):
        """
        Accepts raw telemetry from the FastAPI layer and pushes it to disk asynchronously.
        
        Args:
            session_id (str): The specific conversation UUID.
            rating (str): 'thumbs_up' or 'thumbs_down'.
            feedback_text (str): Optional human-typed description.
            metadata (Dict): Any additional structural tags.
        """
        def _write_async():
            try:
                payload = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session_id,
                    "rating": rating,
                    "feedback_text": feedback_text,
                    "metadata": metadata or {}
                }
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
                logger.info(f"[RLHF] Successfully recorded human feedback for session {session_id}.")
            except Exception as e:
                logger.error(f"[RLHF] Write Failure (Session {session_id}): {e}")
                
        Thread(target=_write_async, daemon=True).start()
