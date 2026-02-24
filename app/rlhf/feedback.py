"""
RLHF Feedback Ingestion API

Captures explicit binary / continuous human feedback from the Streamlit UI and writes it
persistently. This forms the baseline dataset required for the ultimate RLAIF/RLHF 
Distillation loop. 

A "Thumbs Down" from an enterprise user is the most valuable signal the system can receive.
It mathematically represents an Edge Case the system failed to generalize.
"""
import sqlite3
import os
import logging
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic Schema for the API Layer validation
class FeedbackPayload(BaseModel):
    session_id: str = Field(..., description="Unique ID of the query execution.")
    user_query: str = Field(..., description="The original human text.")
    agent_answer: str = Field(..., description="The raw unformatted generation.")
    feedback_score: int = Field(..., description="1 for Thumbs Up, -1 for Thumbs Down.")
    comment: str = Field(default="", description="Optional human rationale.")

class FeedbackService:
    """
    Manages the physical persistence of human feedback into the SQL layer.
    """
    
    def __init__(self, db_path: str = "data/feedback_loop.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Sets up the SQLite table natively mapping RLHF parameters."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS rlhf_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            agent_answer TEXT,
            feedback_score INTEGER,
            comment TEXT,
            timestamp TEXT
        )
        """)
        conn.commit()
        conn.close()

    def record_feedback(self, payload: FeedbackPayload) -> bool:
        """
        Writes the explicitly recorded human signal to the DB mapping.
        
        Args:
            payload (FeedbackPayload): Validated structural dictionary.
            
        Returns:
            bool: Native mapping success boundary.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            ts = datetime.utcnow().isoformat()
            c.execute("""
            INSERT INTO rlhf_evaluations (session_id, user_query, agent_answer, feedback_score, comment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (payload.session_id, payload.user_query, payload.agent_answer, payload.feedback_score, payload.comment, ts))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[RLHF] Successfully recorded human feedback signal (Score: {payload.feedback_score}) for session {payload.session_id}.")
            return True
            
        except Exception as e:
            logger.error(f"[RLHF] Failed to physically persist the human feedback matrix: {e}")
            return False
