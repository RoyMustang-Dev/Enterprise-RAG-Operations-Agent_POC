"""
Central Observability & Telemetry Service

This file is responsible for recording all structural events natively as they traverse the DAG.
It creates a rigid JSONL audit file, fulfilling the Enterprise prerequisite for tracing 
precisely *why* an LLM answered a specific question.
"""
import os
import json
import logging
from datetime import datetime
from threading import Thread
from typing import Optional, Dict, Any

from app.core.types import TelemetryLogRecord

# Set up module-level diagnostic logger for terminal output
logger = logging.getLogger(__name__)

class ObservabilityLayer:
    """
    Singleton-pattern class managing non-blocking writes to the `audit_logs.jsonl` pipeline.
    This class ensures that enterprise metric collection never blocks the `async` FASTApi thread.
    """
    
    _instance = None
    
    def __new__(cls, log_path: str = "data/audit/audit_logs.jsonl"):
        """
        Instantiates exactly one writer queue per Uvicorn worker to prevent file-locking.
        """
        if cls._instance is None:
            cls._instance = super(ObservabilityLayer, cls).__new__(cls)
            cls._instance._initialize(log_path)
        return cls._instance
        
    def _initialize(self, log_path: str):
        """
        Validates target OS path directories cleanly on boot.
        
        Args:
            log_path (str): The physical disk target for the Enterprise append-only tracking file.
        """
        self.log_path = os.path.abspath(log_path)
        log_dir = os.path.dirname(self.log_path)
        
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Initialized new Audit log directory at {log_dir}")
            except Exception as e:
                logger.error(f"FATAL: Telemetry core failed to create audit directory -> {e}")
                raise SystemExit(1)
        
        # Ensure the file exists so the first JSON.dumps does not trigger a missing resource crash
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'a'): pass
            
    def emit(self, record: TelemetryLogRecord):
        """
        Receives a strictly typed Pydantic record and fires a detached thread 
        to execute the physical disk IO asynchronously.
        
        Args:
            record (TelemetryLogRecord): The validated execution mapping state.
        """
        def _write_async():
            try:
                payload = record.model_dump()
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception as e:
                # Intercept catastrophic failures silently so the API response still returns to the user
                logger.error(f"Telemetry Write Failure (Session {record.session_id}): {e}")
                
        Thread(target=_write_async, daemon=True).start()
        
    @staticmethod
    def get_timestamp() -> str:
        """Helper generating precise UTC tracking markers."""
        return datetime.utcnow().isoformat()
