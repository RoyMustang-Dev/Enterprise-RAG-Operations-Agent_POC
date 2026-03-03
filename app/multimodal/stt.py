import io
import os
import tempfile
import logging
from typing import Optional

from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Local Speech-to-Text pipeline (Whisper via Transformers).
    Uses GPU when available; falls back to CPU gracefully.
    """

    def __init__(self, model_name: Optional[str] = None):
        profile = HardwareProbe.get_profile()
        self.device = profile.get("primary_device", "cpu")
        self.model_name = model_name or os.getenv("STT_MODEL_NAME", "openai/whisper-small")
        self._pipeline = None
        logger.info(f"[STT] Local backend initialized model={self.model_name} device={self.device}")

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except Exception as e:
                raise RuntimeError(f"transformers not installed or failed to import: {e}")

            device_index = 0 if self.device == "cuda" else -1
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device_index,
            )
        return self._pipeline

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty.")
        suffix = os.path.splitext(filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            pipe = self._get_pipeline()
            result = pipe(tmp_path)
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            return str(result)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
