"""
OpenTelemetry Bootstrap (Optional)

Loads OTel FastAPI instrumentation when OTEL is enabled and dependencies are present.
"""
import os
import logging

logger = logging.getLogger(__name__)


def init_otel(app):
    """
    Initialize OpenTelemetry instrumentation if enabled via env vars.
    """
    enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    if not enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except Exception as e:
        logger.warning(f"[OTEL] Instrumentation not available: {e}")
        return

    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[OTEL] FastAPI instrumentation enabled.")
    except Exception as e:
        logger.warning(f"[OTEL] Failed to instrument app: {e}")
