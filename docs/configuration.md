# Configuration Flags

## Core
- `GROQ_API_KEY`: Required for inference calls.
- `OPENAI_API_KEY`: Optional for embedding fallback.
- `SARVAM_API_KEY`: Optional for verifier independence.
- `PERSONA_INJECTION_MODE`: `all` (default) or `strict` to limit persona injection stages.
- `PERSONA_MAX_CHARS`: Max characters allowed for full persona injection (default `4000`).
- `PERSONA_PARTIAL_MAX_CHARS`: Max characters for partial injection (default `800`).
- `GUARD_STRICT`: `true|false` to hard-block when guard flags content.
- `GUARD_SOFT_ALLOW`: `true|false` to allow safe OCR/image prompts when guard is noisy.
- `GUARD_SOFT_ALLOW_PHRASES`: Comma-separated allowlist for soft-allow.
- `GUARD_SOFT_ALLOW_RISKY_PHRASES`: Comma-separated blocklist for soft-allow.

## CORS
- `CORS_ALLOW_ORIGINS`: Comma-separated list of allowed origins.
- `CORS_ALLOW_CREDENTIALS`: `true|false`. If true, origins must be explicit.

## Observability
- `OTEL_ENABLED`: `true|false` to enable OpenTelemetry instrumentation.

## Model Cache + Preload
- `MODEL_CACHE_DIR`: Path for HuggingFace/Transformers/SentenceTransformers cache.
- `PRELOAD_MODELS`: `true|false` to preload local models on startup.

## Vision
- `VISION_BACKEND`: `blip`, `llava`, or `auto` (default `blip`).
- `VISION_MODEL_NAME`: Model id for the selected backend.
- `VISION_FALLBACK_MODEL`: Fallback caption model.
- `VISION_ALLOW_FALLBACK`: `true|false` to allow fallback on load errors.
- `VISION_LLAVA_MIN_VRAM_GB`: Minimum VRAM required to keep LLaVA enabled (default `8` GB).

## OCR (EasyOCR)
- `OCR_ENGINE`: Set to `easyocr` (default).
- `OCR_LANG`: OCR language (default `en`).
- `PDF_OCR_FALLBACK`: `true|false` to run OCR when PDFs have little/no extractable text.
- `PDF_OCR_MIN_CHARS`: Minimum extracted characters before OCR fallback triggers.

## Audio (STT/TTS)
- `ENABLE_TRANSCRIBE`: `true|false` to enable `/api/v1/transcribe` (default `true`).
- `STT_BACKEND`: `groq` (default) or `local` (Transformers Whisper pipeline).
- `STT_MODEL_NAME`: Whisper model id for local STT (default `openai/whisper-small`).

## TTS
- `/api/v1/tts` accepts **both** JSON (`{"text": "..."}`) and form-data (`text=...`).

## Retrieval
- `QDRANT_URL`, `QDRANT_API_KEY`: Enable Qdrant Cloud.
- `QDRANT_COLLECTION`: Override collection name.
- `QDRANT_MULTI_TENANT`: `true|false` to use `tenant_id` filter.
- `HYBRID_SEARCH`: `true|false` to enable lexical rerank.
- `RERANKER_ENABLED`: `true|false` to enable cross-encoder reranking.
- `RERANKER_MODEL_NAME`: Override reranker model.
- `RERANK_TOP_K`: Override final top-k after rerank.

## Background Jobs
- `CELERY_ENABLED`: `true|false` to use Celery workers for ingestion/crawl.
- `CELERY_BROKER_URL`: Redis/AMQP broker URL.
- `CELERY_RESULT_BACKEND`: Redis/AMQP backend URL.

## Ephemeral Sessions
- `EPHEMERAL_TTL_HOURS`: Session file collection TTL.
- `EPHEMERAL_CLEANUP_INTERVAL_MINUTES`: Background cleanup interval.
- `MAX_UPLOAD_MB`: Maximum upload size in MB.

## Server
- `GUNICORN_WORKERS`: Worker count for Gunicorn.
- `GUNICORN_BIND`: Bind address.
- `GUNICORN_TIMEOUT`: Request timeout.
