# Configuration Flags

## Core
- `GROQ_API_KEY`: Required for inference calls.
- `OPENAI_API_KEY`: Optional for embedding fallback.
- `SARVAM_API_KEY`: Optional for verifier independence.

## CORS
- `CORS_ALLOW_ORIGINS`: Comma-separated list of allowed origins.
- `CORS_ALLOW_CREDENTIALS`: `true|false`. If true, origins must be explicit.

## Observability
- `OTEL_ENABLED`: `true|false` to enable OpenTelemetry instrumentation.

## Retrieval
- `QDRANT_URL`, `QDRANT_API_KEY`: Enable Qdrant Cloud.
- `QDRANT_COLLECTION`: Override collection name.
- `QDRANT_MULTI_TENANT`: `true|false` to use `tenant_id` filter.
- `HYBRID_SEARCH`: `true|false` to enable lexical rerank.

## Server
- `GUNICORN_WORKERS`: Worker count for Gunicorn.
- `GUNICORN_BIND`: Bind address.
- `GUNICORN_TIMEOUT`: Request timeout.
