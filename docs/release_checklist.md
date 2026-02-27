# Release Checklist

## Regression Matrix
- `/api/v1/health` returns 200 with expected model metadata.
- `/api/v1/chat` basic query returns non-empty `answer`.
- `/api/v1/chat` returns `DATA_NOT_FOUND` when no context is available.
- `/api/v1/ingest/files` accepts upload and returns `job_id`.
- `/api/v1/progress/{job_id}` returns status for active job.
- `/api/v1/ingest/crawler` returns `job_id`.
- `/api/v1/ingest/status` returns vector stats.
- `/api/v1/feedback` accepts payload.

## Performance Smoke
- Latency: chat p95 under agreed SLA.
- Retrieval: top_k rerank returns <= 5 chunks.
- Embedding cache hit on repeated query.

## Observability
- Telemetry logs include: `complexity_score`, `metadata_filters_applied`, `reward_score`.
- If `OTEL_ENABLED=true`, traces are emitted.

## Security
- CORS allows only configured origins if `CORS_ALLOW_CREDENTIALS=true`.
- Prompt guard returns block on malicious prompt.

## RAG Quality
- Verification loop triggers on hallucination.
- Reward score computed and stored in telemetry.
