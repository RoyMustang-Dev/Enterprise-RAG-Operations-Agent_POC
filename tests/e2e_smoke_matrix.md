# E2E Smoke Matrix (Manual/Automatable)

## Health
- `GET /api/v1/health`

## Chat
- `POST /api/v1/chat` with simple query
- `POST /api/v1/chat` with complex query (ensure complexity_score >= 0.8)
- `POST /api/v1/chat` with retrieval returning no context (expect `DATA_NOT_FOUND`)

## Ingestion
- `POST /api/v1/ingest/files`
- `GET /api/v1/progress/{job_id}`
- `POST /api/v1/ingest/crawler`
- `GET /api/v1/ingest/status`

## Feedback
- `POST /api/v1/feedback`
