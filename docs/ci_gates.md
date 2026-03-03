# CI Gates (Client Handoff)

## Gate 1 — Lint + Format
- `python -m ruff check .`
- `python -m ruff format .`

## Gate 2 — Unit Smoke
- `python -m pytest -q`

## Gate 3 — API Contract (FastAPI)
- Spin up API in CI container
- Run health check `GET /api/v1/health`
- Run minimal `/chat` JSON request

## Gate 4 — Multimodal Core
- `/chat` with a text file upload
- `/chat` with an image (OCR mode)
- `/transcribe` with small audio
- `/tts` with short text

## Gate 5 — Persona Bootstrapper
- `POST /api/v1/agents` and verify persona cache refresh
- Ensure `[GLOBAL PERSONA INITIATED]` appears in compiled prompts

## Gate 6 — Ingestion (Optional)
- `/ingest/files` with 1 PDF
- `/ingest/status/{job_id}`

## Gate 7 — Eval Suite (Optional)
- Set `RAG_EVAL_DATASET` and run `scripts/run_rag_eval.ps1`

## Exit Criteria
All gates must pass before release. If any fail, block deployment.
