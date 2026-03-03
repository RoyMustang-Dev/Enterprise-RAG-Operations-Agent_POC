# Final Delivery Runbook

This runbook is the client handoff guide for running the system in dev and prod.

## 1) Environment Setup
- Copy `.env-copy` to `.env` and set required keys.
- Ensure `GROQ_API_KEY`, `SARVAM_API_KEY`, `HF_TOKEN` are valid.
- Set `QDRANT_URL` + `QDRANT_API_KEY` for cloud, or leave unset for local storage fallback.

## 2) Start Services (Dev)
1. Start API:
   - `.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Start Celery (explicit dev worker):
   - `.\scripts\start_celery_worker_dev.ps1`
3. Optional: Start full stack shortcut:
   - `.\scripts\start_stack.ps1` with `CELERY_AUTOSTART=true`

## 3) Start Services (Prod)
1. Start API (recommended):
   - Gunicorn + Uvicorn workers in Linux (example):
     - `gunicorn -c gunicorn.conf.py app.main:app`
2. Start Celery (explicit prod worker):
   - `./scripts/start_celery_worker_prod.sh`

## 4) Health Check
- `GET /api/v1/health`

## 5) Core API Smoke Tests
- Chat JSON:
  - `POST /api/v1/chat` with JSON body
- Chat multipart (files):
  - `POST /api/v1/chat` with `files[]`
- Transcribe:
  - `POST /api/v1/transcribe` (WAV/MP3; `STT_BACKEND=groq|local`)
- TTS:
  - `POST /api/v1/tts` (form `text=...` or JSON `{"text":"..."}`)
- Persona Bootstrapper:
  - `POST /api/v1/agents`

## 6) Critical Ops Notes
- First run downloads models (long cold start).
- OCR uses EasyOCR on GPU if CUDA available.
- Vision uses BLIP by default (switch to LLaVA only on high‑VRAM machines).
- If using LLaVA, ensure `VISION_LLAVA_MIN_VRAM_GB` matches available VRAM.
- If Celery is not running, ingestion falls back to local synchronous behavior.
