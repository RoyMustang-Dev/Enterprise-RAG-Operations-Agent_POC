# Provider Backend Refactor Plan (3 Major Steps)

This is a **plan-only artifact**. It does **not** modify the current Groq backend beyond organizational restructuring.

---

## Major Step 1 — Restructure + Relocation (Keep Groq Working)

### Part 1: Restructure `app/`
- Introduce provider‑scoped folders:
  - `app/groq-core`
  - `app/gemini-core`
  - `app/modelslab-core`
  - `app/anthropic-core`
  - `app/openai-core`

### Part 2: Relocate Groq Backend
- Move Groq-specific files into `app/groq-core`.
- Keep **global** services at root:
  - `app/api`
  - `app/infra`
  - `app/ingestion`
  - `app/multimodal`
  - `app/retrieval`
  - `app/core`

### Part 3: Fix Imports / Wiring Only
- Update imports to new paths.
- No logic changes; preserve runtime behavior.

### Part 4: Full Test Matrix (Groq still active)
Automate tests to validate no regressions.

**A. Ingestion Append**
- Crawler URL: `https://docs.modelslab.com/index-full` (depth=2)
- File Upload: `D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\gemini_models.txt`

**B. Ingestion Overwrite**
- Crawler URL: `https://pinokio.co/docs/#/` (depth=2)
- File Upload: `D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\tests\modelslab-info-test-files\llms-full.txt`

**C. /chat Endpoint**
- Trigger all 12 RAG steps
- Ensure full routing paths

**D. Multimodal**
- Use files in `D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\test-files\new-flow-test`
- Trigger OCR, Vision, and file-based RAG

---

## Major Step 2 — Provider Switching (Pluggable Backends)

### Objective
Enable runtime switching of providers based on API keys.

### Approach
- Build a `ProviderRegistry` at `app/infra/provider_router.py`.
- Registry controls which provider backend handles:
  - core LLM
  - vision
  - audio
  - reranker

### Key Principle
Services like ingestion, crawler, files, and vector DB stay global.
Provider cores only own LLM inference logic.

---

## Major Step 3 — Gemini Backend Buildout

### Part 1: Clone Groq Core → Gemini Core
- Copy `app/groq-core` → `app/gemini-core`

### Part 2: Fix Gemini Implementation
- Apply changes from `gemini_models.txt`
- Ensure `.env` uses Gemini key correctly
- Validate no hardcoded Groq paths remain

### Part 3: Full Regression Matrix (Gemini active)
Repeat the exact tests from Step 1:

**A. Ingestion Append**
- `https://docs.modelslab.com/index-full` (depth=2)
- `gemini_models.txt`

**B. Ingestion Overwrite**
- `https://pinokio.co/docs/#/` (depth=2)
- `llms-full.txt`

**C. /chat Endpoint**
- All 12 RAG steps + routing validation

**D. Multimodal**
- Use `test-files/new-flow-test`

---

## Non‑Negotiable Safety Constraints

1. **Do not change Groq backend after Step 1.**
2. Global services remain shared.
3. Any provider core can be swapped without breaking ingestion/crawler/vector pipelines.
4. All migrations must pass full regression before moving to the next phase.

