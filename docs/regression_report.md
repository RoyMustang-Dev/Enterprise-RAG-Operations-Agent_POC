# Regression Test Report

Date: 2026-03-02

## Summary
- API healthy and responsive.
- JSON + multipart `/chat` works.
- OCR + Vision paths validated.
- Persona bootstrapper works and injects global persona across all prompt stages.
- Audio transcription + TTS validated.

## Tests Executed
1. Health
   - `GET /api/v1/health` → 200 OK

2. Smalltalk (JSON)
   - `POST /api/v1/chat` JSON “Hi”
   - Result: `agent_routed=smalltalk` (success)

3. Coder (JSON)
   - `POST /api/v1/chat` JSON “Write Fibonacci”
   - Result: `agent_routed=coder_agent` (success)

4. RAG with TXT upload (multipart)
   - File: `Automaatte.txt`
   - Result: `agent_routed=rag_agent` (success)

5. OCR image (multipart)
   - File: `mascot2.jpeg`, `image_mode=ocr`
   - Result: OCR extracted text and summarized (success)

6. Vision image (multipart)
   - File: `Gemini_Generated_Image_8rdyl8rdyl8rdyl8.png`, `image_mode=vision`
   - Result: BLIP caption path used (success)

7. DOCX upload (multipart)
   - File: `Implementation-plan.docx`
   - Result: summary returned (success)

8. PDF upload (multipart)
   - File: `list-of-selected-candidates-...pdf`
   - Result: OCR fallback works (success)

9. Transcribe
   - File: `fake-2.mp3`
   - Result: transcript returned (success)

10. TTS
    - `POST /api/v1/tts` with form text
    - Result: audio file generated (success)

11. Persona Bootstrapper
    - `POST /api/v1/agents` with bot details
    - Result: success
    - Verified `[GLOBAL PERSONA INITIATED]` in all compiled prompt stages.

## Notes
- First run may download BLIP/EasyOCR weights.
- Vision on BLIP is lightweight and stable for 4GB VRAM.

