import os
import json
import time
import requests
from pathlib import Path


BASE_URL = os.getenv("PHASE_TEST_BASE_URL", "http://localhost:8000")
TEST_DIR = Path(__file__).resolve().parents[1] / "test-files" / "new-flow-test"


def post_json(path, payload, retries=6):
    delay = 10
    for attempt in range(retries):
        resp = requests.post(f"{BASE_URL}{path}", json=payload, timeout=120)
        if resp.status_code in (200, 201):
            return resp
        if resp.status_code in (429, 500):
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        return resp
    return resp


def post_multipart(path, data=None, files=None, retries=6):
    delay = 10
    for attempt in range(retries):
        resp = requests.post(f"{BASE_URL}{path}", data=data or {}, files=files or [], timeout=240)
        if resp.status_code in (200, 201):
            return resp
        if resp.status_code in (429, 500):
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        return resp
    return resp


def run():
    session = "phase-test-session"
    results = []
    text_only = os.getenv("PHASE_TEST_ONLY_TEXT", "false").lower() == "true"

    # 1) JSON chat (intent/synthesis pipeline)
    results.append(post_json("/api/v1/chat", {
        "query": "Hi there!",
        "chat_history": [],
        "model_provider": "groq",
        "session_id": session,
        "stream": False,
        "reranker_profile": "accurate"
    }).status_code)
    time.sleep(10)

    # 2) metadata query
    results.append(post_json("/api/v1/chat", {
        "query": "Summarize docs from 2024 related to finance.",
        "chat_history": [],
        "model_provider": "groq",
        "session_id": session,
        "stream": False,
        "reranker_profile": "accurate"
    }).status_code)
    time.sleep(10)

    # 3) Coder
    results.append(post_json("/api/v1/chat", {
        "query": "Write a Python script that reads a CSV and computes averages.",
        "chat_history": [],
        "model_provider": "groq",
        "session_id": session,
        "stream": False,
        "reranker_profile": "accurate"
    }).status_code)
    time.sleep(10)

    if text_only:
        ok = all(code == 200 for code in results)
        print("Phase tests OK" if ok else f"Phase tests had non-200: {results}")
        return

    # 4) File upload (txt)
    txt_path = TEST_DIR / "Automaatte.txt"
    with txt_path.open("rb") as f:
        results.append(post_multipart("/api/v1/chat", data={
            "query": "Summarize the Automaatte.txt file.",
            "session_id": "phase-file-txt",
            "image_mode": "auto",
            "reranker_profile": "accurate"
        }, files=[("files", (txt_path.name, f, "text/plain"))]).status_code)

    # 5) File upload (docx)
    docx_path = TEST_DIR / "Implementation-plan.docx"
    with docx_path.open("rb") as f:
        results.append(post_multipart("/api/v1/chat", data={
            "query": "Summarize the Implementation-plan.docx file.",
            "session_id": "phase-file-docx",
            "image_mode": "auto",
            "reranker_profile": "accurate"
        }, files=[("files", (docx_path.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))]).status_code)

    # 6) File upload (pdf)
    pdf_path = TEST_DIR / "list-of-selected-candidates-for-engagement-of-young-professionals-law-gd-i-young-professionals1745234104.pdf"
    with pdf_path.open("rb") as f:
        results.append(post_multipart("/api/v1/chat", data={
            "query": "Summarize the PDF file.",
            "session_id": "phase-file-pdf",
            "image_mode": "auto",
            "reranker_profile": "accurate"
        }, files=[("files", (pdf_path.name, f, "application/pdf"))]).status_code)

    # 7) OCR image
    img_path = TEST_DIR / "mascot2.jpeg"
    with img_path.open("rb") as f:
        results.append(post_multipart("/api/v1/chat", data={
            "query": "Read the text from the image and summarize it.",
            "session_id": "phase-ocr",
            "image_mode": "ocr",
            "reranker_profile": "accurate"
        }, files=[("images", (img_path.name, f, "image/jpeg"))]).status_code)

    # 8) Vision caption
    img2_path = TEST_DIR / "Gemini_Generated_Image_8rdyl8rdyl8rdyl8.png"
    with img2_path.open("rb") as f:
        results.append(post_multipart("/api/v1/chat", data={
            "query": "Describe the image in detail.",
            "session_id": "phase-vision",
            "image_mode": "vision",
            "reranker_profile": "accurate"
        }, files=[("images", (img2_path.name, f, "image/png"))]).status_code)

    # 9) Transcribe wav
    wav_path = TEST_DIR / "emotial-spoken-voice-the-end-of-the-world.wav"
    with wav_path.open("rb") as f:
        results.append(post_multipart("/api/v1/transcribe", files=[("audio_file", (wav_path.name, f, "audio/wav"))]).status_code)

    # 10) Transcribe mp3
    mp3_path = TEST_DIR / "fake-2.mp3"
    with mp3_path.open("rb") as f:
        results.append(post_multipart("/api/v1/transcribe", files=[("audio_file", (mp3_path.name, f, "audio/mpeg"))]).status_code)

    # 11) TTS
    results.append(post_multipart("/api/v1/tts", data={"text": "Hello from phase test."}).status_code)

    ok = all(code == 200 for code in results)
    print("Phase tests OK" if ok else f"Phase tests had non-200: {results}")


if __name__ == "__main__":
    run()
