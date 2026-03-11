import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
api_key = os.getenv("MODELSLAB_API_KEY")

urls_to_test = [
    "https://modelslab.com/api/v1/chat/completions",
    "https://modelslab.com/api/v6/v1/chat/completions",
    "https://modelslab.com/v1/chat/completions",
    "https://modelslab.com/api/v6/chat/completions",
    "https://modelslab.com/api/v7/chat/completions"
]

payload = {
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

for url in urls_to_test:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=5)
        print(f"URL: {url}")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:200]}")
        print("-" * 40)
    except Exception as e:
        print(f"URL: {url} | Error: {e}")
