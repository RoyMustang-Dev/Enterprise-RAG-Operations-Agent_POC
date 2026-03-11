import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MODELSLAB_API_KEY")

def test_modelslab_openai_v1():
    url = "https://modelslab.com/api/v6/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 10
    }
    r = requests.post(url, headers=headers, json=payload)
    print("V6 OPENAI/V1:", r.status_code, r.text[:200])

    url2 = "https://modelslab.com/api/v6/chat/completions"
    r2 = requests.post(url2, headers=headers, json=payload)
    print("V6 CHAT:", r2.status_code, r2.text[:200])

if __name__ == "__main__":
    test_modelslab_openai_v1()
