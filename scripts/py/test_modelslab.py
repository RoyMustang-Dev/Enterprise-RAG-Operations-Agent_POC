import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MODELSLAB_API_KEY")

def test_modelslab_v7():
    import json
    url = "https://modelslab.com/api/v7/llm/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Test 1: Native Payload
    payload1 = {
        "key": api_key,
        "model_id": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }
        ]
    }
    r = requests.post(url, headers=headers, json=payload1)
    print("V7 TOOL CALL STATUS:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Raw:", r.text[:500])

if __name__ == "__main__":
    test_modelslab_v7()
